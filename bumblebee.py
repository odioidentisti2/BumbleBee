import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from molecule_dataset import MoleculeDataset
from attention import SAB, PMA


class ESAMoleculeClassifier(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=128, num_heads=8, num_inds=32, output_dim=1):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.edge_set_proj = nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim)
        self.encoder = nn.Sequential(
            SAB(hidden_dim, hidden_dim, num_heads),
            SAB(hidden_dim, hidden_dim, num_heads),
            SAB(hidden_dim, hidden_dim, num_heads)
        )
        self.pooling = PMA(hidden_dim, num_heads, num_inds)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_inds, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        # data: torch_geometric Batch with .x, .edge_index, .edge_attr, .batch
        h = self.node_proj(data.x)  # [total_nodes, hidden_dim]
        e = self.edge_proj(data.edge_attr)  # [total_edges, hidden_dim]

        # Compute edge features for all edges in the batch
        src, dst = data.edge_index  # [2, total_edges]
        edge_inputs = torch.cat([h[src], h[dst], e], dim=1)  # [total_edges, 3*hidden_dim]
        edge_feats = self.edge_set_proj(edge_inputs)  # [total_edges, hidden_dim]

        # Group edges per graph in the batch
        # Build a batch index for edges using the batch assignment of source nodes
        edge_batch = data.batch[src]  # [total_edges]

        # Find the max number of edges in any graph in the batch
        num_graphs = data.num_graphs
        counts = torch.bincount(edge_batch, minlength=num_graphs)
        max_edges = counts.max().item()

        # Pad edge features per graph to shape [batch, max_edges, hidden_dim]
        E = torch.zeros(num_graphs, max_edges, edge_feats.size(1), device=edge_feats.device)
        adj_mask = torch.zeros(num_graphs, max_edges, dtype=torch.bool, device=edge_feats.device)
        idxs = torch.zeros(num_graphs, dtype=torch.long, device=edge_feats.device)
        for i in range(edge_feats.size(0)):
            g = edge_batch[i].item()
            pos = idxs[g]
            E[g, pos] = edge_feats[i]
            adj_mask[g, pos] = 1
            idxs[g] += 1

        # MAG encoder and pooling (batch)
        for layer in self.encoder:
            E = layer(E, adj_mask=adj_mask)
        pooled = self.pooling(E, adj_mask=adj_mask)
        logits = self.classifier(pooled.view(pooled.size(0), -1)).squeeze(-1)
        return logits

def train(model, loader, optimizer, criterion, epoch):
    model.train()  # set training mode
    total_loss = 0
    correct = 0
    total = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        targets = batch.y.view(-1).to(DEVICE)
        optimizer.zero_grad()  # zero gradients
        logits = model(batch)  # forward pass
        loss = criterion(logits, targets)  # calculate loss
        loss.backward()  # backward pass
        optimizer.step()  # update weights

        # calculate accuracy
        total_loss += loss.item() * batch.num_graphs
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == targets).sum().item()
        total += batch.num_graphs
    return total_loss / total, correct / total

def main():
    dataset = MoleculeDataset('DATASETS/MUTA_SARPY_4204.csv')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = ESAMoleculeClassifier(dataset.node_dim, dataset.edge_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(1, NUM_EPOCHS + 1):
        loss, acc = train(model, loader, optimizer, criterion, epoch)
        print(f"Epoch {epoch}: Loss {loss:.4f} Acc {acc:.4f}")
    print("Training complete.")

if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDEVICE: {DEVICE}")
    BATCH_SIZE = 32
    LR = 1e-4
    NUM_EPOCHS = 20 
    # Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    main()