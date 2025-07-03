import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from molecule_dataset import MoleculeDataset
from attention import SAB, PMA


class MAGClassifier(nn.Module):
    """
    Implements the MAG (Masked Attention for Graphs) model.
    - Graphs are represented as sets of edges (with edge and node features).
    - Uses attention-only (no message passing).
    - Attention is masked according to adjacency matrix.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim=128, num_heads=8, num_inds=32, output_dim=1):
        super(MAGClassifier, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_inds = num_inds
        self.output_dim = output_dim

        # Project node features for edge construction
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        # Project edge features
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        # Masked Attention Blocks for edge set (see "edge masking algorithm")
        self.edge_attention1 = SAB(hidden_dim*2, hidden_dim*2, num_heads, dropout=0.1)
        self.edge_attention2 = SAB(hidden_dim*2, hidden_dim*2, num_heads, dropout=0.1)

        # Pool the set of edge features to a graph-level embedding
        self.pma = PMA(hidden_dim*2, num_heads, num_seeds=1, dropout=0.1)

        # Output classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        # data: a batch from MoleculeDataset (torch_geometric.data.Data)
        # data.x: [num_nodes, node_dim]
        # data.edge_index: [2, num_edges]
        # data.edge_attr: [num_edges, edge_dim]

        x = data.x                          # [num_nodes, node_dim]
        edge_index = data.edge_index        # [2, num_edges]
        edge_attr = data.edge_attr          # [num_edges, edge_dim]

        # 1. Encode node and edge features
        node_feat = self.node_encoder(x)                # [num_nodes, hidden_dim]
        edge_feat = self.edge_encoder(edge_attr)        # [num_edges, hidden_dim]

        # 2. Represent each edge as concatenation of its (masked) node features and edge features
        src, dst = edge_index                          # each [num_edges]
        edge_nodes = torch.cat([node_feat[src], node_feat[dst]], dim=1)  # [num_edges, 2*hidden_dim]
        edge_input = torch.cat([edge_nodes, edge_feat], dim=1)           # [num_edges, 2*hidden_dim + hidden_dim]
        # But for MAG, we want [src_node, dst_node] as single vector, optionally concatenate edge_feat.
        edge_repr = torch.cat([node_feat[src], node_feat[dst]], dim=1)   # [num_edges, 2*hidden_dim]
        edge_repr = edge_repr + torch.cat([edge_feat, edge_feat], dim=1) # add edge_feat to both ends

        # 3. Prepare attention mask for the edge set
        # The mask ensures that attention between edges is allowed only if the two edges share a node
        num_edges = edge_repr.size(0)
        # Build [num_edges, num_edges] mask: mask[i, j] = 1 if edge i and edge j share a node
        src_dst = torch.stack([src, dst], dim=1)  # [num_edges, 2]
        edge_mask = torch.zeros((num_edges, num_edges), dtype=torch.bool, device=edge_repr.device)
        for i in range(num_edges):
            for j in range(num_edges):
                # If two edges share a node, mask is True (allow attention)
                if len(set(src_dst[i].tolist()) & set(src_dst[j].tolist())) > 0:
                    edge_mask[i, j] = True
        # In SAB, mask shape is [batch, num_heads, seq_len, seq_len]; add batch and head dims
        edge_mask = edge_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, num_edges, num_edges]

        # 4. Process edge set with attention blocks (masked)
        # Add batch dimension: [1, num_edges, 2*hidden_dim]
        edge_repr = edge_repr.unsqueeze(0)
        edge_repr = self.edge_attention1(edge_repr, adj_mask=edge_mask)
        edge_repr = self.edge_attention2(edge_repr, adj_mask=edge_mask)
        # Remove batch: [num_edges, 2*hidden_dim]
        edge_repr = edge_repr.squeeze(0)

        # 5. Pool edge set to graph representation (PMA, with mask)
        # PMA expects [batch, seq_len, feature_dim]
        edge_repr = edge_repr.unsqueeze(0)  # [1, num_edges, 2*hidden_dim]
        graph_repr = self.pma(edge_repr, adj_mask=None)  # [1, num_seeds=1, 2*hidden_dim]
        graph_repr = graph_repr.squeeze(0).squeeze(0)    # [2*hidden_dim]

        # 6. Classifier
        out = self.classifier(graph_repr)                # [output_dim]
        return out.view(-1)                              # Match [batch_size] shape for BCEWithLogitsLoss


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
    model = MAGClassifier(dataset.node_dim, dataset.edge_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(1, NUM_EPOCHS + 1):
        loss, acc = train(model, loader, optimizer, criterion, epoch)
        print(f"Epoch {epoch}: Loss {loss:.4f} Acc {acc:.4f}")
    print("Training complete.")

if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDEVICE: {DEVICE}")
    BATCH_SIZE = 1
    LR = 1e-4
    NUM_EPOCHS = 20 
    # Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    main()