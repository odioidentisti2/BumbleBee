import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from attention import SAB, PMA
from molecule_dataset import MoleculeDataset

class MAGClassifier(nn.Module):

    @staticmethod
    # Return a boolean edge adjacency mask
    def _edge_adjacency(source, target):
        # stack and slice
        src_trg = torch.stack([source, target], dim=1) # [num_edges, 2]
        src_nodes = src_trg[:, 0:1]  # [num_edges, 1]
        trg_nodes = src_trg[:, 1:2]  # [num_edges, 1]
        # Create adjacency mask: edges are adjacent if they share a node
        src_adj = (src_nodes == src_nodes.T) | (src_nodes == trg_nodes.T)
        trg_adj = (trg_nodes == src_nodes.T) | (trg_nodes == trg_nodes.T)
        return src_adj | trg_adj  # [num_edges, num_edges]

    def _compute_graph_embeddings(self, edge_repr, src, trg, edge_batch, batch):
        """
        For each graph in the batch, compute the graph embedding using edge representations and attention masks.

        Args:
            edge_repr (Tensor): [total_edges, 2*hidden_dim] edge representations.
            src (Tensor): [total_edges] source node indices for each edge.
            trg (Tensor): [total_edges] target node indices for each edge.
            edge_batch (Tensor): [total_edges] batch indices for each edge.
            batch (Tensor): [total_nodes] batch indices for each node.

        Returns:
            Tensor: Graph embeddings of shape [num_graphs, 2*hidden_dim].
        """
        num_graphs = batch.max().item() + 1
        out = torch.zeros((num_graphs, self.hidden_dim * 2), device=edge_repr.device)
        for g in range(num_graphs):
            mask_g = (edge_batch == g)
            edge_repr_g = edge_repr[mask_g]  # [num_edges_g, 2*hidden_dim]
            src_g = src[mask_g]
            dst_g = trg[mask_g]
            # Compute edge-edge attention mask
            adj_mask = self._edge_adjacency(src_g, dst_g)
            # attn_mask = self._edge_adjacency_CPU(src_g, dst_g)
            adj_mask = adj_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, num_edges_g, num_edges_g]
            # Pass through attention blocks
            edge_repr_g = edge_repr_g.unsqueeze(0)
            for att_layer in self.encoder:
                if att_layer.to_be_masked:
                    edge_repr_g = att_layer(edge_repr_g, adj_mask=adj_mask)
                else:
                    edge_repr_g = att_layer(edge_repr_g)

            # PMA pooling
            pooled = self.pma(edge_repr_g)
            out[g] = pooled.squeeze(0).squeeze(0)  # [2*hidden_dim]
        return out
    
    def __init__(self, node_dim, edge_dim, hidden_dim=128, num_heads=8, num_inds=32, output_dim=1):
        super(MAGClassifier, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_inds = num_inds
        self.output_dim = output_dim

        # Node and edge encoders
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        # ESA encoder
        layer_types = ['M', 'M', 'S']  # M for masked, S for self-attention
        # Always use nn.ModuleList (or nn.Sequential) for lists of layers in PyTorch!
        self.encoder = nn.ModuleList()
        for layer_type in layer_types:
            layer = SAB(hidden_dim * 2, hidden_dim * 2, num_heads, dropout=0.1)
            if layer_type == 'M':
                layer.to_be_masked = True
            elif layer_type == 'S':
                layer.to_be_masked = False
            self.encoder.append(layer)

        # PMA pooling (for each graph in the batch)
        self.pma = PMA(hidden_dim * 2, num_heads, num_seeds=1, dropout=0.1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        # data: batch from DataLoader (torch_geometric.data.Batch)
        x = data.x                  # [total_nodes, node_dim]
        edge_index = data.edge_index # [2, total_edges]
        edge_attr = data.edge_attr   # [total_edges, edge_dim]
        batch = data.batch           # [total_nodes]
        edge_batch = self._edge_batch(edge_index, batch) # [total_edges]

        # Encode node and edge features
        node_feat = self.node_encoder(x)                # [total_nodes, hidden_dim]
        edge_feat = self.edge_encoder(edge_attr)        # [total_edges, hidden_dim]

        # Build edge representation: [src_node, dst_node] features + edge features
        src, dst = edge_index                           # each [total_edges]
        edge_nodes = torch.cat([node_feat[src], node_feat[dst]], dim=1)  # [total_edges, 2*hidden_dim]
        edge_repr = edge_nodes + torch.cat([edge_feat, edge_feat], dim=1)

        graph_repr = self._compute_graph_embeddings(edge_repr, src, dst, edge_batch, batch)  # [batch_size, 2*hidden_dim]
        logits = self.classifier(graph_repr)    # [batch_size, output_dim]
        return logits.view(-1)                  # [batch_size]

    @staticmethod
    def _edge_batch(edge_index, node_batch):
        # Given edge_index [2, num_edges] and node_batch [num_nodes], 
        # return edge_batch [num_edges] where each edge takes the batch idx of its src node
        # (assumes all edges within a graph)
        return node_batch[edge_index[0]]


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
    import time
    start_time = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        loss, acc = train(model, loader, optimizer, criterion, epoch)
        print(f"Epoch {epoch}: Loss {loss:.4f} Acc {acc:.4f} Time {time.time() - start_time:.2f}s")
    print("Training complete.")

if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # DEVICE = torch.device('cpu')
    print(f"\nDEVICE: {DEVICE}")
    BATCH_SIZE = 64
    LR = 1e-4
    NUM_EPOCHS = 20 
    # Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    main()