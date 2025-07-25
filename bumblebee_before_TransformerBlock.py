import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from attention_before_TransformerBlock import SelfAttention, PMA
from molecule_dataset import GraphDataset

class MAGClassifier(nn.Module):

    @staticmethod
    # Return a boolean edge adjacency mask
    def _edge_adjacency(source, target):
        # stack and slice
        src_dst = torch.stack([source, target], dim=1) # [num_edges, 2]
        src_nodes = src_dst[:, 0:1]  # [num_edges, 1]
        dst_nodes = src_dst[:, 1:2]  # [num_edges, 1]
        # Create adjacency mask: edges are adjacent if they share a node
        src_adj = (src_nodes == src_nodes.T) | (src_nodes == dst_nodes.T)
        dst_adj = (dst_nodes == src_nodes.T) | (dst_nodes == dst_nodes.T)
        return src_adj | dst_adj  # [num_edges, num_edges]
    
    def __init__(self, node_dim, edge_dim, hidden_dim=128, num_heads=8, output_dim=1):
        super(MAGClassifier, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.output_dim = output_dim

        # Edge feature encoder (input MLP)
        self.node_edge_mlp = nn.Sequential(
            nn.Linear(2 * self.node_dim + self.edge_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ESA BLOCK :
        layer_types = 'MMSP'
        # Specify the number and order of layers:
        #   S for self-attention (SAB) 
        #   M for masked SAB (MSAB)
        #   P for the PMA decoder 
        # S and M layers can be alternated in any order as desired. 
        # For graph-level tasks, there must be a single P layer specified. 
        # The P layer can be followed by S layers (decoder), but not by M layers.
        # Always use nn.ModuleList (or nn.Sequential) for lists of layers in PyTorch!

        # Encoder
        enc_layers = layer_types[:layer_types.index('P')]
        self.encoder = nn.ModuleList()
        for type in enc_layers:
            assert type in 'MS'
            self.encoder.append(nn.LayerNorm(hidden_dim, eps=1e-8))
            self.encoder.append(SelfAttention(hidden_dim, hidden_dim, num_heads))
                                    # to_be_masked=(type == 'M')))
            self.encoder[-1].to_be_masked = (type == 'M')

        # Decoder
        dec_layers = layer_types[layer_types.index('P') + 1:]
        assert set(dec_layers).issubset({'S'})  # debug
        self.decoder = nn.Sequential(nn.LayerNorm(hidden_dim, eps=1e-8), PMA(hidden_dim, num_heads))
        for type in dec_layers:
            assert type == 'S'
            self.decoder.append(nn.LayerNorm(hidden_dim, eps=1e-8))
            self.decoder.append(SelfAttention(hidden_dim, hidden_dim, num_heads))

        # Classifier (output MLP)
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(), 
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        """
        Args:
            data: batch from DataLoader (torch_geometric.data.Batch)
                data.x              # [batch_nodes, node_dim]
                data.edge_index     # [2, batch_edges]
                data.edge_attr      # [batch_edges, edge_dim]
                data.batch          # [batch_nodes]
        """
        # PER-BATCH: Build edge set embedding
        src, dst = data.edge_index                           # each [batch_edges]
        # Concatenate node (src and dst) and edge features
        edge_feat = torch.cat([data.x[src], data.x[dst], data.edge_attr], dim=1)
        batched_h = self.node_edge_mlp(edge_feat)  # [batch_edges, hidden_dim]

        # PER-GRAPH: Attention
        batch_size = data.batch.max().item() + 1
        edge_batch = self._edge_batch(data.edge_index, data.batch)  # [batch_edges]
        out = torch.zeros((batch_size, self.hidden_dim), device=edge_feat.device)
        for i in range(batch_size):
            graph_mask = (edge_batch == i)
            # Compute edge-edge adjacency mask
            adj_mask = self._edge_adjacency(src[graph_mask], dst[graph_mask])
            adj_mask = adj_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, graph_edges, graph_edges]
            # Encoder
            h = batched_h[graph_mask].unsqueeze(0)  # [1, graph_edges, hidden_dim]
            for layer in self.encoder:
                if getattr(layer, "to_be_masked", False):
                # if layer.to_be_masked:
                    h = layer(h, adj_mask=adj_mask)  # Masked Self-Attention Block
                else:
                    h = layer(h)  #  Self-Attention Block or LayerNorm
            # Decoder
            pooled = self.decoder(h)  # PMA Block
            out[i] = pooled.squeeze(0).mean(dim=0)  # [hidden_dim] (aggregating seeds by mean)

        logits = self.output_mlp(out)    # [batch_size, output_dim]
        return torch.flatten(logits)     # [batch_size]
        # return logits.view(-1)           # [batch_size]

    @staticmethod
    def _edge_batch(edge_index, node_batch):
        # Given edge_index [2, num_edges] and node_batch [num_nodes], 
        # return edge_batch [num_edges] where each edge takes the batch idx of its src node
        # (assumes all edges within a graph)
        return node_batch[edge_index[0]]

# trainer.fit
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
    dataset = GraphDataset('DATASETS/MUTA_SARPY_4204.csv')
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
    ## ESA
    # HIDDEN_DIM = 256  # = MLP_hidden = graph_dim
    # BATCH_SIZE = 128
    # NUM_HEADS = 16
    # LAYER_TYPES = ['MSMSMP']
    # DROPOUT = 0  # mlp?
    # Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    main()