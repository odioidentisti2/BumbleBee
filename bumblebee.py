import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from architectures import ESA, mlp
from molecular_data import GraphDataset
from adj_mask_utils import *
from depict import *


class MAGClassifier(nn.Module):
    
    def __init__(self, node_dim, edge_dim, hidden_dim=128, mlp_hidden_dim=128, num_heads=8, output_dim=1):
        super(MAGClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        # Edge feature encoder (node-edge MLP)
        self.input_mlp = mlp(2 * node_dim + edge_dim, mlp_hidden_dim, hidden_dim)
        # ESA BLOCK
        self.esa = ESA(hidden_dim, num_heads, 'MMSP')
        # Classifier
        self.output_mlp = mlp(hidden_dim, mlp_hidden_dim, output_dim)

    def forward(self, data, return_attention=False):
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
        batched_h = self.input_mlp(edge_feat)  # [batch_edges, hidden_dim]

        batch_size = data.batch.max().item() + 1  # data.num_graphs
        edge_batch = self._edge_batch(data.edge_index, data.batch)  # [batch_edges]

        device = edge_feat.device
        if device.type == 'cuda':  # GPU: batch Attention
            print("Using GPU attention")
            max_edges = max([g.num_edges for g in data.to_data_list()])
            dense_batch_h, _ = to_dense_batch(batched_h, edge_batch, fill_value=0, max_num_nodes=max_edges)
            adj_mask = edge_mask(data.edge_index, data.batch, data.num_graphs, max_edges)
            out = self.esa(dense_batch_h, adj_mask, return_attention=return_attention)  # [batch_size, hidden_dim]

        if True:  # CPU: per-graph Attention
            print("Using CPU attention")
            # Initialize output tensor
            _out = torch.zeros((batch_size, self.hidden_dim), device=edge_feat.device)
            attn_weights = []
            for i in range(batch_size):
                graph_mask = (edge_batch == i)
                h = batched_h[graph_mask]  # [graph_edges, hidden_dim]
                # Compute edge-edge adjacency mask
                adj_mask = edge_adjacency(src[graph_mask], dst[graph_mask])  # [graph_edges, graph_edges]
                h = h.unsqueeze(0)  # Add batch dimension
                adj_mask = adj_mask.unsqueeze(0)  # Add batch dimension
                if return_attention:
                    _out[i], attn = self.esa(h, adj_mask, return_attention)
                    attn = attn.squeeze(0)  # Remove batch dimension
                    _out[i] = _out[i].squeeze(0)  # Remove batch dimension
                    attn_weights.append(attn)
                    # Per-graph edge_index
                    graph_edges = data.edge_index[:, graph_mask]
                    graph_nodes = (data.batch == i).nonzero(as_tuple=True)[0]
                    offset = graph_nodes[0].item()
                    graph_edge_index = graph_edges - offset  # subtracting the offset (1st node index)
                    # depict(data.mol[i], attn, graph_edge_index)
                else:
                    _out[i] = self.esa(h, adj_mask, return_attention)  # [hidden_dim]
        if device.type == 'cuda':
            print('TEST:', out == _out)
            print(out.shape, _out.shape)
            print(out, _out)
        else:
            out = _out

        logits = self.output_mlp(out)    # [batch_size, output_dim]
        # if return_attention:
        #     return torch.flatten(logits), attn_weights
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
    total = 0
    batch_num = 0
    for batch in loader:
        batch_num += 1
        batch = batch.to(DEVICE)
        targets = batch.y.view(-1).to(DEVICE)
        optimizer.zero_grad()  # zero gradients
        # if batch_num == 1:
            # logits, batch_attention = model(batch, return_attention=True)  # forward pass
        # else:
        #     logits = model(batch)  # forward pass
        logits = model(batch)  # forward pass
        loss = criterion(logits, targets)  # calculate loss
        loss.backward()  # backward pass
        optimizer.step()  # update weights
        # calculate statistics
        total_loss += loss.item() * batch.num_graphs
        total += batch.num_graphs
    return total_loss / total

def test(model, loader, criterion):
    model.eval()  # set evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            targets = batch.y.view(-1).to(DEVICE)
            # logits, batch_attention = model(batch, return_attention=True)  # forward pass
            logits = model(batch)  # forward pass
            loss = criterion(logits, targets)
            total_loss += loss.item() * batch.num_graphs
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == targets).sum().item()
            total += batch.num_graphs
    return total_loss / total, correct / total

def main():
    print(f"\nTraining on: {dataset_path}")
    # trainingset = GraphDataset(dataset_path)
    trainingset = GraphDataset(dataset_path, split='Training')
    loader = DataLoader(trainingset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    model = MAGClassifier(trainingset.node_dim, trainingset.edge_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    start_time = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        loss = train(model, loader, optimizer, criterion, epoch)
        print(f"Epoch {epoch}: Loss {loss:.3f} Time {time.time() - start_time:.0f}s")
    print("Training complete.")

    loss, acc  = test(model, loader, criterion)
    print(f"Final Training Loss: {loss:.3f} Acc: {acc:.3f}")

    print(f"\nTesting on: {dataset_path}")
    # testset = GraphDataset(dataset_path)
    testset = GraphDataset(dataset_path, split='Test')
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE)
    test_loss, test_acc = test(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.3f} Test Acc: {test_acc:.3f}")
    print("Testing complete.")

if __name__ == "__main__":
    import time
    print()
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEVICE: {DEVICE}")
    BATCH_SIZE = 64  # reducing waste since drop_last=True
    LR = 1e-4
    NUM_EPOCHS = 20
    ## ESA
    # weight_decay = 1e-10 nel README, 1e-3 come default (AdamW)
    # HIDDEN_DIM = 256  # = MLP_hidden = graph_dim
    # BATCH_SIZE = 128
    # NUM_HEADS = 16
    # LAYER_TYPES = ['MSMSMP']
    # DROPOUT = 0
    # Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    dataset_path = 'DATASETS/MUTA_SARPY_4204.csv'
    main() 