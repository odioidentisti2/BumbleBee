import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
import numpy as np
from architectures import ESA, mlp
from molecular_data import GraphDataset, ATOM_DIM, BOND_DIM
from adj_mask_utils import *
from model_explainer import *


class MAGClassifier(nn.Module):
    
    def __init__(self, node_dim, edge_dim, hidden_dim=128, mlp_hidden_dim=128, num_heads=8, output_dim=1):
        super(MAGClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        # Edge feature encoder (node-edge MLP)
        self.input_mlp = mlp(2 * node_dim + edge_dim, mlp_hidden_dim, hidden_dim)
        # ESA block
        self.esa = ESA(hidden_dim, num_heads, 'MMSP')
        # Classifier
        self.output_mlp = mlp(hidden_dim, mlp_hidden_dim, output_dim)

    def single_forward(self, edge_features, edge_index, batch, return_attention=False):
        # I shouldn't pass the batch here...
        batched_h = self.input_mlp(edge_features)  # [batch_edges, hidden_dim]
        edge_batch = self._edge_batch(edge_index, batch.batch)  # [batch_edges]
        out = torch.zeros((batch.num_graphs, self.hidden_dim), device=edge_features.device)
        src, dst = edge_index
        attn_weights = []
        for i in range(batch.num_graphs):
            graph_mask = (edge_batch == i)
            h = batched_h[graph_mask]  # [graph_edges, hidden_dim]
            adj_mask = edge_adjacency(src[graph_mask], dst[graph_mask])  # [graph_edges, graph_edges]
            h = h.unsqueeze(0)  # Add batch dimension
            adj_mask = adj_mask.unsqueeze(0)  # Add batch dimension
            out[i] = self.esa(h, adj_mask)  # [hidden_dim]
            # out[i] = out[i].squeeze(0)  # Remove batch dimension ???
            if return_attention:
                attn = self.esa.get_attn_weights().squeeze(0)  # Remove batch dimension
                attn_weights.append(attn.detach().cpu().numpy())
                # graph = batch.to_data_list()[i]
                # print("\nDEPICT ATTENTION")
                # depict(graph, attn, graph.edge_index)
        if return_attention:
            return attn_weights
        logits = self.output_mlp(out)    # [batch_size, output_dim]
        return torch.flatten(logits)     # [batch_size]
    
    def batch_forward(self, edge_features, edge_index, batch):
        batched_h = self.input_mlp(edge_features)  # [batch_edges, hidden_dim]
        edge_batch = self._edge_batch(edge_index, batch.batch)  # [batch_edges]
        max_edges = max([g.num_edges for g in batch.to_data_list()])
        dense_batch_h, _ = to_dense_batch(batched_h, edge_batch, fill_value=0, max_num_nodes=max_edges)
        adj_mask = edge_mask(edge_index, batch.batch, batch.num_graphs, max_edges)
        out = self.esa(dense_batch_h, adj_mask,)  # [batch_size, hidden_dim]
        logits = self.output_mlp(out)    # [batch_size, output_dim]
        return torch.flatten(logits)     # [batch_size]

    def forward(self, batch, return_attention=False):
        """
        Args:
            batch: batch from DataLoader (torch_geometric.data.Batch)
                batch.x              # [batch_nodes, node_dim]
                batch.edge_index     # [2, batch_edges]
                batch.edge_attr      # [batch_edges, edge_dim]
                batch.batch          # [batch_nodes]
        """
        edge_feat = MAGClassifier.get_features(batch)

        if edge_feat.device.type == 'cuda':  # GPU: batch Attention
            return self.batch_forward(edge_feat, batch.edge_index, batch)
        else:  # CPU: per-graph Attention
            return self.single_forward(edge_feat, batch.edge_index, batch, return_attention)
        
    @staticmethod
    def get_features(batch):
        # Concatenate node (src and dst) and edge features
        src, dst = batch.edge_index
        return torch.cat([batch.x[src], batch.x[dst], batch.edge_attr], dim=1)

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
    for batch in loader:
        batch = batch.to(DEVICE)
        targets = batch.y.view(-1).to(DEVICE)
        with torch.no_grad():
            logits = model(batch)  # forward pass
            loss = criterion(logits, targets)
            total_loss += loss.item() * batch.num_graphs
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == targets).sum().item()
            total += batch.num_graphs
    return total_loss / total, correct / total

def save(model):
    model_path = f"model_{time.strftime('%Y%m%d_%H%M')}_{glob['BATCH_SIZE']}_{glob['LR']}_{glob['NUM_EPOCHS']}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

def load(model_path):
    model = MAGClassifier(ATOM_DIM, BOND_DIM).to(DEVICE)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def main():
    criterion = nn.BCEWithLogitsLoss()

    # Train
    # print(f"\nTraining on: {DATASET_PATH}")
    # trainingset = GraphDataset(DATASET_PATH, split='Training')
    # loader = DataLoader(trainingset, batch_size=glob['BATCH_SIZE'], shuffle=True, drop_last=True)
    # model = MAGClassifier(ATOM_DIM, BOND_DIM).to(DEVICE)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=glob['LR'])
    # start_time = time.time()
    # for epoch in range(1, glob['NUM_EPOCHS'] + 1):
    #     loss = train(model, loader, optimizer, criterion, epoch)
    #     print(f"Epoch {epoch}: Loss {loss:.3f} Time {time.time() - start_time:.0f}s")
    # save(model)
    ## loader = DataLoader(trainingset, batch_size=glob['BATCH_SIZE'])
    ## loss, acc  = test(model, loader, criterion)
    ## print(f"Training Loss: {loss:.3f} Acc: {acc:.3f}")

    # Load saved model
    model_path = 'model_20250822_210138.pt'
    print(f"\nLoading model {model_path}")
    model = load(model_path)

    # Test
    print(f"\nTesting on: {DATASET_PATH}")
    testset = GraphDataset(DATASET_PATH, split='Test')
    test_loader = DataLoader(testset, batch_size=glob['BATCH_SIZE'])
    test_loss, test_acc = test(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.3f} Test Acc: {test_acc:.3f}")

    # Explain with gradients
    model.eval()
    for dummy_batch in DataLoader(testset, batch_size=1):
        explain_with_gradients(model, dummy_batch, steps=100)

if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    # GLOBALS
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATASET_PATH = 'DATASETS/MUTA_SARPY_4204.csv'
    glob = {
        "BATCH_SIZE": 64,  # I should try reducing waste since drop_last=True
        "LR": 1e-4,
        "NUM_EPOCHS": 20,
    }
    # Print time and model stamps
    print()
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"DEVICE: {DEVICE}")
    # import pprint
    # pprint.pprint(glob)
    main() 

    ## ESA repo
    # weight_decay = 1e-10 nel README, 1e-3 come default (AdamW)
    # HIDDEN_DIM = 256  # = MLP_hidden = graph_dim
    # BATCH_SIZE = 128
    # NUM_HEADS = 16
    # LAYER_TYPES = ['MSMSMP']
    # DROPOUT = 0




# def explain(model, train_loader, test_loader):
#     model.eval()

#     # Place this inside explain_with_attention
#     # In the depiction of attention, I highlight bonds with an intensity proportional to their attention weights.
#     # I'd like this intensity to be "absolute" (referred to the whole training set).
#     # But attention weights are very relative to each molecule (and proportionate to the number of bonds, after softmax).
#     # So I need to find a "max intensity" threshold from the training set, to normalize it,
#     # I can divide by the mean attention weight for each molecule.
#     # In conclusion: given the distribution of attention max/mean ratio in the training set, I can set a threshold
#     # (e.g. mean + std) to be the "max intensity" (1.0) in the depiction, so that only "outlier" weights
#     # are fully highlighted. In other words, let's say that the maximum max/mean ratio in any molecule
#     # in the training set was 30 (30 times more attention than the average), mean + std will be lower than that,
#     # let's say 7, so in the depiction an attention weight 7 times higher than the average
#     # will be highlighted with intensity 1.0.
#     train_attn_weights = []
#     for batch in train_loader:
#         batch = batch.to(DEVICE)
#         with torch.no_grad():
#             train_attn_weights.extend(model(batch, return_attention=True))
#     dist = np.array([aw.max() / aw.mean() for aw in train_attn_weights])
#     top = dist.mean() + dist.std()
#     print(f"Top attention weight: {top:.3f}")
#     for batch in test_loader:
#         batch = batch.to(DEVICE)
#         with torch.no_grad():
#             attn_weights = model(batch, return_attention=True)
#         normalized_attn = []
#         for aw in attn_weights:
#             bond_ratios = aw / aw.mean()  # Relative to this molecule
#             normalized = np.clip(bond_ratios / top, 0, 1)  # Scale by training threshold
#             normalized_attn.append(normalized)

#         # explain_with_attention(batch.to_data_list(), normalized_attn)  # Normalize        
#         for data, attention in zip(batch.to_data_list(), normalized_attn):
#             print("\nDEPICT ATTENTION")
#             depict(data, attention) 