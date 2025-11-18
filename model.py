import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from architectures import ESA, mlp
from adj_mask_utils import edge_adjacency, edge_mask


HIDDEN_DIM = 128  # ESA hidden dimension
HEADS = 8
print("HIDDEN_DIM:", HIDDEN_DIM, "\nHEADS:", HEADS)


class MAGClassifier(nn.Module):

    IN_OUT_MLP_HIDDEN_DIM = 128  # MLP hidden dimension for input/output layers
    
    def __init__(self, node_dim, edge_dim, layer_types, hidden_dim=HIDDEN_DIM, num_heads=HEADS, output_dim=1):
        super(MAGClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        # Edge feature encoder (node-edge MLP)
        self.input_mlp = mlp(2 * node_dim + edge_dim, MAGClassifier.IN_OUT_MLP_HIDDEN_DIM, hidden_dim)
        # ESA block
        self.esa = ESA(hidden_dim, num_heads, layer_types)
        # Classifier
        self.output_mlp = mlp(hidden_dim, MAGClassifier.IN_OUT_MLP_HIDDEN_DIM, output_dim)

    def batch_forward(self, edge_features, edge_index, node_batch):
        batched_h = self.input_mlp(edge_features)  # [batch_edges, hidden_dim]
        edge_batch = self._edge_batch(edge_index, node_batch)  # [batch_edges]
        max_edges = torch.bincount(edge_batch).max().item()
        dense_batch_h, pad_mask = to_dense_batch(batched_h, edge_batch, fill_value=0, max_num_nodes=max_edges)
        batch_size = node_batch.max().item() + 1
        adj_mask = edge_mask(edge_index, node_batch, batch_size, max_edges)
        out = self.esa(dense_batch_h, adj_mask, pad_mask)  # [batch_size, hidden_dim]
        # out = torch.where(pad_mask.unsqueeze(-1), out, torch.zeros_like(out))
        logits = self.output_mlp(out)    # [batch_size, output_dim]
        return torch.flatten(logits)     # [batch_size] 

    def single_forward(self, edge_features, edge_index, node_batch, return_attention=False):
        self.esa.expose_attention(return_attention)
        batched_h = self.input_mlp(edge_features)  # [batch_edges, hidden_dim]
        edge_batch = self._edge_batch(edge_index, node_batch)  # [batch_edges]
        batch_size = node_batch.max().item() + 1
        src, dst = edge_index
        attn_weights = []
        out = torch.zeros((batch_size, self.hidden_dim), device=edge_features.device)
        for i in range(batch_size):
            graph_mask = (edge_batch == i)
            h = batched_h[graph_mask]  # [graph_edges, hidden_dim]
            adj_mask = edge_adjacency(src[graph_mask], dst[graph_mask])  # [graph_edges, graph_edges]
            h = h.unsqueeze(0)  # Add batch dimension
            adj_mask = adj_mask.unsqueeze(0)  # Add batch dimension
            out[i] = self.esa(h, adj_mask)  # [hidden_dim]
            # out[i] = out[i].squeeze(0)  # Remove batch dimension ???
            if return_attention:
                attn = self.esa.get_attention().squeeze(0)  # Remove batch dimension
                attn_weights.append(attn.detach().cpu())
                # if i == batch_size - 1:
                #     return attn_weights
        if return_attention:
            return attn_weights
        # DROPOUT?
        logits = self.output_mlp(out)    # [batch_size, output_dim]
        return torch.flatten(logits)     # [batch_size]
    
    def forward(self, batch, BATCH_DEBUG=False):
        """
        Args:
            batch: batch from DataLoader (torch_geometric.data.Batch)
                batch.x              # [batch_nodes, node_dim]
                batch.edge_index     # [2, batch_edges]
                batch.edge_attr      # [batch_edges, edge_dim]
                batch.batch          # [batch_nodes]
        """
        edge_feat = MAGClassifier.get_features(batch)

        if BATCH_DEBUG or edge_feat.device.type == 'cuda':  # GPU: batch Attention
            return self.batch_forward(edge_feat, batch.edge_index, batch.batch)
        else:  # per-graph Attention (faster on CPU)
            return self.single_forward(edge_feat, batch.edge_index, batch.batch)
        # if not torch.allclose(batch_logits, single_logits, rtol=1e-4, atol=1e-7):
        #     print("WARNING: Batch and Single logits differ!")
        # return batch_logits
        
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