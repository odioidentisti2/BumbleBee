import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Batch
from architectures import ESA, mlp
from adj_mask_utils import edge_adjacency, edge_mask,  atom_adjacency, atom_mask

from parameters import GLOB


class MAG(nn.Module):
    
    def __init__(self, node_dim, edge_dim, layer_types):
        super(MAG, self).__init__()
        self.layer_types = layer_types
        self.hidden_dim = GLOB['hidden_dim']
        # Edge feature encoder (node-edge MLP)
        # self.input_mlp = mlp(2 * node_dim + edge_dim, GLOB['in_out_mlp'], self.hidden_dim)
        self.input_mlp = mlp(node_dim, GLOB['in_out_mlp'], self.hidden_dim)
        # ESA block
        self.esa = ESA(self.hidden_dim, GLOB['heads'], layer_types)
        # Classifier
        self.output_mlp = mlp(self.hidden_dim, GLOB['in_out_mlp'], 1)

    def batch_forward(self, edge_features, edge_index, node_batch):
        batched_h = self.input_mlp(edge_features)  # [batch_edges, hidden_dim]
        edge_batch = self._edge_batch(edge_index, node_batch)  # [batch_edges]
        max_edges = torch.bincount(edge_batch).max().item()
        dense_batch_h, pad_mask = to_dense_batch(batched_h, edge_batch, fill_value=0, max_num_nodes=max_edges)
        batch_size = node_batch.max().item() + 1
        adj_mask = edge_mask(edge_index, node_batch, batch_size, max_edges)  # [batch_size, max_edges, max_edges]
        out = self.esa(dense_batch_h, adj_mask, pad_mask)  # [batch_size, hidden_dim]
        # out = torch.where(pad_mask.unsqueeze(-1), out, torch.zeros_like(out))
        logits = self.output_mlp(out)    # [batch_size, output_dim]
        return torch.flatten(logits)     # [batch_size] 

    def single_forward(self, edge_features, edge_index, node_batch, return_attention=False):
        self.esa.expose_attention(return_attention)
        batched_h = self.input_mlp(edge_features)  # [num_nodes, hidden_dim] - ATOMS not edges
        batch_size = node_batch.max().item() + 1
        src, dst = edge_index
        attn_weights = []
        out = torch.zeros((batch_size, self.hidden_dim), device=edge_features.device)
        for i in range(batch_size):
            graph_mask = (node_batch == i)  # mask for nodes in graph i
            h = batched_h[graph_mask]  # [graph_nodes, hidden_dim] - atoms in this graph
            
            # Filter edges to only those within this graph
            edge_mask = graph_mask[src] & graph_mask[dst]
            graph_edge_index = edge_index[:, edge_mask]
            
            # Remap global node indices to local indices (0, 1, 2, ..., num_nodes-1)
            node_mapping = torch.full((node_batch.size(0),), -1, dtype=torch.long, device=edge_features.device)
            node_mapping[graph_mask] = torch.arange(h.size(0), device=edge_features.device)
            local_edge_index = torch.stack([
                node_mapping[graph_edge_index[0]],
                node_mapping[graph_edge_index[1]]
            ], dim=0)
            
            # Create adjacency mask for atoms
            adj_mask = atom_adjacency(local_edge_index, h.size(0))
            
            # Add batch dimension
            adj_mask = adj_mask.unsqueeze(0)  # Add batch dimension
            h = h.unsqueeze(0)  # Add batch dimension
            out[i] = self.esa(h, adj_mask)  # [hidden_dim]            
            # out[i] = out[i].squeeze(0)  # Remove batch dimension ???
            if return_attention:
                attn = self.esa.get_attention().squeeze(0)  # Remove batch dimension
                attn_weights.append(attn)
        # DROPOUT?
        out = self.output_mlp(out)    # [batch_size, output_dim]
        logits = torch.flatten(out)    # [batch_size]
        if return_attention:
            return logits, attn_weights
        return logits 
    
    def forward(self, batch, return_attention=False, BATCH_DEBUG=False):
        """
        Args:
            batch: batch from DataLoader (torch_geometric.data.Batch)
                batch.x              # [batch_nodes, node_dim]
                batch.edge_index     # [2, batch_edges]
                batch.edge_attr      # [batch_edges, edge_dim]
                batch.batch          # [batch_nodes]
        """
        edge_feat = MAG.get_features(batch)
        node_feat = batch.x
        feat = node_feat

        if BATCH_DEBUG or edge_feat.device.type == 'cuda' and not return_attention:  # GPU: batch Attention
            return self.batch_forward(feat, batch.edge_index, batch.batch)
        else:  # per-graph Attention (faster on CPU)
            return self.single_forward(feat, batch.edge_index, batch.batch, return_attention)
        # if not torch.allclose(batch_logits, single_logits, rtol=1e-4, atol=1e-7):
        #     print("WARNING: Batch and Single logits differ!")
        # return batch_logits

    @staticmethod
    def _edge_batch(edge_index, node_batch):
        # Given edge_index [2, num_edges] and node_batch [num_nodes], 
        # return edge_batch [num_edges] where each edge takes the batch idx of its src node
        # (assumes all edges within a graph)
        return node_batch[edge_index[0]]
    
    @staticmethod
    def get_features(batch_or_graph):
        if not isinstance(batch_or_graph, Batch):  # single graph
            batch = Batch.from_data_list([batch_or_graph])
        else:
            batch = batch_or_graph
        # Concatenate node (src and dst) and edge features
        src, dst = batch.edge_index
        return torch.cat([batch.x[src], batch.x[dst], batch.edge_attr], dim=1)
        
    # @staticmethod
    # def get_features(batch):
    #     # Concatenate node (src and dst) and edge features
    #     src, dst = batch.edge_index
    #     return torch.cat([batch.x[src], batch.x[dst], batch.edge_attr], dim=1)
    
    


    # def get_encoder_output(self, batch, BATCH_DEBUG=False):
    #     """Returns encoder set features [batch, seq_len, hidden_dim] before pooling."""
    #     edge_feat = MAGClassifier.get_features(batch)
    #     batched_h = self.input_mlp(edge_feat)
    #     edge_batch = self._edge_batch(batch.edge_index, batch.batch)
    #     max_edges = torch.bincount(edge_batch).max().item()
    #     dense_batch_h, pad_mask = to_dense_batch(
    #         batched_h, edge_batch, fill_value=0, max_num_nodes=max_edges)
    #     batch_size = batch.batch.max().item() + 1
    #     adj_mask = edge_mask(batch.edge_index, batch.batch, batch_size, max_edges)
        
    #     # Run encoder only
    #     enc = dense_batch_h
    #     for layer in self.esa.encoder:
    #         enc = layer(enc, adj_mask=adj_mask, pad_mask=pad_mask)
        
    #     return enc  # [batch, seq_len, hidden_dim]