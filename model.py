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

        # Bond feature projection for attention bias
        self.bond_projection = nn.Linear(edge_dim, GLOB['heads'])  # edge_attr: [bond_type_onehot(5), is_conjugated(1), is_in_ring(1)]
    
        # ESA block
        self.esa = ESA(self.hidden_dim, GLOB['heads'], layer_types)
        # Classifier
        self.output_mlp = mlp(self.hidden_dim, GLOB['in_out_mlp'], 1)

    def batch_forward(self, node_features, edge_index, edge_attr, node_batch):
        batched_h = self.input_mlp(node_features)  # [num_nodes, hidden_dim] - ATOMS
        max_nodes = torch.bincount(node_batch).max().item()
        dense_batch_h, pad_mask = to_dense_batch(batched_h, node_batch, fill_value=0, max_num_nodes=max_nodes)
        batch_size = node_batch.max().item() + 1
        
        # Create adjacency mask
        adj_mask = atom_mask(edge_index, node_batch, batch_size, max_nodes)  # [batch_size, max_nodes, max_nodes]
        
        # Create bond bias
        bond_bias = self._create_bond_bias_batch(edge_attr, edge_index, node_batch, batch_size, max_nodes)  # [batch_size, num_heads, max_nodes, max_nodes]
        
        # Combine adjacency mask with bond bias
        # Where adj_mask is True (connected), use bond_bias; otherwise -inf (masked out)
        adj_mask_expanded = adj_mask.unsqueeze(1)  # [batch_size, 1, max_nodes, max_nodes]
        combined_mask = torch.where(adj_mask_expanded, bond_bias, torch.tensor(float('-inf'), device=bond_bias.device))
        
        out = self.esa(dense_batch_h, combined_mask, pad_mask)  # [batch_size, hidden_dim]
        logits = self.output_mlp(out)    # [batch_size, output_dim]
        return torch.flatten(logits)     # [batch_size] 

    def single_forward(self, node_features, edge_index, edge_attr, node_batch, return_attention=False):
        self.esa.expose_attention(return_attention)
        batched_h = self.input_mlp(node_features)  # [num_nodes, hidden_dim] - ATOMS not edges
        batch_size = node_batch.max().item() + 1
        src, dst = edge_index
        attn_weights = []
        out = torch.zeros((batch_size, self.hidden_dim), device=node_features.device)
        for i in range(batch_size):
            graph_mask = (node_batch == i)  # mask for nodes in graph i
            h = batched_h[graph_mask]  # [graph_nodes, hidden_dim] - atoms in this graph
            
            # Filter edges to only those within this graph
            edge_mask = graph_mask[src] & graph_mask[dst]
            graph_edge_index = edge_index[:, edge_mask]
            graph_edge_attr = edge_attr[edge_mask]  # Filter bond features for this graph
            
            # Remap global node indices to local indices (0, 1, 2, ..., num_nodes-1)
            node_mapping = torch.full((node_batch.size(0),), -1, dtype=torch.long, device=node_features.device)
            node_mapping[graph_mask] = torch.arange(h.size(0), device=node_features.device)
            local_edge_index = torch.stack([
                node_mapping[graph_edge_index[0]],
                node_mapping[graph_edge_index[1]]
            ], dim=0)
            
            # Create adjacency mask for atoms
            adj_mask = atom_adjacency(local_edge_index, h.size(0))
            
            # Create bond bias for this graph
            bond_bias = self._create_bond_bias_single(graph_edge_attr, local_edge_index, h.size(0))
            
            # Combine adjacency mask with bond bias
            combined_mask = torch.where(adj_mask, bond_bias, torch.tensor(float('-inf'), device=bond_bias.device))
            
            # Add batch dimension
            combined_mask = combined_mask.unsqueeze(0)  # [1, num_heads, num_nodes, num_nodes]
            h = h.unsqueeze(0)  # Add batch dimension
            out[i] = self.esa(h, combined_mask)  # [hidden_dim]            
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
        node_feat = batch.x

        if BATCH_DEBUG or node_feat.device.type == 'cuda' and not return_attention:  # GPU: batch Attention
            return self.batch_forward(node_feat, batch.edge_index, batch.edge_attr, batch.batch)
        else:  # per-graph Attention (faster on CPU)
            return self.single_forward(node_feat, batch.edge_index, batch.edge_attr, batch.batch, return_attention)
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
    
    def _create_bond_bias_batch(self, edge_attr, edge_index, node_batch, batch_size, max_nodes):
        """
        Creates bond bias for batched attention.
        
        Args:
            edge_attr: [num_edges, edge_dim] - [bond_type_onehot(5), is_conjugated(1), is_in_ring(1)]
            edge_index: [2, num_edges] - connectivity
            node_batch: [num_nodes] - batch assignment
            batch_size: int
            max_nodes: int - max atoms per graph in batch
        
        Returns:
            bond_bias: [batch_size, num_heads, max_nodes, max_nodes]
        """
        # Project all bond features directly to num_heads
        bond_bias_flat = self.bond_projection(edge_attr)  # [num_edges, num_heads]
    
        # Get edge batch assignment
        src, dst = edge_index
        edge_batch = node_batch[src]  # [num_edges] - which batch each edge belongs to
        
        # Map global node indices to local indices within each graph
        node_counts = torch.bincount(node_batch, minlength=batch_size)
        cumsum = torch.cat([torch.tensor([0], device=node_batch.device), node_counts.cumsum(0)[:-1]])
        local_src = src - cumsum[node_batch[src]]
        local_dst = dst - cumsum[node_batch[dst]]
        
        # Create bond bias tensor
        bond_bias = torch.zeros(batch_size, max_nodes, max_nodes, self.bond_projection.out_features,
                               device=edge_attr.device)
        
        # Fill in bond biases at edge positions
        bond_bias[edge_batch, local_src, local_dst] = bond_bias_flat
        
        # Permute to [batch, num_heads, max_nodes, max_nodes]
        return bond_bias.permute(0, 3, 1, 2).contiguous()

    def _create_bond_bias_single(self, edge_attr, edge_index, num_nodes):
        """
        Creates bond bias for single graph attention.
        
        Args:
            edge_attr: [num_edges, edge_dim] - [bond_type_onehot(5), is_conjugated(1), is_in_ring(1)]
            edge_index: [2, num_edges] - local edge indices
            num_nodes: int - number of atoms in this graph
        
        Returns:
            bond_bias: [num_heads, num_nodes, num_nodes]
        """
        # Project all bond features directly to num_heads
        bond_bias_flat = self.bond_projection(edge_attr)  # [num_edges, num_heads]
        
        # Create bond bias tensor
        bond_bias = torch.zeros(num_nodes, num_nodes, self.bond_projection.out_features,
                               device=edge_attr.device)
        
        src, dst = edge_index
        bond_bias[src, dst] = bond_bias_flat
        
        # Permute to [num_heads, num_nodes, num_nodes]
        return bond_bias.permute(2, 0, 1).contiguous()

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