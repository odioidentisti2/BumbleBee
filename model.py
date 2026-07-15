import torch
from torch_geometric.utils import to_dense_batch
from architectures import ESA, mlp
from adj_mask_utils import edge_adjacency, edge_mask

from molecular_data import ATOM_DIM, BOND_DIM
from parameters import model_params as PARAMS

# # to be detached
# enc_repr = []
# dec_repr = []   
# att_list = []


class MAG(torch.nn.Module):
    
    def __init__(self, device, node_dim=ATOM_DIM, edge_dim=BOND_DIM):
        super(MAG, self).__init__()
        self.layer_types = PARAMS['layer_types']
        self.hidden_dim = PARAMS['hidden_dim']
        # Edge feature encoder (node-edge MLP)
        self.input_mlp = mlp(2 * node_dim + edge_dim, PARAMS['in_out_mlp'], self.hidden_dim)
        # ESA block
        self.esa = ESA(self.hidden_dim, self.layer_types, PARAMS['mlp_expansion'])
        # Predictor
        self.output_mlp = mlp(self.hidden_dim, PARAMS['in_out_mlp'], 1)
        # Track attention weights for analysis
        self._tracking_attention = False
        self._attention_store = []
        
        self.to(device)
        
    def track_attention(self, enable=True):
        """Enable or disable attention tracking."""
        self._tracking_attention = enable
        self._attention_store = []

    def batch_forward(self, edge_features, edge_index, node_batch):
        batched_h = self.input_mlp(edge_features)  # [batch_edges, hidden_dim]
        edge_batch = self._edge_batch(edge_index, node_batch)  # [batch_edges]
        max_edges = torch.bincount(edge_batch).max().item()
        dense_batch_h, pad_mask = to_dense_batch(batched_h, edge_batch, fill_value=0, max_num_nodes=max_edges)
        batch_size = node_batch.max().item() + 1
        adj_mask = edge_mask(edge_index, node_batch, batch_size, max_edges)  # [batch_size, max_edges, max_edges]
        # ESA forward
        out = self.esa(dense_batch_h, adj_mask, pad_mask=pad_mask, \
                       track_attention=self._tracking_attention)  # [batch_size, hidden_dim]
        if self._tracking_attention:
            attention = self.esa.get_last_attention()  # [batch_size, seq_len]
            for i, graph_attention in enumerate(attention.unbind(dim=0)):
                graph_attention = graph_attention[pad_mask[i]] # Crop padded positions
                self._attention_store.append(graph_attention.detach().cpu())
            # att_list.extend(batch_attention)
            # enc_repr.extend(self.esa.enc_out.unbind(dim=0))
            # dec_repr.extend(self.esa.dec_out.unbind(dim=0))
        # out = torch.where(pad_mask.unsqueeze(-1), out, torch.zeros_like(out))
        # <- DROPOUT here if needed
        # MLP
        logits = torch.flatten(self.output_mlp(out))    # [batch_size]
        return  logits

    def graph_forward(self, edge_features, edge_index, node_batch):
        batched_h = self.input_mlp(edge_features)  # [batch_edges, hidden_dim]
        edge_batch = self._edge_batch(edge_index, node_batch)  # [batch_edges]
        batch_size = node_batch.max().item() + 1
        src, dst = edge_index
        out = torch.zeros((batch_size, self.hidden_dim), device=edge_features.device)
        # Per-graph
        for i in range(batch_size):
            graph_mask = (edge_batch == i)  # mask for graph i
            h = batched_h[graph_mask]  # [graph_edges, hidden_dim]
            adj_mask = edge_adjacency(src[graph_mask], dst[graph_mask])  # [graph_edges, graph_edges]
            # Add batch dimension
            adj_mask = adj_mask.unsqueeze(0)
            h = h.unsqueeze(0)
            # ESA forward
            out[i] = self.esa(h, adj_mask, track_attention=self._tracking_attention)  # [hidden_dim]  (pythorch auto-removes batch dimension)
            if self._tracking_attention:
                graph_attention = self.esa.get_last_attention().squeeze(0)  # Remove batch dimension
                self._attention_store.append(graph_attention.detach().cpu())
                # att_list.extend(graph_attention.unbind(dim=0))
                # enc_repr.extend(self.esa.enc_out.unbind(dim=0))
                # dec_repr.extend(self.esa.dec_out.unbind(dim=0))
        # MLP
        logits = torch.flatten(self.output_mlp(out))    # [batch_size]
        return logits
    
    def forward(self, batch):
        """
        Args:
            batch: batch from DataLoader (torch_geometric.data.Batch)
                batch.x              # [batch_nodes, node_dim]
                batch.edge_index     # [2, batch_edges]
                batch.edge_attr      # [batch_edges, edge_dim]
                batch.batch          # [batch_nodes]
        """
        edge_feat = MAG.get_features(batch)

        if (not PARAMS['BATCH_DEBUG'] and
            edge_feat.device.type == 'cpu' and
            batch.num_graphs > 16):  # CPU + big batch: graph attention (faster, less peak memory)
            ##  WARNING: checking num_graphs implies that the last batch can follow a different path!!!!!!!!!!
            return self.graph_forward(edge_feat, batch.edge_index, batch.batch)
        else:  # Batch attention
            return self.batch_forward(edge_feat, batch.edge_index, batch.batch)

        # DEBUG: Compare batch vs single graph Attention
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
    