import torch

# Return a boolean edge adjacency mask
def edge_adjacency(source, target):
    # stack and slice
    src_dst = torch.stack([source, target], dim=1) # [num_edges, 2]
    src_nodes = src_dst[:, 0:1]  # [num_edges, 1]
    dst_nodes = src_dst[:, 1:2]  # [num_edges, 1]
    # Create adjacency mask: edges are adjacent if they share a node
    src_adj = (src_nodes == src_nodes.T) | (src_nodes == dst_nodes.T)
    dst_adj = (dst_nodes == src_nodes.T) | (dst_nodes == dst_nodes.T)
    adj_mask = src_adj | dst_adj  # [num_edges, num_edges]
    return adj_mask.fill_diagonal_(0)  # # Mask out self-adjacency

def edge_mask(b_edge_index, b_map, batch_size, num_edges):
    adj_mask = torch.full(
        size=(batch_size, num_edges, num_edges),
        fill_value=False,
        device=b_edge_index.device,
        dtype=torch.bool,
        requires_grad=False,
    )
    edge_batch_mapping = b_map.index_select(0, edge_index[0, :])
    edge_batch_mapping = b_map[b_edge_index[0]]