import torch


def get_first_unique_index(t):
    # This is taken from Stack Overflow :)
    unique, idx, counts = torch.unique(t, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    zero = torch.tensor([0], device=t.device)
    cum_sum = torch.cat((zero, cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]
    return first_indicies

def generate_consecutive_tensor(input_tensor, final):
    # Calculate the length of each segment
    lengths = input_tensor[1:] - input_tensor[:-1]
    # Append the final length
    lengths = torch.cat((lengths, torch.tensor([final - input_tensor[-1]], device=input_tensor.device)))
    # Create ranges for each segment
    ranges = [torch.arange(0, length, device=input_tensor.device) for length in lengths]
    # Concatenate all ranges into a single tensor
    result = torch.cat(ranges)
    return result


# Return a boolean edge adjacency mask
def edge_adjacency(source, target):
    """
    Returns a boolean adjacency mask indicating which edges are adjacent (i.e., share a node).

    Note:
        This implementation uses broadcasting and may be inefficient for large graphs.
        Consider profiling or optimizing this function for large inputs.

    Args:
        source (torch.Tensor): Source node indices of edges.
        target (torch.Tensor): Target node indices of edges.

    Returns:
        torch.Tensor: Boolean adjacency mask of shape [num_edges, num_edges].
    """
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
    """
    Generates a boolean adjacency mask for edges in a batched graph.

    Args:
        b_edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
        b_map (torch.Tensor): Mapping from node indices to batch indices.
        batch_size (int): Number of batches in the graph.
        num_edges (int): Number of edges in the graph.

    Returns:
        torch.Tensor: Boolean adjacency mask of shape [batch_size, num_edges, num_edges].
    """
    adj_mask = torch.full(
        size=(batch_size, num_edges, num_edges),
        fill_value=False,
        device=b_edge_index.device,
        dtype=torch.bool,
        requires_grad=False,
    )
    edge_to_graph = b_map[b_edge_index[0]]
    edge_adj_matrix = edge_adjacency(b_edge_index[0], b_edge_index[1])
    ei_to_original_index = generate_consecutive_tensor(
        get_first_unique_index(edge_to_graph), edge_to_graph.shape[0]
    )
    eam_nonzero = edge_adj_matrix.nonzero()
    adj_mask[
        edge_to_graph[eam_nonzero[:, 0]],
        ei_to_original_index[eam_nonzero[:, 0]],
        ei_to_original_index[eam_nonzero[:, 1]],
    ] = True
    adj_mask = ~adj_mask  # WHY INVERTING THE MASK?
    return adj_mask
