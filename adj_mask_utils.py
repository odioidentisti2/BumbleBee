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
    return adj_mask.fill_diagonal_(False)  # # Mask out self-adjacency

def edge_mask(b_edge_index, b_map, batch_size, num_edges):
    """
    Generates a boolean adjacency mask for edges across a batch of graph instances.

    Args:
        b_edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
        b_map (torch.Tensor): Mapping from node indices to batch indices.
        batch_size (int): Number of graphs in the batch.
        num_edges (int): Maximum edge count across graphs in the batch (i.e., the padded edge dimension).

    Returns:
        torch.Tensor: Boolean adjacency mask of shape [batch_size, num_edges, num_edges],
                      where adj_mask[b, i, j] is True if edge i and edge j are adjacent within graph b.
    """

    edge_to_graph = b_map[b_edge_index[0]]
    ei_to_original_index = generate_consecutive_tensor(
        get_first_unique_index(edge_to_graph), edge_to_graph.shape[0]
    )
    edge_adj_matrix = edge_adjacency(b_edge_index[0], b_edge_index[1])

    eam_nonzero = edge_adj_matrix.nonzero()
    adj_mask = torch.full(
        size=(batch_size, num_edges, num_edges),
        fill_value=False,
        device=b_edge_index.device,
        dtype=torch.bool,
        requires_grad=False,
    )
    adj_mask[
        edge_to_graph[eam_nonzero[:, 0]],
        ei_to_original_index[eam_nonzero[:, 0]],
        ei_to_original_index[eam_nonzero[:, 1]],
    ] = True

    return adj_mask

# def proximity_masks(source, target, hops):
#     """    
#     Args:
#         edge_adj_mask: [num_edges, num_edges] boolean or float tensor
#         max_hops: number of hops to compute
    
#     Returns:
#         list of [num_edges, num_edges] boolean masks
#     """

#     adj_mask = edge_adjacency(source, target)
#     masks = [adj_mask]
#     A = adj_mask.float()  # Convert to float for matmul
#     cumulative = A.clone()
#     current = A.clone()
    
#     for _ in range(hops):        
#         current = torch.matmul(current, A)
#         # # Cumulative
#         # cumulative = cumulative + current
#         # mask = (cumulative > 0).fill_diagonal_(0).bool()  # Binarize, remove diagonal, convert to bool
        
#         # Exclusive
#         cumulative = cumulative.bool()
#         current = current.bool()
#         current = current & ~cumulative  # Get only new connections at this hop
#         cumulative = cumulative | current
#         mask = current.fill_diagonal_(0)
#         current = current.float()
#         cumulative = cumulative.float()

#         masks.append(mask)
#         # CHECK IF IDENTITY (EXCEPT DIAGONAL) THEN STOP AND FILL THE REST WITH LAST MASK
    
#     # # Verify cumulative property
#     # for i in range(len(masks) - 1):
#     #     assert (masks[i] <= masks[i+1]).all(), f"Mask {i+1} should include mask {i}"
#     #     assert not torch.equal(masks[i], masks[i+1]) or torch.equal(masks[i+1], torch.ones_like(masks[i+1]).fill_diagonal_(False))
#     # # Verify no self-loops
#     # for i, mask in enumerate(masks):
#     #     assert not mask.diag().any(), f"Mask {i} has self-loops!"

#     return masks

# def edge_mask(b_edge_index, b_map, batch_size, num_edges, hops):
#     masks = []
#     edge_to_graph = b_map[b_edge_index[0]]
#     ei_to_original_index = generate_consecutive_tensor(
#         get_first_unique_index(edge_to_graph), edge_to_graph.shape[0]
#     )
#     # edge_adj_matrix = edge_adjacency(b_edge_index[0], b_edge_index[1])
#     prox_masks = proximity_masks(b_edge_index[0], b_edge_index[1], hops=hops)

#     for hop in range(hops + 1):
#         edge_adj_matrix = prox_masks[hop]
#         eam_nonzero = edge_adj_matrix.nonzero()
#         adj_mask = torch.full(
#             size=(batch_size, num_edges, num_edges),
#             fill_value=False,
#             device=b_edge_index.device,
#             dtype=torch.bool,
#             requires_grad=False,
#         )
#         adj_mask[
#             edge_to_graph[eam_nonzero[:, 0]],
#             ei_to_original_index[eam_nonzero[:, 0]],
#             ei_to_original_index[eam_nonzero[:, 1]],
#         ] = True
#         masks.append(adj_mask)

#     return masks