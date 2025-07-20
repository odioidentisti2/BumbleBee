from esa import consecutive, first_unique_index

def edge_adjacency(batched_edge_index):
    N_e = batched_edge_index.size(1)
    source_nodes = batched_edge_index[0]
    target_nodes = batched_edge_index[1]

    # unsqueeze and expand
    exp_src = source_nodes.unsqueeze(1).expand(-1, N_e)
    exp_trg = target_nodes.unsqueeze(1).expand(-1, N_e)

    src_adj = exp_src == exp_src.T
    trg_adj = exp_trg == exp_trg.T
    cross = (exp_src == exp_trg.T) | (exp_trg == exp_src.T)

    return (src_adj | trg_adj | cross)


def edge_mask(b_ei, b_map, B, L):
    mask = torch.full((B, L, L), fill=False)
    edge_to_graph = b_map.index_select(0, b_ei[0, :])

    edge_adj = edge_adjacency(b_ei)
    ei_to_original = consecutive(
        first_unique_index(edge_to_graph), b_ei.size(1)
    )

    edges = edge_adj.nonzero()
    graph_index = edge_to_graph.index_select(0, edges[:, 0])
    coord_1 = ei_to_original.index_select(0, edges[:, 0])
    coord_2 = ei_to_original.index_select(0, edges[:, 1])

    mask[graph_index, coord_1, coord_2] = True

    return ~mask

