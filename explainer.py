import torch
 
from explainer_att import AttentionExplainer
from explainer_ig import IGExplainer

from parameters import OPTIMAL_BATCH_SIZE

from logging_config import getLogger
logger = getLogger(__name__)


# Every registered explainer is run by Explainer.explanations()
EXPLAINERS = (AttentionExplainer, IGExplainer)


class Explainer:

    def __init__(self, calibration_data):
        self.calibration_data = calibration_data
        self.explainers = [cls(calibration_data) for cls in EXPLAINERS]

    def render(self, index, graph, weights, intensity=1.0):
        """Render every explainer for one molecule: name -> PNG bytes."""
        return {e.name: e.depict(graph, weights[e.name], index, intensity=intensity)
                for e in self.explainers}

    def explanations(self, model, dataset):
        """Yield (index, graph, weights) for one molecule at a time. No rendering, no input()."""
        device = next(model.parameters()).device
        model.eval()
        c = 0
        for batch in dataset.get_loader(OPTIMAL_BATCH_SIZE[device.type], is_train=False):
            batch = batch.to(device)
            directional_weights = {e.name: e.attributions(model, batch.clone()) for e in self.explainers}  # why clone()? for random seed consistency?
            for i, graph in enumerate(batch.to_data_list()):
                weights = {e.name: _to_undirected(graph, directional_weights[e.name][i]) for e in self.explainers}
                for e in self.explainers:
                    e.annotate(graph, c)  # DEBUG
                yield c, graph, weights
                c += 1

# The two directions of each bond are summed into a single per-bond weight
def _to_undirected(graph, edge_weights, verbose=True):
    """4th aggregation step: sum the two directional weights of each bond -> [num_bonds]."""
    undirected_weights = torch.zeros(graph.mol.GetNumBonds(),
                            dtype=edge_weights.dtype,
                            device=edge_weights.device)
    seen = set()  # DEBUG
    for weight, (src, dst) in zip(edge_weights, graph.edge_index.T):
        bond_idx = graph.mol.GetBondBetweenAtoms(int(src), int(dst)).GetIdx()
        if verbose and bond_idx in seen:  # DEBUG: 2nd directional pass, for logging
            logger.debug(f"Bond {bond_idx}: {undirected_weights[bond_idx] + weight:.2f} = "  # DEBUG
                        f"({undirected_weights[bond_idx]:.4f} + {weight:.4f})")  # DEBUG
        seen.add(bond_idx)  # DEBUG
        undirected_weights[bond_idx] += weight
    return undirected_weights