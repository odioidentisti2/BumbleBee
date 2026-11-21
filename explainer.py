import torch

from parameters import OPTIMAL_BATCH_SIZE
from graphic import depict as draw_molecule

from abc import ABC, abstractmethod

from logging_config import getLogger
logger = getLogger(__name__)


class Explainer(ABC):
    """Base class for explainers."""

    name = "Base"
    color_scheme = 'red_or_green'

    def __init__(self, calibration_data):
        """Initialize the explainer with calibration data."""
        self.shift = 0
        self.factor = 1
        self.batch_logits = None  # DEBUG: checked in an assert in explanations()

    def attributions(self, model, batch):
        """Return a list of UNDIRECTED edge attribution tensors for the batch."""
        directional = self._attributions(model, batch)
        return [_to_undirected(graph, w) for graph, w in zip(batch.to_data_list(), directional)]

    @abstractmethod
    def _attributions(self, model, batch):
        """Return a list of DIRECTIONAL edge attribution tensors for the batch."""

    def extras(self):
        """Return a dict of extra info."""
        return {}
    
    def summary(self, graph, weights, prediction):
        """Log the explanation of one molecule."""
        logger.debug(f"{self.name.upper()} EXPLANATION", highlight=True)
        logger.debug(f"{graph.y.item():.2f} {graph.smiles}")
        logger.debug(f"PREDICTION: {prediction:.2f}")
        self._summary(weights)

    def _summary(self, weights, print_mean=False):
        """Log the method-specific part of summary()."""
        logger.debug(weight_summary(weights, print_mean=print_mean))
        logger.debug(scaled_summary(weights, self._transform(weights)))

    def _transform(self, weights):
        return (weights + self.shift) * self.factor
    
    def depict(self, graph, weights, intensity=1):
        return draw_molecule(graph, self._transform(weights) * intensity, self.color_scheme)


class ExplainerSuite:
    def __init__(self, explainers, calibration_data):
        self.explainers = [cls(calibration_data) for cls in explainers]

    def explanations(self, model, dataset):
        """Yield (graph, weights) for one molecule at a time."""
        device = next(model.parameters()).device
        model.eval()
        # c = 0
        loader = dataset.get_loader(OPTIMAL_BATCH_SIZE[device.type], is_train=False)
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            batch_weights = {e.name: e.attributions(model, batch.clone()) for e in self.explainers}
            for i, graph in enumerate(batch.to_data_list()):
                weights = {e.name: batch_weights[e.name][i] for e in self.explainers}
                # logits = dataset.logits[c]
                # for e in self.explainers:  # DEBUG
                #     assert abs(logits - e.batch_logits[i].item()) < 1e-6
                yield graph, weights
                # c += 1

    def render(self, graph, weights, intensity=1.0):
        return {e.name: e.depict(graph, weights[e.name], intensity=intensity) for e in self.explainers}

    def extras(self):
        return {e.name: e.extras() for e in self.explainers}
    

def weight_summary(weights, print_mean=False, title="WEIGHTS:"):
    lines = [
        f"{title}",
        str(weights),
        f"Weights range: {weights.min():.4f} - {weights.max():.4f}",
    ]
    if print_mean:
        lines.append(f"Weights mean: {weights.mean().item():.4f}")
    lines.extend([f"Weight sum: {weights.sum():.2f}"])
    return '\n' + '\n'.join(lines) + '\n'

def scaled_summary(weights, transformed, title="WEIGHTS (scaled):"):
    """Per-edge raw vs. scaled weights, plus the sum (pairs with weight_summary)."""
    raw, scaled = weights.tolist(), transformed.tolist()
    lines = [title]
    lines.extend(f"Edge {i}: {t:.2f} <- {w:.2f}" for i, (w, t) in enumerate(zip(raw, scaled)))
    lines.append(f"Sum: {sum(scaled):.2f}")
    return '\n' + '\n'.join(lines) + '\n'

# The two directions of each bond are summed into a single per-edge weight
def _to_undirected(graph, edge_weights, verbose=False):
    """Sum the two directional weights of each bond -> [num_bonds]."""
    undirected_weights = torch.zeros(graph.mol.GetNumBonds(),
                            dtype=edge_weights.dtype,
                            device=edge_weights.device)
    seen = set()  # DEBUG
    for weight, (src, dst) in zip(edge_weights, graph.edge_index.T):
        bond_idx = graph.mol.GetBondBetweenAtoms(int(src), int(dst)).GetIdx()
        if verbose and bond_idx in seen:  # DEBUG: 2nd directional pass, for logging
            logger.debug(f"Edge {bond_idx}: {undirected_weights[bond_idx] + weight:.2f} = "  # DEBUG
                        f"({undirected_weights[bond_idx]:.4f} + {weight:.4f})")  # DEBUG
        seen.add(bond_idx)  # DEBUG
        undirected_weights[bond_idx] += weight
    return undirected_weights