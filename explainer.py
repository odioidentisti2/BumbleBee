import torch
import base64

from explainer_att import AttentionExplainer
from explainer_ig import IGExplainer

from parameters import OPTIMAL_BATCH_SIZE

from logging_config import getLogger
logger = getLogger(__name__)


# Every registered explainer is run by explain()
EXPLAINERS = (AttentionExplainer, IGExplainer)


class Explainer:

    def __init__(self, calibration_data):
        self.calibration_data = calibration_data
        self.explainers = [cls(calibration_data) for cls in EXPLAINERS]

    def explain(self, model, dataset):
        device = next(model.parameters()).device
        model.eval()
        all_weights = {e.name: [] for e in self.explainers}
        c = b = 0
        for batch in dataset.get_loader(OPTIMAL_BATCH_SIZE[device.type], is_train=False):
            batch = batch.to(device)
            directional_weights = {e.name: e.attributions(model, batch.clone()) for e in self.explainers}  # why clone()? for random seed consistency?
            for i, graph in enumerate(batch.to_data_list()):
                weights = {e.name: _to_undirected(graph, directional_weights[e.name][i]) for e in self.explainers}
                for e in self.explainers:
                    e.annotate(graph, c)  # DEBUG
                repeat = True
                while repeat:
                    for e in self.explainers:
                        e.depict(graph, weights[e.name], c)
                    user_input = ''
                    # user_input = input("Press Enter to continue, '-' to halve intensity, '+' to double intensity: ")
                    plus_count = user_input.count('+')
                    minus_count = user_input.count('-')
                    if plus_count + minus_count > 0:
                        for e in self.explainers:
                            e.depicter.intensity *= (2 ** plus_count) / (2 ** minus_count)
                    else:
                        repeat = False  # Move to next molecule
                for e in self.explainers:
                    all_weights[e.name].append(weights[e.name])
                c += 1
                if b >= 1:
                    break  # DEBUG
            if b >= 1:
                break  # DEBUG
            b += 1

        return all_weights

    def explain_first(self, model, dataset):
        device = next(model.parameters()).device
        model.eval()
        batch = next(iter(dataset.get_loader(OPTIMAL_BATCH_SIZE[device.type], is_train=False))).to(device)
        graph = batch.to_data_list()[0]

        raw = {e.name: e.attributions(model, batch.clone()) for e in self.explainers}
        for e in self.explainers:
            e.annotate(graph, 0)

        explanations = {}
        for e in self.explainers:
            edge_weights = _to_undirected(graph, raw[e.name][0])
            image = e.depict(graph, edge_weights, 0)
            explanations[e.name] = {"label": e.name,
                                    "image": _data_uri(image),
                                    "tokens": _tokens(edge_weights)}
        return explanations


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


def _data_uri(png):
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _tokens(weights):
    return [{"name": f"token{bond_idx}", "weight": float(weight)}
            for bond_idx, weight in enumerate(weights.tolist())]
