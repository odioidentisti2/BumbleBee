import torch

from graphic import depict as draw_molecule, weight_summary

from logging_config import getLogger
logger = getLogger(__name__)


class AttentionExplainer:

    name = "Attention"

    def __init__(self, calibration_data):
        top = calibration_data['attn_factor_mean'] + calibration_data['attn_factor_std']
        self.shift = -1  # shift so that average attention is at 0
        self.factor = 1 / (top + self.shift)  # scale so that top attention is at 1
        self.last_batch_logits = None  # DEBUG: checked in an assert in Explainer base class

        logger.debug(f"ATT top: {top:.2f}")
        logger.debug(f"ATT factor: {self.factor:.2f}")

    def attributions(self, model, batch):
        model.track_attention()
        with torch.no_grad():
            logits = model(batch)  # Forward pass to populate attention store
        weight_list = [aw[:graph.edge_index.size(1)] for aw, graph in zip(model._attention_store, batch.to_data_list())]  # Remove padding
        model.track_attention(enable=False)
        self.last_batch_logits = logits.cpu()  # DEBUG
        return weight_list

    def depict(self, graph, weights, intensity=1):
        # Weights come after softmax (they add up to 1):
        # => weights.mean() = 1 / len(weights)
        # Therefore:
        #   weight * len(weights) == 1  means "average attention"
        attention_ratios = weights * len(weights)  # ratio to average attention

        return draw_molecule(graph, attention_ratios, attention=True,
                             factor=self.factor * intensity, shift=self.shift)

    def summary(self, graph, weights, index):
        logger.debug("ATTENTION-BASED EXPLANATION", highlight=True)
        logger.debug(f"{graph.y.item():.2f} {graph.smiles}")
        logger.debug(weight_summary(weights, average=True, title="ATTENTION WEIGHTS:"))
