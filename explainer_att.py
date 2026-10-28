import torch

from graphic import depict, weight_summary

from logging_config import getLogger
logger = getLogger(__name__)


class Att_Depicter():

    def __init__(self, top):
        self.shift = -1  # shift so that average attention is at 0
        self.factor = 1 / (top + self.shift)  # scale so that top attention is at 1

        logger.debug(f"ATT top: {top:.2f}")
        logger.debug(f"ATT factor: {self.factor:.2f}")

    def depict(self, graph, weights, intensity=1):
        logger.debug("ATTENTION-BASED EXPLANATION", highlight=True)
        logger.debug(f"{graph.y.item():.2f} {graph.smiles}")
        logger.debug(weight_summary(weights, average=True, title="ATTENTION WEIGHTS:"))

        return depict(graph, self.att_ratios(weights), attention=True, factor=self.factor * intensity, shift=self.shift)

    @staticmethod
    def att_ratios(weights):
        # Weights come after softmax (they add up to 1): 
        # => weights.mean() = 1 / len(weights)
        # Therefore:
        #   weight * len(weights) == 1  means "average attention"
        ratios = weights * len(weights)  # ratio to average attention
        return ratios


class AttentionExplainer:

    name = "Attention"

    def __init__(self, calibration_data):
        self.depicter = Att_Depicter(top=calibration_data['attn_factor_mean'] + calibration_data['attn_factor_std'])

    def attributions(self, model, batch):
        model.track_attention()
        with torch.no_grad():
            _ = model(batch)  # Forward pass to populate attention store
        weight_list = [aw[:graph.edge_index.size(1)] for aw, graph in zip(model._attention_store, batch.to_data_list())]  # Remove padding
        model.track_attention(enable=False)
        return weight_list

    # def annotate(self, graph, index):
    #     pass

    def depict(self, graph, weights, index, intensity=1):
        return self.depicter.depict(graph, weights, intensity=intensity)