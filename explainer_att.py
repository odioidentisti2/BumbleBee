import torch

from explainer import Explainer

from logging_config import getLogger
logger = getLogger(__name__)


class AttentionExplainer(Explainer):

    name = "Attention"
    color_scheme = 'yellow'

    def __init__(self, calibration_data):
        super().__init__(calibration_data)
        top = calibration_data['attn_factor_mean'] + calibration_data['attn_factor_std']
        self.shift = -1  # shift so that average attention is at 0
        self.factor = 1 / (top + self.shift)  # scale so that top attention is at 1
        
        logger.debug(f"ATT top: {top:.2f}")
        logger.debug(f"ATT factor: {self.factor:.2f}")

    def attributions(self, model, batch):
        model.track_attention()
        with torch.no_grad():
            logits = model(batch)  # Forward pass to populate attention store
        weight_list = [aw[:graph.edge_index.size(1)] for aw, graph in zip(model._attention_store, batch.to_data_list())]  # Remove padding
        model.track_attention(enable=False)
        self.batch_logits = logits.cpu()  # DEBUG
        return weight_list
    
    def _transform(self, weights):
        # Weights come after softmax (they add up to 1):
        # => weights.mean() = 1 / len(weights)
        # Therefore:
        #   weight * len(weights) == 1  means "average attention"
        attention_ratios = weights * len(weights)  # ratio to average attention
        return super()._transform(attention_ratios)
    
    def _summary(self, weights):
        super()._summary(weights, print_mean=True)
