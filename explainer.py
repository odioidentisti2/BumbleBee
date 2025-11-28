import numpy as np
import torch
from torch_geometric.data import Batch
from graphic import *


# def sum_bond_weights(edge_index, weights):
#     # edge_index: shape [2, num_edges]
#     # weights: shape [num_edges]
#     src = edge_index[0]
#     dst = edge_index[1]
#     # Make undirected: always (min, max)
#     bond_pairs = torch.stack([torch.minimum(src, dst), torch.maximum(src, dst)], dim=1)  # [num_edges, 2]
#     # Find unique bonds and sum weights
#     bond_keys, inverse_indices = torch.unique(bond_pairs, dim=0, return_inverse=True)
#     summed_weights = torch.zeros(len(bond_keys), dtype=weights.dtype, device=weights.device)
#     summed_weights.scatter_add_(0, inverse_indices, weights)
#     return bond_keys, summed_weights

def get_upper_limits(model, calibration_loader):
    predictions = []
    train_attn_weights = []
    with torch.no_grad():
        for batch in calibration_loader:
            batch = batch.to('cpu')
            preds, attn_weights = model(batch, return_attention=True)
            predictions.extend(preds)
            train_attn_weights.extend(attn_weights)

    # IG
    ig_dist = np.array(predictions)
    ig_iqr = np.percentile(ig_dist, 75) - np.percentile(ig_dist, 25)
    max_abs_ig = ig_iqr / 4
    print("\nIG Intensity calibration:")
    print("IQR = ", ig_iqr)
    print("STD =", ig_dist.std())
    print(f"dist range: {ig_dist.min()} - {ig_dist.max()}")
    print(f"Max IG intensity to: {max_abs_ig:.4f}")

    # Att
    att_factor_dist = np.array([aw.max() * len(aw) for aw in train_attn_weights])
    max_att_factor = att_factor_dist.mean() + att_factor_dist.std()
    att_iqr = np.percentile(att_factor_dist, 75) - np.percentile(att_factor_dist, 25)
    print("\nAttention Intensity calibration:")
    print("IGR =", att_iqr)
    print("STD =", att_factor_dist.std())
    print(f"dist range: {att_factor_dist.min():.2f} - {att_factor_dist.max():.2f}")
    print(f"Max Attention intensity to: {max_att_factor:.4f}")
    return max_att_factor, max_abs_ig

# DEBUG
def print_weights(weights, average=False):
    print("\nWEIGHTS:")
    print(weights)
    print(f"Weights range: {weights.min():.4f} - {weights.max():.4f}")
    if average: print("Weights Average: ", weights.mean().item())
    print(f"Weight sum: {weights.sum():.2f}")


class Explainer:

    def __init__(self, model):
        self.model = model.to('cpu')
        self.intensity = 1
        self.att_factor_top = 10
        self.ig_top = None

    def calibrate(self, calibration_loader):
        self.att_factor_top, self.ig_top = get_upper_limits(self.model, calibration_loader)

    def attention(self, graph, intensity=1):
        graph = graph.to('cpu')
        batched_graph = Batch.from_data_list([graph])
        # edge_feat = model.get_features(batched_graph)
        with torch.no_grad():
            _, weights = self.model(batched_graph, return_attention=True)
            # _, weights = model.single_forward(edge_feat, batched_graph.edge_index, batched_graph.batch, return_attention=True)[0]  # remove batch
        weights = weights[0].detach().cpu()  # remove batch

        print("\nDEPICT ATTENTION")
        print_weights(weights, average=True)
        # Weights come after softmax (they add up to 1): 
        # => weights.mean() = 1 / len(weights)
        # Therefore:
        # - weight * len(weights) == 1  means "average attention"
        # - weight * len(weights) > 1  means "increased attention"
        # - weight * len(weights) < 1  means "decreased attention" => clip
        # I measure how many times above average each weight is and I normalize [1, top] to [0, 1])
        attention_factors = weights * len(weights)
        clip_weights = torch.clip(attention_factors, 1, self.att_factor_top)
        norm_weights = (clip_weights - 1) / (self.att_factor_top - 1)
        depict(graph, norm_weights.numpy()*intensity)


    def integrated_gradients(self, graph, steps=100, intensity=1):
        """Integrated gradients explanation for edge features"""
        graph = graph.to('cpu')
        batched_graph = Batch.from_data_list([graph])
        edge_feat = self.model.get_features(batched_graph)   
        baseline = torch.zeros_like(edge_feat)    # TRY MEANINGFUL BASELINE!
        integrated_grads = torch.zeros_like(edge_feat)

        for alpha in torch.linspace(0, 1, steps):  # alpha on cpu by default
            # Interpolate between baseline and input
            interp_feat = baseline + alpha * (edge_feat - baseline)
            interp_feat.requires_grad_(True)
            # Forward pass
            prediction = self.model.single_forward(interp_feat, batched_graph.edge_index, batched_graph.batch)
            integrated_grads += torch.autograd.grad(
                                outputs=prediction,
                                inputs=interp_feat,
                                create_graph=False
                            )[0]
        
        # Average gradients and scale by input difference
        integrated_grads /= steps
        attributions = (edge_feat - baseline) * integrated_grads
        edge_importance = attributions.sum(dim=1)  # Sum across feature dimensions

        print("\n\nDEPICT INTEGRATED GRADIENTS")
        print(int(graph.y.item()), graph.smiles)
        # Get baseline and final predictions for verification
        with torch.no_grad():
            baseline_pred = self.model.single_forward(baseline, batched_graph.edge_index, batched_graph.batch)
            final_pred = self.model.single_forward(edge_feat, batched_graph.edge_index, batched_graph.batch)
        # Verify the sum property (CRITICAL for IG correctness)
        attribution_sum = edge_importance.sum().item()
        # expected_sum = (final_pred - baseline_pred).item()    
        print(f"\nBaseline prediction: {baseline_pred.item():.2f}")
        print(f"Attribution sum: {attribution_sum:.2f}")
        print(f"Baseline + Attribution sum: {baseline_pred.item() + attribution_sum:.2f}")    
        print(f"PREDICTION: {final_pred.item():.2f}")

        weights = edge_importance.detach().cpu()
        print_weights(weights)
        clip_weights = torch.clip(weights, -self.ig_top, self.ig_top)
        norm_weights = clip_weights / self.ig_top

        # Before depict I should normalize edge_importance by 0.5 - baseline
        depict(graph, norm_weights.numpy() * intensity, attention=False)
        # depict(graph, weights.numpy() / (0.5 - baseline_pred.item()), attention=False)


    def explain_with_mlp_IG(self, graph, steps=50, intensity=1.0):
        """
        Integrated gradients for the output of input_mlp (edge embeddings).
        """
        graph = graph.to('cpu')

        # Get edge features and baseline
        batched_graph = Batch.from_data_list([graph])
        edge_feat = self.model.get_features(batched_graph)
        baseline_emb = torch.zeros_like(self.model.input_mlp(edge_feat))
        edge_emb = self.model.input_mlp(edge_feat).detach()

        integrated_grads = torch.zeros_like(edge_emb)

        for alpha in torch.linspace(0, 1, steps):
            interp_emb = baseline_emb + alpha * (edge_emb - baseline_emb)
            interp_emb = interp_emb.detach().clone().requires_grad_(True)

            # --- Forward hook trick: temporarily replace input_mlp's forward to return interp_emb ---
            orig_forward = self.model.input_mlp.forward
            self.model.input_mlp.forward = lambda *_: interp_emb

            # Forward pass
            pred = self.model(batched_graph)
            pred = torch.flatten(pred)[0]

            # Backward pass
            self.model.zero_grad()
            pred.backward()
            grad = interp_emb.grad.detach()
            integrated_grads += grad

            # Restore original forward
            self.model.input_mlp.forward = orig_forward

        integrated_grads /= steps
        attributions = (edge_emb - baseline_emb) * integrated_grads
        edge_importance = attributions.sum(dim=1)  # Sum across embedding dimensions

        # After computing attributions
        pred_real = float(self.model(batched_graph).detach().cpu().numpy())
        # Temporarily override input_mlp to always return baseline_emb
        orig_forward = self.model.input_mlp.forward
        self.model.input_mlp.forward = lambda *_: baseline_emb
        pred_base = float(self.model(batched_graph).detach().cpu().numpy())
        self.model.input_mlp.forward = orig_forward

        weights = edge_importance.detach().cpu()


        print("\nDEPICT MLP INTEGRATED GRADIENTS")
        print(f"Prediction (real): {pred_real:.2f}")
        print(f"Prediction (baseline): {pred_base:.2f}")
        print(f"Sum of attributions: {edge_importance.sum():.2f}")
        print(f"Difference: {pred_real - pred_base:.2f}")
        print_weights(weights)
        depict(graph, weights.numpy() * intensity/ 10, attention=False)
        return edge_importance


#     # Place this inside explain_with_attention
#     # In the depiction of attention, I highlight bonds with an intensity proportional to their attention weights.
#     # I'd like this intensity to be "absolute" (referred to the whole training set).
#     # But attention weights are very relative to each molecule (and proportionate to the number of bonds, after softmax).
#     # So I need to find a "max intensity" threshold from the training set, to normalize it,
#     # I can divide by the mean attention weight for each molecule.
#     # In conclusion: given the distribution of attention max/mean ratio in the training set, I can set a threshold
#     # (e.g. mean + std) to be the "max intensity" (1.0) in the depiction, so that only "outlier" weights
#     # are fully highlighted. In other words, let's say that the maximum max/mean ratio in any molecule
#     # in the training set was 30 (30 times more attention than the average), mean + std will be lower than that,
#     # let's say 7, so in the depiction an attention weight 7 times higher than the average
#     # will be highlighted with intensity 1.0.
#     train_attn_weights = []
#     for batch in train_loader:
#         batch = batch.to(DEVICE)
#         with torch.no_grad():
#             train_attn_weights.extend(model(batch, return_attention=True))
#     dist = np.array([aw.max() / aw.mean() for aw in train_attn_weights])
#     top = dist.mean() + dist.std()