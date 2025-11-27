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

# DEBUG
def print_weights(weights, average=False):
    print("\nWEIGHTS:")
    print(weights)
    print(f"Weights range: {weights.min():.4f} - {weights.max():.4f}")
    if average: print("Weights Average: ", weights.mean().item())
    print(f"Weight sum: {weights.sum():.2f}")


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
    max_ig_intensity = ig_iqr / 4
    print("\nIG Intensity calibration:")
    print("\nIQR = ", ig_iqr)
    print("STD =", ig_dist.std())
    print(f"dist range: {ig_dist.min()} - {ig_dist.max()}")
    print(f"Max IG intensity to: {max_ig_intensity:.4f}")

    # Att
    att_dist = np.array([aw.max() * len(aw) for aw in train_attn_weights])
    max_att_intensity = att_dist.mean() + att_dist.std()
    att_iqr = np.percentile(att_dist, 75) - np.percentile(att_dist, 25)
    print("\nAttention Intensity calibration:")
    print("IGR =", att_iqr)
    print("STD =", att_dist.std())
    print(f"dist range: {att_dist.min():.2f} - {att_dist.max():.2f}")
    print(f"Max Attention intensity to: {max_att_intensity:.4f}")
    return 1 / max_att_intensity, 1 / max_ig_intensity


def explain_with_attention(model, graph, intensity=1):
    print("\nDEPICT ATTENTION")
    top = 7.77  # if w > average weight above amount of times,  then clip to 1
    batched_graph = Batch.from_data_list([graph])
    # edge_feat = model.get_features(batched_graph)
    with torch.no_grad():
        _, weights = model(batched_graph, return_attention=True)
        # _, weights = model.single_forward(edge_feat, batched_graph.edge_index, batched_graph.batch, return_attention=True)[0]  # remove batch
    weights = weights[0]  # remove batch
    print_weights(weights, average=True)
    # depict(graph, weights.numpy() * len(weights) / 10, attention=True)
    # weights come after softmax (they add up to 1): 
    # - weight * len(weights) > 1  means "increased attention"
    # - weight * len(weights) < 1  means "decreased attention" => clip
    ratios = weights * len(weights) # / weights.mean()  # Relative to this molecule
    norm_weights = (torch.clip(ratios, 1, top) - 1) / (top - 1)  # clipping upper and lower (no need threshold)
    # norm_weights = torch.clip(ratios / top, 0, 1)  # Scale by training threshold
    depict(graph, norm_weights.numpy()*intensity)


def explain_with_IG(model, graph, steps=100, intensity=1):
    """Integrated gradients explanation for edge features"""
    batched_graph = Batch.from_data_list([graph])
    edge_feat = model.get_features(batched_graph)   
    baseline = torch.zeros_like(edge_feat)    # TRY MEANINGFUL BASELINE!
    integrated_grads = torch.zeros_like(edge_feat)

    for alpha in torch.linspace(0, 1, steps):  # alpha on cpu by default
        # Interpolate between baseline and input
        interp_feat = baseline + alpha * (edge_feat - baseline)
        interp_feat.requires_grad_(True)
        # Forward pass
        prediction = model.single_forward(interp_feat, batched_graph.edge_index, batched_graph.batch)
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
        baseline_pred = model.single_forward(baseline, batched_graph.edge_index, batched_graph.batch)
        final_pred = model.single_forward(edge_feat, batched_graph.edge_index, batched_graph.batch)
    # Verify the sum property (CRITICAL for IG correctness)
    attribution_sum = edge_importance.sum().item()
    # expected_sum = (final_pred - baseline_pred).item()    
    print(f"\nBaseline prediction: {baseline_pred.item():.2f}")
    print(f"Attribution sum: {attribution_sum:.2f}")
    print(f"Baseline + Attribution sum: {baseline_pred.item() + attribution_sum:.2f}")    
    print(f"PREDICTION: {final_pred.item():.2f}")

    weights = edge_importance.detach().cpu()
    print_weights(weights)

    # Before depict I should normalize edge_importance by 0.5 - baseline
    depict(graph, weights.numpy() * intensity, attention=False)
    # depict(graph, weights.numpy() / (0.5 - baseline_pred.item()), attention=False)


def explain_with_mlp_IG(model, graph, steps=50, intensity=1.0):
    """
    Integrated gradients for the output of input_mlp (edge embeddings).
    """

    # Get edge features and baseline
    batched_graph = Batch.from_data_list([graph])
    edge_feat = model.get_features(batched_graph)
    baseline_emb = torch.zeros_like(model.input_mlp(edge_feat))
    edge_emb = model.input_mlp(edge_feat).detach()

    integrated_grads = torch.zeros_like(edge_emb)

    for alpha in torch.linspace(0, 1, steps):
        interp_emb = baseline_emb + alpha * (edge_emb - baseline_emb)
        interp_emb = interp_emb.detach().clone().requires_grad_(True)

        # --- Forward hook trick: temporarily replace input_mlp's forward to return interp_emb ---
        orig_forward = model.input_mlp.forward
        model.input_mlp.forward = lambda *_: interp_emb

        # Forward pass
        pred = model(batched_graph)
        pred = torch.flatten(pred)[0]

        # Backward pass
        model.zero_grad()
        pred.backward()
        grad = interp_emb.grad.detach()
        integrated_grads += grad

        # Restore original forward
        model.input_mlp.forward = orig_forward

    integrated_grads /= steps
    attributions = (edge_emb - baseline_emb) * integrated_grads
    edge_importance = attributions.sum(dim=1)  # Sum across embedding dimensions

    # After computing attributions
    pred_real = float(model(batched_graph).detach().cpu().numpy())
    # Temporarily override input_mlp to always return baseline_emb
    orig_forward = model.input_mlp.forward
    model.input_mlp.forward = lambda *_: baseline_emb
    pred_base = float(model(batched_graph).detach().cpu().numpy())
    model.input_mlp.forward = orig_forward

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