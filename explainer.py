import torch
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

def explain_with_attention(model, single_batch, intensity=1):
    top = 7.77
    with torch.no_grad():
        graph = single_batch.to_data_list()[0]
        # weights = model(single_batch, return_attention=True)[0]  # single_batch! (I should fix it at the origin)

        edge_feat = model.get_features(single_batch)
        weights = model.single_forward(edge_feat, single_batch.edge_index, single_batch.batch, return_attention=True)[0]

        # depict(graph, weights.numpy() * len(weights) / 10, attention=True)
        ratios = weights / weights.mean()  # Relative to this molecule
        norm_weights = (torch.clip(ratios, 1, top) - 1) / (top - 1)  # clipping upper and lower (no need threshold)
        # norm_weights = torch.clip(ratios / top, 0, 1)  # Scale by training threshold
        print("\nDEPICT ATTENTION")
        depict(graph, norm_weights.numpy()*intensity)

def explain_with_gradients(model, single_batch, steps=5, intensity=1):
    """Integrated gradients explanation for edge features"""
    edge_feat = model.get_features(single_batch)   
    baseline = torch.zeros_like(edge_feat)    
    integrated_grads = torch.zeros_like(edge_feat)

    # Get baseline and final predictions for verification
    with torch.no_grad():
        baseline_pred = model.single_forward(baseline, single_batch.edge_index, single_batch.batch)
        final_pred = model.single_forward(edge_feat, single_batch.edge_index, single_batch.batch)
    print(f"\nBaseline prediction: {baseline_pred.item():.4f}")
    # print(f"Expected attribution sum: {(final_pred - baseline_pred).item():.4f}")


    for i, alpha in enumerate(torch.linspace(0, 1, steps)):
        # print(f"  Step {i+1}/{steps}: alpha={alpha:.2f}")
        # Interpolate between baseline and input
        interp_feat = baseline + alpha * (edge_feat - baseline)
        interp_feat.requires_grad_(True)
        
        # Forward pass
        prediction = model.single_forward(interp_feat, single_batch.edge_index, single_batch.batch)

        grad = torch.autograd.grad(
            outputs=prediction,
            inputs=interp_feat,
            create_graph=False
        )[0]
        integrated_grads += grad
    
    # Average gradients and scale by input difference
    integrated_grads /= steps
    attributions = (edge_feat - baseline) * integrated_grads
    edge_importance = attributions.sum(dim=1)  # Sum across feature dimensions
    
    # Verify the sum property (CRITICAL for IG correctness)
    attribution_sum = edge_importance.sum().item()
    expected_sum = (final_pred - baseline_pred).item()
    
    print(f"Attribution sum: {attribution_sum:.2f}")
    print(f"Baseline + Attribution sum: {baseline_pred.item() + attribution_sum:.2f}")    
    print(f"PREDICTION: {final_pred.item():.2f}")

    graph = single_batch.to_data_list()[0]
    weights = edge_importance.detach().cpu()
    # Before depict I should normalize edge_importance by 0.5 - baseline
    print("\nDEPICT EDGE IMPORTANCE")
    depict(graph, weights.numpy() * intensity, attention=False)
    # depict(graph, weights.numpy() / (0.5 - baseline_pred.item()), attention=False)

def explain_with_mlp_integrated_gradients(model, single_batch, steps=50, intensity=1.0):
    """
    Integrated gradients for the output of input_mlp (edge embeddings).
    """
    single_batch = single_batch.cpu()
    graph = single_batch.to_data_list()[0]

    # Get edge features and baseline
    edge_feat = model.get_features(single_batch)
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
        pred = model(single_batch)
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
    pred_real = float(model(single_batch).detach().cpu().numpy())
    # Temporarily override input_mlp to always return baseline_emb
    orig_forward = model.input_mlp.forward
    model.input_mlp.forward = lambda *_: baseline_emb
    pred_base = float(model(single_batch).detach().cpu().numpy())
    model.input_mlp.forward = orig_forward

    print(f"Prediction (real): {pred_real:.2f}")
    print(f"Prediction (baseline): {pred_base:.2f}")
    print(f"Sum of attributions: {edge_importance.sum():.2f}")
    print(f"Difference: {pred_real - pred_base:.2f}")

    print("\nDEPICT MLP EDGE IMPORTANCE (Integrated Gradients, after input_mlp)")
    depict(graph, edge_importance.numpy() * intensity/ 10, attention=False)
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