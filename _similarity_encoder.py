from bumblebee import *
import time
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

## Inputs
dataset_info = datasets.muta
model_name = 'muta.pt'
start_time = time.time()

trainer = Trainer(dataset_info['task'], device)
model = load(f"MODELS/{model_name}", device)

## Dataset
dataset = GraphDataset(dataset_info, split=dataset_info['test_split'])
dataset_loader = DataLoader(dataset, batch_size=test_batch_size)
trainer.eval(model, dataset_loader, flag="Test")

print(f"\nEvaluation TIME: {time.time() - start_time:.0f}s")

# Import saved data from model.py
from model import enc_repr, att_list

print(f"\nCollected {len(enc_repr)} molecule encoder representations")
print(f"Collected {len(att_list)} molecule attention weights")
print(f"First molecule: {enc_repr[0].shape[0]} tokens, dimension {enc_repr[0].shape[1]}")
import copy
all_predictions = torch.cat(copy.deepcopy(trainer.statistics.stats[-1]['predictions'])).cpu()
all_targets = torch.cat(copy.deepcopy(trainer.statistics.stats[-1]['targets'])).cpu()
print(f"Collected {all_predictions.shape[0]} predictions and {all_targets.shape[0]} targets")

# ============================================================================
# TOKEN EXTRACTION & FILTERING
# ============================================================================
def extract_important_tokens(enc_tokens, pma_attention, threshold='mean'):
    """
    Extract high-attention tokens from encoder output
    
    Args:
        enc_tokens: [num_edges, hidden_dim] - encoder output for one molecule
        pma_attention: [num_edges] - PMA attention weights for one molecule
        threshold: 'mean' or float value
    
    Returns:
        important_tokens: [num_important, hidden_dim]
        important_attention: [num_important]
        selected_indices: [num_important]
    """
    if threshold == 'mean':
        # FIXED: Compute mean only over non-padded (non-zero) tokens
        non_zero_mask = pma_attention > 0
        if non_zero_mask.sum() > 0:
            threshold = pma_attention[non_zero_mask].mean()
        else:
            # Fallback if all zeros (shouldn't happen)
            threshold = pma_attention.mean()
    
    # Filter tokens above threshold
    high_attn_mask = pma_attention > threshold
    
    # Safety check: at least 1 token
    if high_attn_mask.sum() == 0:
        # Fallback: use highest attention token
        high_attn_mask = pma_attention == pma_attention.max()
    
    important_tokens = enc_tokens[high_attn_mask]
    important_attention = pma_attention[high_attn_mask]
    selected_indices = high_attn_mask.nonzero(as_tuple=True)[0]
    
    # Sort by attention (descendec_reprding) - for inspection
    sort_idx = important_attention.argsort(descending=True)
    important_tokens = important_tokens[sort_idx]
    important_attention = important_attention[sort_idx]
    selected_indices = selected_indices[sort_idx]
    
    # Safety: check for NaN
    if torch.isnan(important_tokens).any():
        valid_mask = ~torch.isnan(important_tokens).any(dim=1)
        important_tokens = important_tokens[valid_mask]
        important_attention = important_attention[valid_mask]
        selected_indices = selected_indices[valid_mask]
    
    return important_tokens, important_attention, selected_indices


# Extract important tokens for all molecules
print("\nExtracting important tokens...")
all_important_tokens = []
all_important_attention = []
all_selected_indices = []

for i in range(len(enc_repr)):
    tokens, attn, indices = extract_important_tokens(enc_repr[i], att_list[i])
    all_important_tokens.append(tokens)
    all_important_attention.append(attn)
    all_selected_indices.append(indices)
    
print(f"Extracted important tokens for {len(all_important_tokens)} molecules")
print(f"Average number of important tokens: {sum(len(t) for t in all_important_tokens) / len(all_important_tokens):.1f}")

# ============================================================================
# SOFT JACCARD SIMILARITY
# ============================================================================
def soft_jaccard(tokens_A, tokens_B):
    """
    Compute soft Jaccard similarity between two token sets
    
    Args:
        tokens_A: [num_A, hidden_dim]
        tokens_B: [num_B, hidden_dim]
    
    Returns:
        similarity: float in [0, 1]
    """
    # Pairwise cosine similarity matrix
    sim_matrix = F.cosine_similarity(
        tokens_A.unsqueeze(1),  # [num_A, 1, hidden_dim]
        tokens_B.unsqueeze(0),  # [1, num_B, hidden_dim]
        dim=2
    )  # [num_A, num_B]
    
    # Best matches for each direction
    best_A_to_B = sim_matrix.max(dim=1)[0].sum()  # Sum of max similarities for A→B
    best_B_to_A = sim_matrix.max(dim=0)[0].sum()  # Sum of max similarities for B→A
    
    # Soft intersection (symmetric)
    soft_intersection = (best_A_to_B + best_B_to_A) / 2
    
    # Soft union
    soft_union = len(tokens_A) + len(tokens_B) - soft_intersection
    
    # Avoid division by zero
    if soft_union == 0:
        return 1.0
    
    return (soft_intersection / soft_union).item()


def find_similar_molecules_tokens(query_idx, all_tokens, top_k=5):
    """
    Find molecules with similar important token sets
    
    Args:
        query_idx: Index of query molecule
        all_tokens: List of [num_important, hidden_dim] tensors
        top_k: Number of similar molecules to return
    
    Returns:
        List of (index, similarity_score) tuples
    """
    query_tokens = all_tokens[query_idx]
    similarities = []
    
    for i, tokens in enumerate(all_tokens):
        if i == query_idx:
            continue  # Skip self-comparison
        
        sim = soft_jaccard(query_tokens, tokens)
        similarities.append((i, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


# ============================================================================
# VISUALIZATION
# ============================================================================
def visualize_similar_molecules_tokens(query_idx, similar_indices, dataset, 
                                       similarities=None, num_tokens_list=None,
                                       save_path=None):
    """
    Visualize query molecule and its most similar matches (token-level)
    """
    query_data = dataset[query_idx]
    query_mol = Chem.MolFromSmiles(query_data.smiles)
    
    similar_mols = []
    labels = []
    
    # Query molecule
    query_label = f"Query (idx={query_idx})"
    if num_tokens_list:
        query_label += f"\n{num_tokens_list[0]} important tokens"
    query_label += f"\n{query_data.smiles[:30]}..."
    labels.append(query_label)
    similar_mols.append(query_mol)
    
    # Similar molecules
    for i, idx in enumerate(similar_indices):
        similar_data = dataset[idx]
        mol = Chem.MolFromSmiles(similar_data.smiles)
        
        label = f"Match {i+1} (idx={idx})"
        if similarities:
            label += f"\nToken Sim: {similarities[i]:.3f}"
        if num_tokens_list:
            label += f"\n{num_tokens_list[i+1]} tokens"
        label += f"\n{similar_data.smiles[:25]}..."
        
        labels.append(label)
        similar_mols.append(mol)
    
    # Draw molecules
    img = Draw.MolsToGridImage(
        similar_mols,
        molsPerRow=3,
        subImgSize=(350, 350),
        legends=labels,
        returnPNG=False
    )
    
    if save_path:
        img.save(save_path)
        print(f"Saved visualization to {save_path}")
    
    return img


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
query_idx = 1

print(f"\n{'='*60}")
print(f"Finding molecules with similar TOXICITY PATTERNS (token-level)")
print(f"Query molecule #{query_idx}")
print(f"{'='*60}\n")

print(f"Query SMILES: {dataset[query_idx].smiles}")
print(f"Query target: {all_targets[query_idx].item():.3f}")
print(f"Query prediction: {all_predictions[query_idx].item():.3f}")
query_attention = att_list[query_idx]
non_zero_mask = query_attention > 0
num_non_zero = non_zero_mask.sum()
threshold = query_attention[non_zero_mask].mean() if num_non_zero > 0 else query_attention.mean()
print(f"Attention threshold (mean): {threshold:.6f}")
print(f"Attention range: [{query_attention.min():.6f}, {query_attention.max():.6f}]")
print(f"Number of important tokens: {len(all_important_tokens[query_idx])}")
print(f"Important token attention weights: {all_important_attention[query_idx].tolist()}")
print()

# Find similar molecules
similar = find_similar_molecules_tokens(query_idx, all_important_tokens, top_k=10)

print("Most similar molecules (by token-level similarity):")
print(f"{'='*60}\n")

for rank, (idx, sim) in enumerate(similar, 1):
    mol_data = dataset[idx]
    print(f"{rank}. Index: {idx} | Token Similarity: {sim:.4f}")
    print(f"   Target: {all_targets[idx].item():.3f}")
    print(f"   Prediction: {all_predictions[idx].item():.3f}")
    print(f"   Important tokens: {len(all_important_tokens[idx])}")
    print(f"   SMILES: {mol_data.smiles}")
    print()

# Visualize
similar_indices = [idx for idx, _ in similar]
similarities = [sim for _, sim in similar]
num_tokens = [len(all_important_tokens[query_idx])] + [len(all_important_tokens[idx]) for idx in similar_indices]

img = visualize_similar_molecules_tokens(
    query_idx, 
    similar_indices, 
    dataset, 
    similarities,
    num_tokens,
    save_path=f"{query_idx}_encoder_similarity.png"
)

# ============================================================================
# COMPARISON: Token-level vs Graph-level Similarity
# ============================================================================
print(f"\n{'='*60}")
print("COMPARISON: Token-level vs Graph-level Similarity")
print(f"{'='*60}\n")

# Import decoder (graph-level) representations
from model import dec_repr

# Stack decoder representations for cosine similarity
dec_repr_tensor = torch.stack(dec_repr)  # [num_molecules, hidden_dim]

# Compute graph-level similarities for the same query
query_dec_repr = dec_repr_tensor[query_idx]
graph_similarities = []

for i in range(len(dec_repr)):
    if i == query_idx:
        continue
    graph_sim = F.cosine_similarity(
        query_dec_repr.unsqueeze(0),
        dec_repr_tensor[i].unsqueeze(0)
    ).item()
    graph_similarities.append((i, graph_sim))

graph_similarities.sort(key=lambda x: x[1], reverse=True)

# Create lookup for graph-level similarities
graph_sim_dict = {idx: sim for idx, sim in graph_similarities}

# Compare token-level vs graph-level for top matches
print(f"For query molecule {query_idx}:\n")
print(f"{'Rank':<6}{'Index':<8}{'Token Sim':<12}{'Graph Sim':<12}{'Tokens':<10}{'Target':<10}{'Pred'}")
print("-" * 80)

for rank, (idx, token_sim) in enumerate(similar[:10], 1):
    graph_sim = graph_sim_dict.get(idx, 0.0)
    num_tok = len(all_important_tokens[idx])
    target = all_targets[idx].item()
    pred = all_predictions[idx].item()
    
    print(f"{rank:<6}{idx:<8}{token_sim:<12.3f}{graph_sim:<12.3f}{num_tok:<10}{target:<10.3f}{pred:.3f}")

print(f"\n{'='*60}")
print("Top-5 by GRAPH-LEVEL similarity (for comparison):")
print(f"{'='*60}\n")

for rank, (idx, graph_sim) in enumerate(graph_similarities[:5], 1):
    # Find this molecule's token similarity
    token_sim = None
    for t_idx, t_sim in similar:
        if t_idx == idx:
            token_sim = t_sim
            break
    
    num_tok = len(all_important_tokens[idx])
    target = all_targets[idx].item()
    pred = all_predictions[idx].item()
    
    print(f"{rank}. Index: {idx}")
    print(f"   Graph Similarity: {graph_sim:.4f}")
    if token_sim is not None:
        print(f"   Token Similarity: {token_sim:.4f}")
    else:
        print("   Token Similarity: Not in top-10")
    print(f"   Important tokens: {num_tok}")
    print(f"   Target: {target:.3f} | Prediction: {pred:.3f}")
    print(f"   SMILES: {dataset[idx].smiles}")
    print()

print(f"\n{'='*60}")
print("Analysis complete!")
print(f"{'='*60}\n")

# Statistics
token_similarities = [sim for _, sim in similar[:20]]
graph_similarities_top20 = [graph_sim_dict[idx] for idx, _ in similar[:20]]

print(f"Token-level similarity statistics (top 20 matches):")
print(f"  Mean: {sum(token_similarities)/len(token_similarities):.3f}")
print(f"  Min:  {min(token_similarities):.3f}")
print(f"  Max:  {max(token_similarities):.3f}")

print(f"\nGraph-level similarity for same molecules:")
print(f"  Mean: {sum(graph_similarities_top20)/len(graph_similarities_top20):.3f}")
print(f"  Min:  {min(graph_similarities_top20):.3f}")
print(f"  Max:  {max(graph_similarities_top20):.3f}")

# Correlation analysis
import numpy as np
correlation = np.corrcoef(token_similarities, graph_similarities_top20)[0, 1]
print(f"\nCorrelation between token-level and graph-level similarity: {correlation:.3f}")

print(f"\n\nTOTAL TIME: {time.time() - start_time:.0f}s")