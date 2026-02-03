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
dataset_loader = DataLoader(dataset, batch_size=optimal_batch_size)
trainer.eval(model, dataset_loader, flag="Test")

print(f"\nEvaluation TIME: {time.time() - start_time:.0f}s")

# Import saved data from model.py
from model import repr_list, att_list

print(f"\nCollected {len(repr_list)} molecule encoder representations")
print(f"Collected {len(att_list)} molecule attention weights")
print(f"First molecule: {repr_list[0].shape[0]} tokens, dimension {repr_list[0].shape[1]}")

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
    
    # Sort by attention (descending)
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
        print(f"Warning: Removed {(~valid_mask).sum()} NaN tokens")
    
    return important_tokens, important_attention, selected_indices


# Extract important tokens for all molecules
print("\nExtracting important tokens...")
all_important_tokens = []
all_important_attention = []
all_selected_indices = []

for i in range(len(repr_list)):
    tokens, attn, indices = extract_important_tokens(repr_list[i], att_list[i])
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

query_idx = 0

print(f"\n{'='*60}")
print(f"Finding molecules with similar TOXICITY PATTERNS (token-level)")
print(f"Query molecule #{query_idx}")
print(f"{'='*60}\n")

print(f"Query SMILES: {dataset[query_idx].smiles}")
print(f"Query target: {all_targets[query_idx].item():.3f}")
print(f"Query prediction: {all_predictions[query_idx].item():.3f}")
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
    save_path=f"{query_idx}_token_similarity.png"
)

# ============================================================================
# COMPARISON: Token-level vs Graph-level Similarity
# ============================================================================

print(f"\n{'='*60}")
print("COMPARISON: Token-level vs PMA Graph-level Similarity")
print(f"{'='*60}\n")

# Load PMA representations (from _hidden_dim_PMA.py results)
# Assuming repr_list from model.py contains decoder output
from _similarity_decoder import find_similar_molecules as find_similar_pma

# For each top token-match, check PMA similarity
print(f"For query molecule {query_idx}:\n")
print(f"{'Rank':<6}{'Index':<8}{'Token Sim':<12}{'Token Count':<14}{'Target':<10}{'Prediction'}")
print("-" * 70)

for rank, (idx, token_sim) in enumerate(similar[:5], 1):
    num_tok = len(all_important_tokens[idx])
    target = all_targets[idx].item()
    pred = all_predictions[idx].item()
    
    print(f"{rank:<6}{idx:<8}{token_sim:<12.3f}{num_tok:<14}{target:<10.3f}{pred:.3f}")

print(f"\n{'='*60}")
print("Analysis complete!")
print(f"{'='*60}\n")

# Statistics
token_similarities = [sim for _, sim in similar[:20]]
print(f"Token similarity statistics (top 20 matches):")
print(f"  Mean: {sum(token_similarities)/len(token_similarities):.3f}")
print(f"  Min:  {min(token_similarities):.3f}")
print(f"  Max:  {max(token_similarities):.3f}")

print(f"\n\nTOTAL TIME: {time.time() - start_time:.0f}s")