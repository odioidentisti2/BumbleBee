from bumblebee import *
import time
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

## Inputs
# dataset_info = datasets.logp_split
# model_name = 'logp.pt'
dataset_info = datasets.muta
model_name = 'muta.pt'
start_time = time.time()

trainer = Trainer(dataset_info['task'], device)
model = load(f"MODELS/{model_name}", device)

## Test
testset = GraphDataset(dataset_info, split=dataset_info['test_split'])
test_loader = DataLoader(testset, batch_size=optimal_batch_size)
trainer.eval(model, test_loader, flag="Test")

print(f"\nEvaluation TIME: {time.time() - start_time:.0f}s")

from model import repr_list, att_list

print(f"\nCollected {len(repr_list)} molecule representations")
print(f"Representation dimension: {repr_list[0].shape[0]}")

import copy
all_predictions = torch.cat(copy.deepcopy(trainer.statistics.stats[-1]['predictions'])).cpu()
all_targets = torch.cat(copy.deepcopy(trainer.statistics.stats[-1]['targets'])).cpu()
print(f"Collected {all_predictions.shape[0]} predictions and {all_targets.shape[0]} targets")

# ============================================================================
# SIMILARITY SEARCH
# ============================================================================

def find_similar_molecules(query_idx, representations, top_k=5, metric='cosine'):
    """
    Find molecules with similar representations
    
    Args:
        query_idx: Index of query molecule
        representations: List of [hidden_dim] tensors
        top_k: Number of similar molecules to return
        metric: 'cosine' or 'euclidean'
    
    Returns:
        List of (index, similarity_score) tuples
    """
    query_repr = representations[query_idx]
    similarities = []
    
    for i, repr in enumerate(representations):
        if i == query_idx:
            continue  # Skip self-comparison
        
        if metric == 'cosine':
            sim = F.cosine_similarity(
                query_repr.unsqueeze(0), 
                repr.unsqueeze(0)
            ).item()
        elif metric == 'euclidean':
            dist = torch.dist(query_repr, repr, p=2).item()
            sim = 1.0 / (1.0 + dist)  # Convert distance to similarity
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        similarities.append((i, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


def visualize_similar_molecules(query_idx, similar_indices, dataset, 
                                 similarities=None, save_path=None):
    """
    Visualize query molecule and its most similar matches
    
    Args:
        query_idx: Index of query molecule
        similar_indices: List of indices of similar molecules
        dataset: GraphDataset containing SMILES strings
        similarities: Optional list of similarity scores
        save_path: Optional path to save the figure
    """
    # Get molecules
    query_data = dataset[query_idx]
    query_mol = Chem.MolFromSmiles(query_data.smiles)
    
    similar_mols = []
    labels = []
    
    # Query molecule
    labels.append(f"Query (idx={query_idx})\n{query_data.smiles[:30]}...")
    similar_mols.append(query_mol)
    
    # Similar molecules
    for i, idx in enumerate(similar_indices):
        similar_data = dataset[idx]
        mol = Chem.MolFromSmiles(similar_data.smiles)
        
        if similarities:
            label = f"Match {i+1} (idx={idx})\nSim: {similarities[i]:.3f}\n{similar_data.smiles[:30]}..."
        else:
            label = f"Match {i+1} (idx={idx})\n{similar_data.smiles[:30]}..."
        
        labels.append(label)
        similar_mols.append(mol)
    
    # Draw molecules
    img = Draw.MolsToGridImage(
        similar_mols,
        molsPerRow=3,
        subImgSize=(300, 300),
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

# Stack representations for efficient batch processing
repr_tensor = torch.stack(repr_list)  # [num_molecules, hidden_dim]
print(f"Representation tensor shape: {repr_tensor.shape}")

# Choose a query molecule (e.g., first one)
query_idx = 1

print(f"\n{'='*60}")
print(f"Finding molecules similar to molecule #{query_idx}")
print(f"Query SMILES: {testset[query_idx].smiles}")
print(f"{'='*60}\n")

# Find similar molecules
similar = find_similar_molecules(query_idx, repr_list, top_k=5, metric='cosine')

print(f"Query molecule:")
print(f"  Target: {all_targets[query_idx].item():.3f}")
print(f"  Prediction: {all_predictions[query_idx].item():.3f}\n")

print("Most similar molecules:")
for rank, (idx, sim) in enumerate(similar, 1):
    mol_data = testset[idx]
    print(f"{rank}. Index: {idx} | Similarity: {sim:.4f}")
    print(f"   Target: {all_targets[idx].item():.3f}")
    print(f"   Prediction: {all_predictions[idx].item():.3f}")
    print(f"   SMILES: {mol_data.smiles}")
    print()

# Visualize
similar_indices = [idx for idx, _ in similar]
similarities = [sim for _, sim in similar]

img = visualize_similar_molecules(
    query_idx, 
    similar_indices, 
    testset, 
    similarities,
    save_path=f"similar_molecules_query_{query_idx}.png"
)

# Display in notebook (if using Jupyter)
# from IPython.display import display
# display(img)

# Or save as matplotlib figure
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')
plt.title(f"Query Molecule {query_idx} and Top-5 Similar Molecules", fontsize=14)
plt.tight_layout()
plt.savefig(f"similarity_analysis_query_{query_idx}.png", dpi=150, bbox_inches='tight')
print(f"\nSaved matplotlib figure")

# ============================================================================
# BATCH ANALYSIS: Find clusters of similar molecules
# ============================================================================

def analyze_similarity_distribution(representations, num_samples=100):
    """
    Analyze overall similarity distribution in the dataset
    """
    import random
    
    all_similarities = []
    sample_indices = random.sample(range(len(representations)), 
                                   min(num_samples, len(representations)))
    
    for idx in sample_indices:
        similar = find_similar_molecules(idx, representations, top_k=10)
        all_similarities.extend([sim for _, sim in similar])
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(all_similarities, bins=50, edgecolor='black')
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Top-10 Similarity Scores', fontsize=14)
    plt.axvline(x=0.9, color='r', linestyle='--', label='High similarity (>0.9)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('similarity_distribution.png', dpi=150, bbox_inches='tight')
    print(f"\nSimilarity statistics:")
    print(f"  Mean: {sum(all_similarities)/len(all_similarities):.3f}")
    print(f"  Min:  {min(all_similarities):.3f}")
    print(f"  Max:  {max(all_similarities):.3f}")

print(f"\n{'='*60}")
print("Analyzing overall similarity distribution...")
print(f"{'='*60}\n")
analyze_similarity_distribution(repr_list, num_samples=50)

print(f"\n\nTOTAL TIME: {time.time() - start_time:.0f}s")