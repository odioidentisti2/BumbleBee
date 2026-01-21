from rdkit.Chem.Draw import rdMolDraw2D

import utils

# Check if running in Google Colab
# I think if I use matplotlib I won't need this
IN_COLAB = False
try:
    import google.colab # Try importing, if it succeeds, it's Colab
    IN_COLAB = True
except ImportError:
    pass # Not in Colab

if IN_COLAB:
    from IPython.display import Image, display
else:
    from PIL import Image
    import io


def red_or_green(weight):
    if weight > 0:
        return (1.0, 1.0-weight, 1.0-weight)
    else:
        return (1.0-abs(weight), 1.0, 1.0-abs(weight))

def yellow(weight):
    if weight < 0: weight = 0
    return (1.0, 1.0, 1.0-weight)

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

def print_weights(weights, average=False, title="WEIGHTS:"):
    print(f"\n{title}")
    print(weights)
    print(f"Weights range: {weights.min():.4f} - {weights.max():.4f}")
    if average: print("Weights Average: ", f"{weights.mean().item():.4f}")
    print(f"Weight sum: {weights.sum():.2f}")
    print()

def att_scores(weights):
    # Weights come after softmax (they add up to 1): 
    # => weights.mean() = 1 / len(weights)
    # Therefore:
    #   weight * len(weights) == 1  means "average attention"
    scores = weights * weights.shape[0]  # visualize the proportion to average attention
    return scores

def depict_tokens(graph, weights, attention=False, factor=None, shift=None):

    if attention:
        weights = att_scores(weights)
    weights = weights.cpu().numpy().astype(float)

    threshold = 0  # not needed anymore? (normalization in explainer)
    # if attention:
    #     threshold = weights.mean() * 2  # + weights.std()  # Mean + 1 std dev
    # print(f"\nThreshold: {threshold:.2f}")

    if (mol := graph.mol) is None:
        print("Invalid molecule object")
        return
    
    bond_weights = {}
    atom_weights = {}
    bond_colors = {}
    atom_colors = {}

    # Populating bond_intensity dict
    for directional_weight, (src, dst) in zip(weights, graph.edge_index.T):
        bond = mol.GetBondBetweenAtoms(int(src), int(dst))
        bond_idx = bond.GetIdx()
        if bond_idx not in bond_weights:
            bond_weights[bond_idx] = directional_weight
            continue  # bonds are duplicated (directional), apply the logic at the 2nd pass
        weight = (bond_weights[bond_idx] + directional_weight)  # SUM weight for bidirectional bonds
        if factor is None and shift is None:
            print(f"Bond {bond_idx}: {weight:.2f} = ({bond_weights[bond_idx]:.4f} + {directional_weight:.4f})")
        else:
            original_weight = weight
            if shift is not None:
                weight = weight + shift * 2  # sum of 2 weights (clipping in yellow() for attention)
            if factor is not None:
                weight = weight * factor
            print(f"Bond {bond_idx}: {weight:.2f} <- {original_weight:.2f} = ({bond_weights[bond_idx]:.4f} + {directional_weight:.4f})")            
        bond_weights[bond_idx] = weight
        bond.SetProp("bondNote", str(bond_idx))  # DEBUG: Draw bond index

        if abs(weight) > abs(threshold):  # Only highlight important bonds 
            bond_colors[bond_idx] = yellow(weight) if attention else red_or_green(weight)

            # Atoms connected by this bond
            for atom in (bond.GetBeginAtom(), bond.GetEndAtom()):
                atom_idx = atom.GetIdx()
                if atom_idx not in atom_weights:
                    atom_weights[atom_idx] = weight
                elif abs(weight) > abs(atom_weights[atom_idx]):
                    atom_weights[atom_idx] = weight
                else:
                    continue
                atom_colors[atom_idx] = yellow(weight) if attention else red_or_green(weight)
    
    print(f"Sum: {sum(bond_weights.values()):.2f}")

    draw(graph, atom_colors, bond_colors)


def depict_atom_bond(graph, atom_importance, bond_importance, positive_only=False, factor=None, shift=None):
    graph = graph.detach()

    threshold = 0  # not needed anymore? (normalization in explainer)
    mol = graph.mol

    bond_colors = {}
    atom_colors = {}

    # --- Aggregate edge attributions to bond instances ---
    import numpy as np
    # Expect atom_importance: 1D array length == num_atoms
    #        bond_importance: 1D array length == num_bonds
    bond_arr = np.asarray(bond_importance, dtype=float)
    atom_arr = np.asarray(atom_importance, dtype=float)
    num_bonds = mol.GetNumBonds()
    num_atoms = mol.GetNumAtoms()
    if bond_arr.ndim != 1 or bond_arr.size != num_bonds:
        raise ValueError(f"bond_importance must be 1D of length {num_bonds}, got {bond_arr.shape}")
    if atom_arr.ndim != 1 or atom_arr.size != num_atoms:
        raise ValueError(f"atom_importance must be 1D of length {num_atoms}, got {atom_arr.shape}")
    bond_instance_importance = bond_arr.tolist()
    atom_instance_importance = atom_arr.tolist()

    print(f"Sum atom_instance_importance: {sum(atom_instance_importance):.2f}")
    print(f"Sum bond_instance_importance: {sum(bond_instance_importance):.2f}")

    # Color bonds
    print("\nBonds:")
    for bond in mol.GetBonds():
        bond_idx = bond.GetIdx()
        weight = bond_instance_importance[bond_idx]
        original_weight = weight
        if factor is not None:
            weight = weight * factor
        if shift is not None:
            weight = weight + shift
        if abs(weight) > abs(threshold):
            bond_colors[bond_idx] = yellow(weight) if positive_only else red_or_green(weight)
        bond.SetProp("bondNote", str(bond_idx))  # DEBUG: Draw bond index
        # Print bond details like in depict()
        if factor is None and shift is None:
            print(f"Bond {bond_idx}: {weight:.2f}")
        else:
            print(f"Bond {bond_idx}: {weight:.2f} <- {original_weight:.2f}")

    # Color atoms
    print("\nAtoms:")
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        weight = atom_instance_importance[atom_idx]
        original_weight = weight
        if factor is not None:
            weight = weight * factor
        if shift is not None:
            weight = weight + shift
        if abs(weight) > abs(threshold):
            atom_colors[atom_idx] = yellow(weight) if positive_only else red_or_green(weight)
        if factor is None and shift is None:
            print(f"Atom {atom_idx}: {weight:.2f}")
        else:
            print(f"Atom {atom_idx}: {weight:.2f} <- {original_weight:.2f}")

    draw(graph, atom_colors, bond_colors, atom_indices=True)


def draw(graph, atom_colors, bond_colors, atom_indices=False, bond_indices=True):  
    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    
    # Set drawing options
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = bond_indices
    opts.addAtomIndices = atom_indices
    opts.multipleBondOffset = 0.18
    opts.annotationFontScale = 0.8  # Adjust font size for bond labels
    opts.prepareMolsBeforeDrawing = False  # Important for custom drawing

    target_label = graph.label if hasattr(graph, 'label') else f"{graph.y.item():.2f}"
    legend = f"{graph.smiles}\n{target_label}"  # \nPrediction: {getattr(graph, 'prediction', float('nan')):.2f}"

    # Draw molecule with highlighting
    drawer.DrawMolecule(graph.mol,
                        highlightAtoms=atom_colors.keys(),
                        highlightAtomColors=atom_colors,
                        highlightBonds=bond_colors.keys(),
                        highlightBondColors=bond_colors,
                        legend=legend)
    drawer.FinishDrawing()

    # Check if running in Google Colab
    if IN_COLAB:
        image_bytes = drawer.GetDrawingText()
        display(Image(data=image_bytes))
    else:
        Image.open(io.BytesIO(drawer.GetDrawingText())).show()

