from rdkit.Chem.Draw import rdMolDraw2D


def weight_summary(weights, average=False, title="WEIGHTS:"):
    lines = [
        f"{title}",
        str(weights),
        f"Weights range: {weights.min():.4f} - {weights.max():.4f}",
    ]
    if average:
        lines.append(f"Weights Average: {weights.mean().item():.4f}")
    lines.extend([f"Weight sum: {weights.sum():.2f}"])
    return '\n' + '\n'.join(lines) + '\n'

def bond_summary(weights, transformed, title="BOND WEIGHTS:"):
    """Per-bond raw vs. scaled weights, plus the sum (pairs with weight_summary)."""
    raw, scaled = weights.tolist(), transformed.tolist()
    lines = [title]
    lines.extend(f"Bond {i}: {t:.2f} <- {w:.2f}" for i, (w, t) in enumerate(zip(raw, scaled)))
    lines.append(f"Sum: {sum(scaled):.2f}")
    return '\n' + '\n'.join(lines) + '\n'

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

def depict(graph, bond_weights, attention=False):
    """Render `graph`. `bond_weights` must already be shifted/scaled by the explainer."""

    threshold = 0  # not needed anymore? (normalization in explainer)

    atom_weights = {}
    bond_colors = {}
    atom_colors = {}

    for bond_idx, weight in enumerate(bond_weights):
        weight = weight.item()
        bond = graph.mol.GetBondWithIdx(bond_idx)

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

    return draw(graph, atom_colors, bond_colors)

def draw(graph, atom_colors, bond_colors, atom_indices=False, bond_indices=True):  
    drawer = rdMolDraw2D.MolDraw2DCairo(640, 640)
    
    # Set drawing options
    opts = drawer.drawOptions()
    opts.clearBackground = False  # transparent background
    opts.addStereoAnnotation = bond_indices
    opts.addAtomIndices = atom_indices
    opts.multipleBondOffset = 0.18
    opts.annotationFontScale = 0.8  # Adjust font size for bond labels
    opts.prepareMolsBeforeDrawing = False  # Important for custom drawing

    target_label = graph.label if hasattr(graph, 'label') else f"{graph.y.item():.2f}"
    legend = f"{graph.smiles}\n{target_label}"  #\nPrediction: {getattr(graph, 'prediction', float('nan')):.2f}"

    # Draw molecule with highlighting
    drawer.DrawMolecule(graph.mol,
                        highlightAtoms=atom_colors.keys(),
                        highlightAtomColors=atom_colors,
                        highlightBonds=bond_colors.keys(),
                        highlightBondColors=bond_colors,
                        legend=legend)
    drawer.FinishDrawing()


    return drawer.GetDrawingText()



# def depict_atom_bond(graph, atom_importance, bond_importance, positive_only=False, factor=None, shift=None):
#     graph = graph.detach()

#     threshold = 0  # not needed anymore? (normalization in explainer)
#     mol = graph.mol

#     bond_colors = {}
#     atom_colors = {}

#     # --- Aggregate edge attributions to bond instances ---
#     import numpy as np
#     # Expect atom_importance: 1D array length == num_atoms
#     #        bond_importance: 1D array length == num_bonds
#     bond_arr = np.asarray(bond_importance, dtype=float)
#     atom_arr = np.asarray(atom_importance, dtype=float)
#     num_bonds = mol.GetNumBonds()
#     num_atoms = mol.GetNumAtoms()
#     if bond_arr.ndim != 1 or bond_arr.size != num_bonds:
#         raise ValueError(f"bond_importance must be 1D of length {num_bonds}, got {bond_arr.shape}")
#     if atom_arr.ndim != 1 or atom_arr.size != num_atoms:
#         raise ValueError(f"atom_importance must be 1D of length {num_atoms}, got {atom_arr.shape}")
#     bond_instance_importance = bond_arr.tolist()
#     atom_instance_importance = atom_arr.tolist()

#     logger.debug(f"Sum atom_instance_importance: {sum(atom_instance_importance):.2f}")
#     logger.debug(f"Sum bond_instance_importance: {sum(bond_instance_importance):.2f}")

#     # Color bonds
#     logger.debug("\nBonds:")
#     for bond in mol.GetBonds():
#         bond_idx = bond.GetIdx()
#         weight = bond_instance_importance[bond_idx]
#         original_weight = weight
#         if factor is not None:
#             weight = weight * factor
#         if shift is not None:
#             weight = weight + shift
#         if abs(weight) > abs(threshold):
#             bond_colors[bond_idx] = yellow(weight) if positive_only else red_or_green(weight)
#         bond.SetProp("bondNote", str(bond_idx))  # DEBUG: Draw bond index

#         # Log bond details like in depict()
#         if factor is None and shift is None:
#             logger.debug(f"Bond {bond_idx}: {weight:.2f}")
#         else:
#             logger.debug(f"Bond {bond_idx}: {weight:.2f} <- {original_weight:.2f}")

#     # Color atoms
#     logger.debug("\nAtoms:")
#     for atom in mol.GetAtoms():
#         atom_idx = atom.GetIdx()
#         weight = atom_instance_importance[atom_idx]
#         original_weight = weight
#         if factor is not None:
#             weight = weight * factor
#         if shift is not None:
#             weight = weight + shift
#         if abs(weight) > abs(threshold):
#             atom_colors[atom_idx] = yellow(weight) if positive_only else red_or_green(weight)

#         if factor is None and shift is None:
#             logger.debug(f"Atom {atom_idx}: {weight:.2f}")
#         else:
#             logger.debug(f"Atom {atom_idx}: {weight:.2f} <- {original_weight:.2f}")

#     draw(graph, atom_colors, bond_colors, atom_indices=True)