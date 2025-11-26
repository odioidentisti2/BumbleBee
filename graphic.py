import torch
from rdkit.Chem.Draw import rdMolDraw2D
# from rdkit.Geometry import Point2D

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
    return (1.0, 1.0, 1.0-abs(weight))

def sum_bond_weights(edge_index, weights):
    # edge_index: shape [2, num_edges]
    # weights: shape [num_edges]
    src = edge_index[0]
    dst = edge_index[1]
    # Make undirected: always (min, max)
    bond_pairs = torch.stack([torch.minimum(src, dst), torch.maximum(src, dst)], dim=1)  # [num_edges, 2]
    # Find unique bonds and sum weights
    bond_keys, inverse_indices = torch.unique(bond_pairs, dim=0, return_inverse=True)
    summed_weights = torch.zeros(len(bond_keys), dtype=weights.dtype, device=weights.device)
    summed_weights.scatter_add_(0, inverse_indices, weights)
    return bond_keys, summed_weights


def depict(graph, weights, attention=True):
    graph = graph.to('cpu').detach()  # Ensure data is on CPU for RDKit
    weights = weights.astype(float)
    edge_index = graph.edge_index
    threshold = 0
    # if attention:
    #     threshold = weights.mean() * 2  # + weights.std()  # Mean + 1 std dev

    print(f"\nThreshold: {threshold:.2f}")

    if (mol := graph.mol) is None:
        print("Invalid molecule object")
        return
    
    bond_weights = {}
    atom_weights = {}
    bond_colors = {}
    atom_colors = {}

    # Populating bond_intensity dict
    for directional_weight, (src, dst) in zip(weights, edge_index.T):
        bond = mol.GetBondBetweenAtoms(int(src), int(dst))
        bond_idx = bond.GetIdx()
        if bond_idx not in bond_weights:
            bond_weights[bond_idx] = directional_weight
            continue  # bonds are duplicated (directional), apply the logic at the 2nd pass
        weight = (bond_weights[bond_idx] + directional_weight)  # SUM weight for bidirectional bonds
        print(f"Bond {bond_idx}: {weight:.2f} \t({bond_weights[bond_idx]:.4f} + {directional_weight:.4f})")
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
    
    # Create drawer
    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    
    # Set drawing options
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = True
    opts.addAtomIndices = False  # Set to True if you want to see atom indices
    opts.multipleBondOffset = 0.18
    opts.annotationFontScale = 0.8  # Adjust font size for bond labels
    opts.prepareMolsBeforeDrawing = False  # Important for custom drawing

    target_label = 'Toxic' if graph.y.item() == 1 else 'Non-toxic'
    legend = f"{graph.smiles}\n{target_label}\n\n{'Attention' if attention else 'Gradients'}"

    # Draw molecule with highlighting
    drawer.DrawMolecule(mol,
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

    print(f"Sum: {sum(bond_weights.values()):.2f}")
