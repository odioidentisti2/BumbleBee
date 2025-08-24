from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point2D
from torch_geometric.utils import unbatch
from PIL import Image
import io


def red_or_green(weight):
    if weight > 0:
        return (1.0, 1.0-weight, 1.0-weight)
    else:
        return (1.0-abs(weight), 1.0, 1.0-abs(weight))

def yellow(weight):
    return (1.0, 1.0, 1.0-abs(weight))

def depict(data, weights, attention=True):
    data = data.to('cpu').detach()  # Ensure data is on CPU for RDKit
    weights = weights.astype(float)
    edge_index = data.edge_index
    threshold = 0
    # if attention:
    #     threshold = weights.mean() * 2  # + weights.std()  # Mean + 1 std dev

    # DEBUG
    print(int(data.y.item()), data.smiles)
    print(f"Weights range: {weights.min():.2f} - {weights.max():.2f}")
    print(weights)
    print(f"Weight sum: {weights.sum():.2f}")
    print(f"Threshold: {threshold:.2f}")

    if (mol := data.mol) is None:
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
        print(f"Bond {bond_idx}: {bond_weights[bond_idx]:.2f}, {directional_weight:.2f} => {weight:.2f}")
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

    target_label = 'Toxic' if data.y.item() == 1 else 'Non-toxic'
    legend = f"{data.smiles}\n{target_label}\n\n{'Attention' if attention else 'Gradients'}"

    # Draw molecule with highlighting
    drawer.DrawMolecule(mol,
                        highlightAtoms=atom_colors.keys(),
                        highlightAtomColors=atom_colors,
                        highlightBonds=bond_colors.keys(),
                        highlightBondColors=bond_colors,
                        legend=legend)
    drawer.FinishDrawing()
    Image.open(io.BytesIO(drawer.GetDrawingText())).show()

    print(f"Sum: {sum(bond_weights.values()):.2f}")