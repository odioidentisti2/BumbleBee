from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point2D
from torch_geometric.utils import unbatch
from PIL import Image
import torch
import io


def red_or_green(weight):
    if weight > 0:
        return (1.0, 1.0-weight, 1.0-weight)
    else:
        return (1.0-abs(weight), 1.0, 1.0-abs(weight))

def depict(data, weights, attention=True):
    data = data.to('cpu').detach()  # Ensure data is on CPU for RDKit
    weights = weights.astype(float)
    edge_index = data.edge_index
    threshold = 0
    if attention:
        threshold = weights.mean() # + weights.std()  # Mean + 1 std dev

    # DEBUG
    print("\nDEPICT EDGE IMPORTANCE")
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
        if attention:  # temp... I should proably divide by 2 (in the normalization) and sum them here
            weight = max(bond_weights[bond_idx], directional_weight)  # MAX weight for bidirectional bonds
        else:
            weight = (bond_weights[bond_idx] + directional_weight)  # SUM weight for bidirectional bonds
        print(f"Bond {bond_idx}: {bond_weights[bond_idx]:.2f}, {directional_weight:.2f} => {weight:.2f}")
        bond_weights[bond_idx] = weight
        bond.SetProp("bondNote", str(bond_idx))  # DEBUG: Draw bond index

        if abs(weight) > abs(threshold):  # Only highlight important bonds 
            bond_colors[bond_idx] = red_or_green(weight)

            # Atoms connected by this bond
            for atom in (bond.GetBeginAtom(), bond.GetEndAtom()):
                atom_idx = atom.GetIdx()
                if atom_idx not in atom_weights:
                    atom_weights[atom_idx] = weight
                elif abs(weight) > abs(atom_weights[atom_idx]):
                    atom_weights[atom_idx] = weight
                else:
                    continue
                atom_colors[atom_idx] = red_or_green(weight)
    
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
    legend = f"{data.smiles}\n{target_label}"

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
    input("Press Enter to continue...")


def explain_with_attention(batch, attn_weights):
    for data, attention in zip(batch, attn_weights):
        print("\nDEPICT ATTENTION")
        depict(data, attention)        

def explain_with_gradients(model, single_batch, steps=5):
    """Integrated gradients explanation for edge features"""
    edge_feat = model.get_features(single_batch)   
    baseline = torch.zeros_like(edge_feat)    
    integrated_grads = torch.zeros_like(edge_feat)

    # Get baseline and final predictions for verification
    with torch.no_grad():
        baseline_pred = model.single_forward(baseline, single_batch.edge_index, single_batch)
        final_pred = model.single_forward(edge_feat, single_batch.edge_index, single_batch)
    print(f"\nBaseline prediction: {baseline_pred.item():.4f}")
    # print(f"Expected attribution sum: {(final_pred - baseline_pred).item():.4f}")


    for i, alpha in enumerate(torch.linspace(0, 1, steps)):
        # print(f"  Step {i+1}/{steps}: alpha={alpha:.2f}")
        # Interpolate between baseline and input
        interp_feat = baseline + alpha * (edge_feat - baseline)
        interp_feat.requires_grad_(True)
        
        # Forward pass
        prediction = model.single_forward(interp_feat, single_batch.edge_index, single_batch)

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
    weights = edge_importance.detach().cpu().numpy()
    # Before depict I should normalize edge_importance by 0.5 - baseline
    depict(graph, weights, attention=False)