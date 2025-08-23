from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point2D
from torch_geometric.utils import unbatch
from PIL import Image
import torch
import io


def depict(data, weights, target_only=True):
    data = data.to('cpu').detach()  # Ensure data is on CPU for RDKit
    print(int(data.y.item()), data.smiles)
    
    if (mol := data.mol) is None:
        print("Invalid molecule object")
        return None

    weights = weights.astype(float)
    edge_index = data.edge_index
    
    bond_intensity = {}
    highlight_bonds = []
    highlight_atoms = set()
    bond_colors = {}
    atom_colors = {}

    threshold = 0
    if target_only:
        threshold = weights.mean() # + weights.std()  # Mean + 1 std dev

    # Populating bond_intensity dict, bonds are duplicated, now I keep the max but maybe I should average them
    for i, (src, dst) in enumerate(edge_index.T):
        src_idx, dst_idx = int(src), int(dst)
        bond_idx = mol.GetBondBetweenAtoms(src_idx, dst_idx).GetIdx()
        weight = weights[i]
        if bond_intensity.get(bond_idx) is None:
            bond_intensity[bond_idx] = weight
        else:
            bond_intensity[bond_idx] = max(bond_intensity[bond_idx], weight)  # Keep max weight for bidirectional bonds
    
    atom_norm_intensity = {}
    for bond_idx, weight in bond_intensity.items():
        bond = mol.GetBondWithIdx(bond_idx)
        bond.SetProp("bondNote", str(bond_idx))  # DEBUG: Draw bond index
        if abs(weight) > threshold:  # Only highlight important bonds
            highlight_bonds.append(bond_idx)
            if target_only:
                norm_intensity = weight  # / max(bond_intensity.values())
            else:
                # TODO: pre-norm it in his own function, using 0.5 - baseline
                norm_intensity = abs(weight) / max(abs(weights.min()), abs(weights.max()))  # Normalize to [0, 1]
            if weight > 0:
                bond_colors[bond_idx] = (1.0, 1.0-norm_intensity, 1.0-norm_intensity)
            else:
                bond_colors[bond_idx] = (1.0-norm_intensity, 1.0, 1.0-norm_intensity)
            # Atoms connected by this bond
            for atom in (bond.GetBeginAtom(), bond.GetEndAtom()):
                atom_idx = atom.GetIdx()
                if atom_norm_intensity.get(atom_idx) is None:
                    atom_norm_intensity[atom_idx] = norm_intensity
                    highlight_atoms.add(atom_idx)
                elif norm_intensity > atom_norm_intensity[atom_idx]:
                    atom_norm_intensity[atom_idx] = norm_intensity
                else:
                    continue
                if weight > 0:
                    atom_colors[atom_idx] = (1.0,
                                            1.0 - atom_norm_intensity[atom_idx], 
                                            1.0 - atom_norm_intensity[atom_idx])
                else:
                    atom_colors[atom_idx] = (1.0 - atom_norm_intensity[atom_idx], 
                                            1.0, 
                                            1.0 - atom_norm_intensity[atom_idx])
                # else:
                # #     atom_norm_intensity[atom_idx] = max(atom_norm_intensity[atom_idx], norm_intensity)
                # if weight > 0:
                #     atom_colors[atom_idx] = (1.0,
                #                             1.0 - atom_norm_intensity[atom_idx], 
                #                             1.0 - atom_norm_intensity[atom_idx])
                # else:
                #     atom_colors[atom_idx] = (1.0 - atom_norm_intensity[atom_idx], 
                #                             1.0, 
                #                             1.0 - atom_norm_intensity[atom_idx])
    
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
    if highlight_bonds or highlight_atoms:
        drawer.DrawMolecule(mol,
                           highlightAtoms=list(highlight_atoms),
                           highlightAtomColors=atom_colors,
                           highlightBonds=highlight_bonds,
                           highlightBondColors=bond_colors,
                           legend=legend) 
    else:
        # No highlighting if no important edges found
        drawer.DrawMolecule(mol, legend=legend) 

    drawer.FinishDrawing()
    
    # Get image data and display in popup
    img_data = drawer.GetDrawingText()

    # Convert to PIL Image
    img = Image.open(io.BytesIO(img_data))
    
    # Show in popup window
    img.show()
    
    print(f"Displayed molecule with {len(highlight_bonds)} highlighted bonds and {len(highlight_atoms)} highlighted atoms")
    print(f"Weights range: {weights.min():.3f} - {weights.max():.3f}")
    print(f"Threshold: {threshold:.3f}")
    print(weights)
    print(f"Sum: {weights.sum():.3f}")
    for i, norm_intensity in bond_intensity.items():
        print(f"Bond {i}: {norm_intensity:.3f}")
    input("Press Enter to continue...")

    return img_data

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
    
    print(f"Attribution sum: {attribution_sum:.4f}")
    print(f"Baseline + Attribution sum: {baseline_pred.item() + attribution_sum:.4f}")
    
    print(f"PREDICTION: {final_pred.item():.4f}")
    print("\nDEPICT EDGE IMPORTANCE")
    graph = single_batch.to_data_list()[0]
    depict(graph, edge_importance.detach().cpu().numpy(), target_only=False)