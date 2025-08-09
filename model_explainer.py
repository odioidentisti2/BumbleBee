from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point2D
from torch_geometric.utils import unbatch
from PIL import Image
import torch
import io


def depict(data, attention, edge_index):
    data = data.to('cpu')  # Ensure data is on CPU for RDKit
    print(int(data.y.item()), data.smiles)
    
    if (mol := data.mol) is None:
        print("Invalid molecule object")
        return None
    
    weights = attention.detach().cpu().numpy()
    edge_index = edge_index.detach().cpu().numpy()
    
    bond_intensity = {}
    atom_intensity = {}
    highlight_bonds = []
    highlight_atoms = set()
    bond_colors = {}
    atom_colors = {}
    # bond_labels = {} 
    
    # threshold = 1 / len(weights)  # to be used only after softmax
    threshold = weights.mean() + weights.std()  # Mean + 1 std dev
    
    # Bond intensity
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
        bond.SetProp("bondNote", str(bond_idx))  # Draw bond index
        if weight > threshold:  # Only highlight important bonds
            highlight_bonds.append(bond_idx)
            norm_intensity = float((weight / max(bond_intensity.values())))
            bond_colors[bond_idx] = (1.0, 1.0-norm_intensity, 1.0-norm_intensity)
            # Atoms connected by this bond
            for atom in (bond.GetBeginAtom(), bond.GetEndAtom()):
                atom_idx = atom.GetIdx()
                if atom_norm_intensity.get(atom_idx) is None:
                    atom_norm_intensity[atom_idx] = norm_intensity
                else:
                    atom_norm_intensity[atom_idx] = max(atom_norm_intensity[atom_idx], norm_intensity)
                atom_colors[atom_idx] = (1.0,
                                         1.0 - atom_norm_intensity[atom_idx], 
                                         1.0 - atom_norm_intensity[atom_idx])
                highlight_atoms.add(atom_idx)
    
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
    print(f"Weights range: {attention.min():.3f} - {attention.max():.3f}")
    print(attention)
    for i, norm_intensity in bond_intensity.items():
        print(f"Bond {i}: {norm_intensity:.3f}")
    input("Press Enter to continue...")

    return img_data


def explain(model, batch, steps=5):
    """Integrated gradients explanation for edge features"""
    src, dst = batch.edge_index
    edge_feat = torch.cat([batch.x[src], batch.x[dst], batch.edge_attr], dim=1)
    
    # Baseline: zero features
    baseline = torch.zeros_like(edge_feat)
    
    # Integrated gradients computation
    integrated_grads = torch.zeros_like(edge_feat)

    for i, alpha in enumerate(torch.linspace(0, 1, steps)):
        print(f"Step {i+1}/{steps}: alpha={alpha:.2f}")
        # Interpolate between baseline and input
        interp_feat = baseline + alpha * (edge_feat - baseline)
        interp_feat.requires_grad_(True)
        
        # Forward pass
        prediction = model.graph_forward(interp_feat, batch.edge_index, batch)
        
        # Compute gradients for each graph
        for i in range(prediction.size(0)):
            if prediction[i].requires_grad:
                grad = torch.autograd.grad(
                    outputs=prediction[i],
                    inputs=interp_feat,
                    retain_graph=True,
                    create_graph=False
                )[0]
                integrated_grads += grad
    
    # Average gradients and scale by input difference
    integrated_grads /= steps
    attributions = (edge_feat - baseline) * integrated_grads
    
    print("\nDEPICT EDGE IMPORTANCE")
    edge_batch = model._edge_batch(batch.edge_index, batch.batch)
    edge_importance_list = unbatch(attributions.norm(dim=1), edge_batch)
    i = 0
    for graph, edge_importance in zip(batch.to_data_list(), edge_importance_list):
        print("PREDICTION: ", prediction[i].item())
        depict(graph, edge_importance, graph.edge_index)
        i += 1
