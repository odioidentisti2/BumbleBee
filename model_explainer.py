from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point2D
from torch_geometric.utils import unbatch
from PIL import Image
import torch
import io


def depict(data, attention, edge_index):
    """
    Visualize molecule with attention-based highlighting in a popup window
    
    Args:
        mol: RDKit molecule object
        graph_attention: tensor of attention weights per edge [num_edges]
        graph_edge_index: edge connectivity [2, num_edges]
    """
    data = data.to('cpu')  # Ensure data is on CPU for RDKit
    print(int(data.y.item()), data.smiles)
    if (mol := data.mol) is None:
        print("Invalid molecule object")
        return None
    
    # Convert attention weights to numpy
    if torch.is_tensor(attention):
        weights = attention.detach().cpu().numpy()
    else:
        weights = attention
    
    # Convert edge_index to numpy if needed
    if torch.is_tensor(edge_index):
        edge_index = edge_index.detach().cpu().numpy()
    else:
        edge_index = edge_index
    
    # Get bond indices and atoms for highlighting
    highlight_bonds = []
    highlight_atoms = set()
    bond_colors = {}
    atom_colors = {}
    # bond_labels = {} 
    
    # threshold = 1 / len(weights)  # to be used only after softmax
    threshold = weights.mean() + weights.std()  # Mean + 1 std dev
    
    # Map edge_index to RDKit bond indices
    for i, (src, dst) in enumerate(edge_index.T):
        src_idx, dst_idx = int(src), int(dst)
        
        # Find corresponding bond in RDKit molecule
        bond = mol.GetBondBetweenAtoms(src_idx, dst_idx)
        if bond is not None:
            bond_idx = bond.GetIdx()
            weight = weights[i]
            # bond_labels[bond_idx] = f"{weight:.3f}" 
            
            if weight > threshold:  # Only highlight important edges
                highlight_bonds.append(bond_idx)
                highlight_atoms.add(src_idx)
                highlight_atoms.add(dst_idx)
                
                # Color intensity based on attention (red scale)
                # intensity = float(min(weight / weights.max(), 1.0))
                intensity = float(((weight / weights.max()) / 2) + 0.5)
                bond_colors[bond_idx] = (1.0, 1.0-intensity, 1.0-intensity)
                atom_colors[src_idx] = (1.0, 1.0-intensity*0.5, 1.0-intensity*0.5)
                atom_colors[dst_idx] = (1.0, 1.0-intensity*0.5, 1.0-intensity*0.5)
                # print(bond_idx, intensity)
    
    # Create drawer
    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    
    # Set drawing options
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = True
    opts.addAtomIndices = False  # Set to True if you want to see atom indices
    opts.multipleBondOffset = 0.18
    opts.annotationFontScale = 0.8  # Adjust font size for bond labels
    opts.prepareMolsBeforeDrawing = False  # Important for custom drawing

    for bond in mol.GetBonds():
        bond.SetProp("bond_idx", str(bond.GetIdx()))  # Set bond annotation

    # Draw molecule with highlighting
    if highlight_bonds or highlight_atoms:
        drawer.DrawMolecule(mol, 
                           highlightAtoms=list(highlight_atoms),
                           highlightAtomColors=atom_colors,
                           highlightBonds=highlight_bonds,
                           highlightBondColors=bond_colors)
    else:
        # No highlighting if no important edges found
        drawer.DrawMolecule(mol, data.smiles)
    drawer.DrawString(data.smiles, Point2D(10, 480))  # Position at bottom-left
    
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
