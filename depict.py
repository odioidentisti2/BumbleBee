from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import torch
import io


def depict(mol, graph_attention, graph_edge_index):
    """
    Visualize molecule with attention-based highlighting in a popup window
    
    Args:
        mol: RDKit molecule object
        graph_attention: tensor of attention weights per edge [num_edges]
        graph_edge_index: edge connectivity [2, num_edges]
    """

    
    if mol is None:
        print("Invalid molecule object")
        return None
    
    # Convert attention weights to numpy
    if torch.is_tensor(graph_attention):
        weights = graph_attention.detach().cpu().numpy()
    else:
        weights = graph_attention
    
    # Convert edge_index to numpy if needed
    if torch.is_tensor(graph_edge_index):
        edge_index = graph_edge_index.detach().cpu().numpy()
    else:
        edge_index = graph_edge_index
    
    # Get bond indices and atoms for highlighting
    highlight_bonds = []
    highlight_atoms = set()
    bond_colors = {}
    atom_colors = {}
    
    # Set threshold for highlighting (top 50% of attention weights)
    # threshold = weights.mean() if len(weights) > 0 else 0
    max_weight = weights.max()
    threshold = 1 / len(weights)
    
    # Map edge_index to RDKit bond indices
    for i, (src, dst) in enumerate(edge_index.T):
        src_idx, dst_idx = int(src), int(dst)
        
        # Find corresponding bond in RDKit molecule
        bond = mol.GetBondBetweenAtoms(src_idx, dst_idx)
        if bond is not None:
            bond_idx = bond.GetIdx()
            weight = weights[i]
            
            if weight > threshold:  # Only highlight important edges
                highlight_bonds.append(bond_idx)
                highlight_atoms.add(src_idx)
                highlight_atoms.add(dst_idx)
                
                # Color intensity based on attention (red scale)
                # intensity = float(min(weight / max_weight, 1.0))
                intensity = float(((weight / max_weight) / 2) + 0.5)
                bond_colors[bond_idx] = (1.0, 1.0-intensity, 1.0-intensity)
                
                # Also color the connected atoms
                atom_colors[src_idx] = (1.0, 1.0-intensity*0.5, 1.0-intensity*0.5)
                atom_colors[dst_idx] = (1.0, 1.0-intensity*0.5, 1.0-intensity*0.5)
                print(bond_idx, intensity)
    
    # Create drawer
    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    
    # Set drawing options
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = True
    opts.addAtomIndices = False  # Set to True if you want to see atom indices
    
    # Draw molecule with highlighting
    if highlight_bonds or highlight_atoms:
        drawer.DrawMolecule(mol, 
                           highlightAtoms=list(highlight_atoms),
                           highlightAtomColors=atom_colors,
                           highlightBonds=highlight_bonds,
                           highlightBondColors=bond_colors)
    else:
        # No highlighting if no important edges found
        drawer.DrawMolecule(mol)
    
    drawer.FinishDrawing()
    
    # Get image data and display in popup
    img_data = drawer.GetDrawingText()
    
    # Convert to PIL Image
    img = Image.open(io.BytesIO(img_data))
    
    # Show in popup window
    img.show()
    
    print(f"Displayed molecule with {len(highlight_bonds)} highlighted bonds and {len(highlight_atoms)} highlighted atoms")
    # print(f"Attention range: {weights.min():.3f} - {weights.max():.3f}")
    
    return img_data