import torch
from torch_geometric.data import Data, Dataset
from rdkit import Chem
import numpy as np
import csv

smiles_header = 'SMILES'
label_header = 'Experimental_value'
split_header = 'Status'
TOX_MAP = {'Mutagenic': 1, 'NON-Mutagenic': 0}


def encoding(value, allowed_values):
    """
    Return one-hot encoding of value. 
    Add an extra category for "other" if value is not in allowed_values
    """
    if value not in allowed_values:
        return [0] * len(allowed_values) + [1]  # last category is "other"
    else:
        return [float(value == v) for v in allowed_values] + [0]
    # In the ESA repo this is implemented weirdly: the "other" category is not extra 
    # but it overwrites the last category, even if that is quite ok for positive choices, 
    # for "formal_charge": [-1, -2, 1, 2, 0] any other charge (like -3 or +3) is mapped 
    # the same way as 0 charge.... ASK the developers!

def atom_features(atom):
    return (
        encoding(atom.GetAtomicNum(), list(range(1, 54))) + 
        encoding(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5]) +
        encoding(atom.GetFormalCharge(), [-2, -1, 0, 1, 2]) +
        encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
        encoding(atom.GetChiralTag(), [    
            Chem.rdchem.CHI_UNSPECIFIED,
            Chem.rdchem.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.CHI_TETRAHEDRAL_CCW,
            # Chem.rdchem.CHI_OTHER,  # "other" is already added by the encoding function
            ]) +
        encoding(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            ]) +
        [float(atom.GetIsAromatic())]
    )

def bond_features(bond):
    return (
        encoding(bond.GetBondType(), [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ]) +
        [float(bond.GetIsConjugated()),
         float(bond.IsInRing())]
    )

# ESA/data_loading/transforms.py > add_chemprop_features
def smiles2graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
    ei = torch.nonzero(torch.from_numpy(adj_matrix)).T  # Symmetric edge index src/dst [2, num_edges]
    node_feat = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)
    edge_feat = torch.tensor([
        bond_features(mol.GetBondBetweenAtoms(ei[0][i].item(), ei[1][i].item()))
        for i in range(ei.shape[1])
    ], dtype=torch.float)  # Edges
    return Data(x=node_feat, edge_index=ei, edge_attr=edge_feat)

class GraphDataset(Dataset):
    def __init__(self, csv_path, split=None):
        super().__init__()
        self.graphs = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split is not None and row[split_header] != split:
                    continue                    
                smiles = row[smiles_header]
                label = row[label_header]
                data = smiles2graph(smiles)
                if data is not None:
                    target = TOX_MAP[label]
                    data.y = torch.tensor([target], dtype=torch.float)
                    self.graphs.append(data)
        
        if self.graphs:  # Check if we have any graphs
            self.node_dim = self.graphs[0].x.size(1)
            self.edge_dim = self.graphs[0].edge_attr.size(1)
        else:
            raise ValueError(f"No graphs found for split: {split}")

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]




# def encoding(value, choices, include_unknown=True):
    # # Return one-hot encoding of value in choices with an extra category for "other" if value is not in choices
    # # alternative version where the extra "other" category is optional
    # encoding = [int(value == s) for s in choices]
    # if include_unknown:
    #     if any(encoding):
    #         encoding += [0]
    #     else:
    #         encoding += [1]
    # return encoding
