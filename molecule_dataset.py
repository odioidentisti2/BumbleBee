import torch
from torch_geometric.data import Data, Dataset
from rdkit import Chem
from rdkit.Chem import rdmolops
import numpy as np
import csv

TOX_MAP = {'Mutagenic': 1, 'NON-Mutagenic': 0}
smiles_header = 'SMILES'
label_header = 'Experimental_value'


# FEATURE EXTRACTORS

def encoding(value, allowed_values):
    # Return one-hot encoding of value with an extra category for "other" if value is not in allowed_values
    if value not in allowed_values:
        return [0] * len(allowed_values) + [1]  # last category is "other"
    else:
        return [float(value == v) for v in allowed_values] + [0]
# in the ESA repo this is implemented weirdly: the "other" category is not extra but it overwrites the last category,
# even if that is quite ok for positive choices only, for "formal_charge": [-1, -2, 1, 2, 0]
#  any other charge (like -3 or +3) is mapped the same way as 0 charge.... ASK the developers!


def atom_features(atom):
    return torch.tensor(
        encoding(atom.GetAtomicNum(), list(range(1, 54))) + 
        encoding(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5]) +
        encoding(atom.GetFormalCharge(), [-2, -1, 0, 1, 2]) +
        encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
        encoding(atom.GetChiralTag(), [         
            Chem.rdchem.CHI_UNSPECIFIED,
            Chem.rdchem.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.CHI_TETRAHEDRAL_CCW,
            # Chem.rdchem.CHI_OTHER,  # "other" is already added in the encoding function
            ]) +
        encoding(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            ]) +
        [float(atom.GetIsAromatic())],
        dtype=torch.float
    )

def bond_features(bond):
    return torch.tensor(
        encoding(bond.GetBondType(), [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ]) +
        [float(bond.GetIsConjugated()),
         float(bond.IsInRing())],
        dtype=torch.float
    )

def smiles2graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node
    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()])

    # Edges   
    edge_index = torch.nonzero(torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol))).T
    edge_attr = torch.stack(
        [bond_features(mol.GetBondBetweenAtoms(edge_index[0][i].item(), edge_index[1][i].item()))
        for i in range(edge_index.shape[1])]
    )

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class MoleculeDataset(Dataset):
    def __init__(self, csv_path):
        super().__init__()
        self.graphs = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                smiles = row[smiles_header]
                label = row[label_header]
                data = smiles2graph(smiles)
                if data is not None:
                    data.y = torch.tensor([TOX_MAP[label]], dtype=torch.float)
                    self.graphs.append(data)
        self.node_dim = self.graphs[0].x.size(1)
        self.edge_dim = self.graphs[0].edge_attr.size(1)

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
