from rdkit import Chem
import csv
import torch
from torch.utils.data import Dataset

TOX_MAP = {'Mutagenic': 1, 'NON-Mutagenic': 0}
smiles_header = 'SMILES'
label_header = 'Experimental_value'


class MoleculeDataset(Dataset):

    def __init__(self, filepath):
        self.data = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                smiles = row[smiles_header]
                label = row[label_header]
                d = smiles_to_data(smiles, TOX_MAP[label])
                if d is not None:
                    self.data.append(d)
        self.node_dim = self.data[0]['x'].size(-1)
        self.edge_dim = self.data[0]['edge_attr'].size(-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def batch_iter(self, batch_size, shuffle=False):
        indices = list(range(len(self.data)))
        if shuffle:
            import random
            random.shuffle(indices)
        for start_idx in range(0, len(self.data), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch = [self.data[i] for i in batch_indices]
            yield batch


# FEATURE EXTRACTORS

# # alternative version where the extra "other" category is optional
# def encoding(value, choices, include_unknown=True):
    # # Return one-hot encoding of value in choices with an extra category for "other" if value is not in choices
    # encoding = [int(value == s) for s in choices]
    # if include_unknown:
    #     if any(encoding):
    #         encoding += [0]
    #     else:
    #         encoding += [1]
    # return encoding

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

def smiles_to_data(smiles, target):
    mol = Chem.MolFromSmiles(smiles)  # sanitize=True)
    if mol is None:
        return None
    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()])
    edge_index = torch.nonzero(torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol))).T
    edge_attr = torch.stack(
        [bond_features(mol.GetBondBetweenAtoms(edge_index[0][i].item(), edge_index[1][i].item()))
        for i in range(edge_index.shape[1])]
    )
    # edge_attr = []
    # src, dst = [], []
    # for bond in mol.GetBonds():
    #     i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    #     feat = bond_features(bond)
    #     src += [i, j]
    #     dst += [j, i]
    #     edge_attr += [feat, feat]
    # edge_attr = torch.stack(edge_attr)
    # edge_index = torch.tensor([src, dst], dtype=torch.long)
    y = torch.tensor([target], dtype=torch.float)
    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "y": y}