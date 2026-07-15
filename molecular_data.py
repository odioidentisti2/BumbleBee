from random import sample

import torch
from torch_geometric.data import Data, Dataset as PyGDataset
from torch_geometric.loader import  DataLoader
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
import reproducibility
import csv

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # Disable warnings


# Atoms explicitly encoded (+ fallback category)
ELEMENTS = ['H', 'Li', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Si', 'P', 'S', \
            'Cl', 'K', 'Ca', 'Cr', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Se', 'Br', \
            'Sn', 'I']  # Ge, As, Pb, Hg ?
ATOMIC_NUMBERS = [Chem.GetPeriodicTable().GetAtomicNumber(element) for element in ELEMENTS]


def encoding(value, allowed_values):
    """
    Return one-hot encoding of value. 
    Add an extra category for "other" if value is not in allowed_values
    """
    if value not in allowed_values:
        # print(f"WARNING: {value} not in {allowed_values}")  # DEBUG
        # HybridizationType = UNSPECIFIED is the only "other", I should probably encode it
        return [0] * len(allowed_values) + [1]  # last category is "other"
    else:
        return [float(value == v) for v in allowed_values] + [0]
    # In the ESA repo this is implemented weirdly: the "other" category is not extra 
    # but it overwrites the last category, even if that is quite ok for positive choices, 
    # for "formal_charge": [-1, -2, 1, 2, 0] any other charge (like -3 or +3) is mapped 
    # the same way as 0 charge.... ASK the developers!

def atom_features(atom):
    # print(atom.GetHybridization())  # DEBUG
    return (
        # encoding(atom.GetAtomicNum(), list(range(1, 54))) +  # 53 + 1
        encoding(atom.GetAtomicNum(), ATOMIC_NUMBERS) +  # 25 + 1

        # WARNING: TotalDegree and TotalValence can't be 0 because the mol would be rejected (atom with no bonds)
        encoding(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5]) +  # 6 + 1
        # encoding(atom.GetTotalValence(), [1, 2, 3, 4, 5, 6]) +  # 6 + 1

        encoding(atom.GetFormalCharge(), [-2, -1, 0, 1, 2]) +  # 5 + 1
        encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +  # 5 + 1
        encoding(atom.GetChiralTag(), [
            Chem.rdchem.CHI_UNSPECIFIED,
            Chem.rdchem.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.CHI_TETRAHEDRAL_CCW,
            ]) +  # 3 + 1
        encoding(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.UNSPECIFIED,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,  # rare
            Chem.rdchem.HybridizationType.SP3D2,  # rare
            # OTHER: S, SP2D
            ]) +  # 6 + 1
        [float(atom.GetIsAromatic())]  # 1
    )

def bond_features(bond):
    return (
        encoding(bond.GetBondType(), [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ]) +  # 4 + 1
        [float(bond.GetIsConjugated()),  # 1
         float(bond.IsInRing())]  # 1
    )

_mol = Chem.MolFromSmiles("CC")
ATOM_DIM = len(atom_features(_mol.GetAtomWithIdx(0)))
BOND_DIM = len(bond_features(_mol.GetBondWithIdx(0)))

# ESA/data_loading/transforms.py > add_chemprop_features
def smiles2graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Salt removal
    if '.' in smiles:
        mol = SaltRemover().StripMol(mol)
        if len(Chem.GetMolFrags(mol)) > 1:  # Disconnected
            return None
        else:
            print(f"Removed salts from: {smiles}")  # DEBUG

    mol = Chem.AddHs(mol)

    adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
    node_attr = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)
    edge_index = torch.nonzero(torch.from_numpy(adj_matrix)).T  # Symmetric edge index src/dst [2, num_edges]    
    edge_attr = torch.tensor([
        bond_features(mol.GetBondBetweenAtoms(src.item(), dst.item()))
        for src, dst in edge_index.T
    ], dtype=torch.float)
    return Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, mol=mol, smiles=smiles)


class Dataset(PyGDataset):
    def __init__(self, graphs):
        super().__init__()
        self.graphs = graphs
        self.generator = reproducibility.torch_generator()
    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]    
    
    def get_loader(self, batch_size, is_train=False):
        self.generator = reproducibility.torch_generator()
        return DataLoader(self, batch_size=batch_size, shuffle=is_train, drop_last=is_train, generator=self.generator)


class InjectedDataset(Dataset):
    injection_probability = 0.001

    def __init__(self, graphs):
        super().__init__(graphs)
        self.inject = False
        self.inject_generator = reproducibility.torch_generator()
        sample = graphs[0].target
        if isinstance(sample, float):
            self.baseline = sum(g.target for g in graphs) / len(graphs)
        elif isinstance(sample, str):
            unique_y = set(g.y.item() for g in graphs)
            self.baseline = sum(unique_y) / len(unique_y)
        else:
            raise ValueError(f"Unexpected target type: {type(sample)}")
        print(f"DEBUG: baseline = {self.baseline:.2f}")
        
    def get(self, idx):
        data = super().get(idx)
        if self.inject and \
            torch.rand(1, generator=self.inject_generator).item() < InjectedDataset.injection_probability:
            data = data.clone()
            data.x = torch.zeros_like(data.x)
            data.edge_attr = torch.zeros_like(data.edge_attr)
            # Inject baseline target
            data.y = torch.tensor(self.baseline, dtype=torch.float)
        return data
        
    def get_loader(self, batch_size, is_train=False):
        self.inject = is_train
        self.inject_generator = reproducibility.torch_generator()
        return super().get_loader(batch_size, is_train)


def load_from_csv(dataset_info, split=None):
    task = dataset_info['task']
    graphs = []
    with open(dataset_info['path'], 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split and row[dataset_info['split_header']] != split:
                continue
            smiles = row[dataset_info['smiles_header']]
            data = smiles2graph(smiles)
            if not (data
                    and len(data.x.shape) == 2 and data.x.shape[1] == ATOM_DIM
                    and len(data.edge_attr.shape) == 2 and data.edge_attr.shape[1] == BOND_DIM):
                print(f"Invalid SMILES: {smiles}")
                continue
            label = row[dataset_info['target_header']]
            if task == 'binary_classification':
                data.target = label
                data.y = torch.tensor(dataset_info['tox_map'][label], dtype=torch.float)
            elif task == 'regression':
                data.target = float(label)
                data.y = torch.tensor(data.target, dtype=torch.float)
            graphs.append(data)
    if not graphs:
        raise ValueError("No data")
    if task == 'binary_classification':
        classes = set(g.target for g in graphs)
        if len(classes) != 2:
            raise ValueError(f"Multiclass not supported, got {len(classes)} classes: {classes}")  # HANDLE IT!
    return graphs


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
