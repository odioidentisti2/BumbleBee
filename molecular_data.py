import torch
from torch_geometric.data import Data, Dataset
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
import csv

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # Disable warnings


# These param are hardcoded
ATOM_DIM = 57
BOND_DIM = 7
ELEMENTS = ['H', 'Li', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Cr', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Se', 'Br', 'Sn', 'I']
ATOMIC_NUMBERS = [Chem.GetPeriodicTable().GetAtomicNumber(element) for element in ELEMENTS]
# Ge, As, Pb, Hg ?


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
        # TotalDegree and TotalValence can't be 0 because the mol would be rejected (atom with no bonds)
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
            print(f"Removed salts from: {smiles}")    

    mol = Chem.AddHs(mol)

    adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
    node_attr = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)
    edge_index = torch.nonzero(torch.from_numpy(adj_matrix)).T  # Symmetric edge index src/dst [2, num_edges]    
    edge_attr = torch.tensor([
        bond_features(mol.GetBondBetweenAtoms(src.item(), dst.item()))
        for src, dst in edge_index.T
    ], dtype=torch.float)
    return Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, mol=mol, smiles=smiles)

class GraphDataset(Dataset):
    def __init__(self, dataset_info, split=None):
        print(f"\n{split} set: {dataset_info['path']}")
        super().__init__()
        self.task = dataset_info['task']
        self.graphs = []
        self.targets = []
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
                target = row[dataset_info['target_header']]
                if self.task == 'binary_classification':
                    data.label = target
                    target = dataset_info['tox_map'][target]
                elif self.task == 'regression':
                    target = float(target)                
                data.y = torch.tensor([target], dtype=torch.float)
                self.graphs.append(data)
                self.targets.append(target)
        print(f"Loaded {len(self.graphs)} molecules")
        
        if not self.graphs:
            raise ValueError(f"No data")

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
