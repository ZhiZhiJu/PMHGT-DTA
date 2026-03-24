# import dgl
import numpy as np
import rdkit
import torch
from rdkit import Chem
# from .graph_features import atom_features
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]

def atom_features(atom, explicit_H=False, use_chirality=True):
    """Generate atom features including atom symbol(17),degree(7),formal charge(1),
    radical electrons(1),hybridization(6),aromatic(1),hydrogen atoms attached(5),Chirality(3)
    """
    symbol = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'other']  # 17-dim
    degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2,
                         'other']  # 6-dim
    results = one_of_k_encoding_unk(atom.GetSymbol(), symbol) + \
              one_of_k_encoding(atom.GetDegree(), degree) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [
                  atom.GetIsAromatic()]  # 17+7+2+6+1=33

    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])  # 33+5=38
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 38+3 =41
    return results


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def get_bond_features(bond, use_chirality=True):
    """Generate bond features including bond type(4), conjugated(1), in ring(1), stereo(4)"""
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats).astype(int)


def smile2graph4drugood(smile):
    mol = Chem.MolFromSmiles(smile)
    # if (mol is None):
    #     return None
    src = []
    dst = []
    atom_feature = []
    bond_feature = []

    for atom in mol.GetAtoms():
        one_atom_feature = atom_features(atom)
        atom_feature.append(one_atom_feature)
    atom_feature = np.array(atom_feature)
    atom_feature = torch.tensor(atom_feature).float()

    if len(mol.GetBonds()) > 0:  # mol has bonds
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            one_bond_feature = get_bond_features(bond)
            src.append(i)
            dst.append(j)
            bond_feature.append(one_bond_feature)
            src.append(j)
            dst.append(i)
            bond_feature.append(one_bond_feature)

        src = torch.tensor(src).long()
        dst = torch.tensor(dst).long()
        bond_feature = np.array(bond_feature)
        bond_feature = torch.tensor(bond_feature).float()
        edge_index = torch.vstack([src, dst])
        # graph_cur_smile = dgl.graph((src, dst), num_nodes=len(mol.GetAtoms()))
        # graph_cur_smile.ndata['x'] = atom_feature
        # graph_cur_smile.edata['x'] = bond_feature
    else:
        edge_index = torch.empty((2, 0)).long()
        bond_feature = torch.empty((0, 10)).float()

    return atom_feature, edge_index, bond_feature


def featurize_atoms(mol):
    feats = []
    for atom in mol.GetAtoms():
        feats.append(atom.GetAtomicNum())
    return {'atomic': torch.tensor(feats).reshape(-1).to(torch.int64)}


def featurize_bonds(mol):
    feats = []
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    for bond in mol.GetBonds():
        btype = bond_types.index(bond.GetBondType())
        # One bond between atom u and v corresponds to two edges (u, v) and (v, u)
        feats.extend([btype, btype])
    return {'type': torch.tensor(feats).reshape(-1).to(torch.int64)}
