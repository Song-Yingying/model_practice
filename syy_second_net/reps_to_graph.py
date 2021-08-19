# -*- coding:utf-8 -*-
"""
Author : Junwu Chen
Time   : 2021/02/07
E-mail : 845017597@qq.com
Desc.  :
"""

from rdkit import Chem
# https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html


def one_hot(value, choices):
    """ Build a vector by one-hot encoding

    Args:
        value: the value for which the encoding should be one
        choices: a list of possible values

    Returns:
       list: a vector after one-hot encoding
    """
    return [int(value == choice) for choice in choices]


def get_atom_fv(atom, func_g=None):
    atom_features = {
        'atomic_num': list(range(1, 99)),
        'degree': [0, 1, 2, 3, 4, 5],
        'formal_charge': [-1, -2, 1, 2, 0],
        'chiral_tag': [0, 1, 2, 3],
        'num_Hs': [0, 1, 2, 3, 4],
        'hybridization': [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ]
    }

    # atom_fv: feature vector of an atom
    atom_fv = \
        one_hot(atom.GetAtomicNum(), atom_features['atomic_num']) + \
        one_hot(atom.GetTotalDegree(), atom_features['degree']) + \
        one_hot(atom.GetFormalCharge(), atom_features['formal_charge']) + \
        one_hot(int(atom.GetChiralTag()), atom_features['chiral_tag']) + \
        one_hot(int(atom.GetTotalNumHs()), atom_features['num_Hs']) + \
        one_hot(atom.GetHybridization(), atom_features['hybridization']) + \
        [1 if atom.GetIsAromatic() else 0] + \
        [atom.GetMass() * 0.01]  # scaling 0.01
    if func_g is not None:
        atom_fv += func_g

    return atom_fv


def get_bond_fv(bond):
    bond_features = {
        'bond_type': [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ],
        'stereo': [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS
        ]
    }
    # bond_fv: feature vector of a bond
    bond_fv = \
        one_hot(bond.GetBondType(), bond_features['bond_type']) + \
        [1 if bond.GetIsConjugated() else 0] + \
        [1 if bond.IsInRing() else 0] + \
        one_hot(bond.GetStereo(), bond_features['stereo'])

    return bond_fv


def smi_to_graph(smiles):
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    # Treat H atoms as explicit nodes in the graph
    # mol = Chem.AddHs(mol)
    # atom_fvs: a list of all atom_feature_vectors
    atom_fvs = [get_atom_fv(atom) for atom in mol.GetAtoms()]

    # bond_fvs: a list of all bond_feature_vectors
    row, col, bond_fvs = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        bond_fvs += 2 * [get_bond_fv(bond)]
    bond_index = [row, col]

    return (atom_fvs, bond_index, bond_fvs)
