import os
import json
import torch
import argparse
from collections import OrderedDict
from rdkit import Chem
from torch_geometric.data import Data
from unimol_tools import UniMolRepr  # Requires `pip install unimol_tools`
import pandas as pd

torch.cuda.is_available = lambda: False
# Set environment variable for Hugging Face mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Define a function to extract edge features from RDKit bond objects
def edge_features(bond):
    """Extracts features for a bond, including bond type and properties."""
    bond_type = bond.GetBondType()
    return torch.tensor([
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]).long()

# Function to convert a molecule object into a graph format
def mol_to_graph(mol_graph, features):
    """Converts RDKit molecule object to a graph with edge and node features."""
    # Build edge list and edge features for each bond in the molecule
    edge_list = torch.LongTensor([
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) 
        for b in mol_graph.GetBonds()
    ])
    # Split the edge list into indices and features, handling cases with no edges
    edge_list, edge_feats = (
        (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list)
        else (torch.LongTensor([]), torch.FloatTensor([]))
    )
    # Add reverse edges for undirected graph
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    edge_feats = torch.cat([edge_feats] * 2, dim=0) if len(edge_feats) else edge_feats

    # Construct line graph edges (connecting edges in the original graph)
    line_graph_edge_index = torch.LongTensor([])
    if edge_list.nelement() != 0:
        conn = (edge_list[:, 1].unsqueeze(1) == edge_list[:, 0].unsqueeze(0)) & (
            edge_list[:, 0].unsqueeze(1) != edge_list[:, 1].unsqueeze(0)
        )
        line_graph_edge_index = conn.nonzero(as_tuple=False).T
    new_edge_index = edge_list.T

    # Construct the graph with nodes, edges, and additional attributes
    mol_graph = Data(
        x=features, 
        edge_index=new_edge_index, 
        line_graph_edge_index=line_graph_edge_index, 
        edge_attr=edge_feats
    )
    return mol_graph

def main():
    # Initialize UniMolRepr with molecule data type and optionally remove hydrogens
    clf = UniMolRepr(data_type='molecule', remove_hs=args.remove_hs)
    clf.params['max_atoms'] = 1000  # Set maximum atom limit

    # Preprocess data for each specified dataset
    for dataset in ['davis', 'kiba']:
        data_dir = os.path.join('../', args.dataset_root, dataset)
        if not os.path.exists(data_dir):
            print(f'Cannot find {data_dir}')
            continue

        # Prepare directory to save processed data
        split_dir = os.path.join(data_dir, 'processed')
        os.makedirs(split_dir, exist_ok=True)



        ligands = pd.read_csv(os.path.join(data_dir, 'unique_compound_iso_smiles.csv'))

        smiles_list = [
            Chem.MolToSmiles(Chem.MolFromSmiles(ligand), isomericSmiles=True) for ligand in ligands['compound_iso_smiles']
        ]

        smiles_key = ligands['compound_iso_smiles'].tolist()


        # Generate atomic-level representations with UniMolRepr
        unimol_repr = clf.get_repr(smiles_list, return_atomic_reprs=True)
        unimol_repr = [torch.from_numpy(repr) for repr in unimol_repr['atomic_reprs']]

        # Convert each molecule to a graph and collect them in a list
        compound_graph_dict = {}
        for smiles, features,key in zip(smiles_list, unimol_repr, smiles_key):
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            compound_graph = mol_to_graph(mol, features)
            compound_graph_dict[key] = compound_graph

        # Save the list of compound graphs to disk
        torch.save(compound_graph_dict, os.path.join(split_dir, 'unimol_compounds.pt'))

if __name__ == '__main__':
    # Parse command-line arguments for dataset root path and hydrogen removal option
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='data', help="Root directory for datasets.")
    parser.add_argument('--remove_hs', type=bool, default=False, help="Whether to remove hydrogens from molecules.")
    args = parser.parse_args()
    main()
