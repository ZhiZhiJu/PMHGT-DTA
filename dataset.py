from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data as DATA
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import os
from torch_geometric.data import DataLoader
# from torch.utils.data import DataLoader, Dataset

import re
from smiles2graph import smile2graph4drugood


def create_fold_setting_cold(df, fold_seed, frac, entities):
    """create cold-split where given one or multiple columns, it first splits based on
    entities in the columns and then maps all associated data points to the partition

    Args:
            df (pd.DataFrame): dataset dataframe
            fold_seed (int): the random seed
            frac (list): a list of train/valid/test fractions
            entities (Union[str, List[str]]): either a single "cold" entity or a list of
                    "cold" entities on which the split is done

    Returns:
            dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
    """
    if isinstance(entities, str):
        entities = [entities]

    train_frac, val_frac, test_frac = frac

    # For each entity, sample the instances belonging to the test datasets
    test_entity_instances = [
        df[e]
        .drop_duplicates()
        .sample(frac=test_frac, replace=False, random_state=fold_seed)
        .values
        for e in entities
    ]

    # Select samples where all entities are in the test set
    test = df.copy()
    for entity, instances in zip(entities, test_entity_instances):
        test = test[test[entity].isin(instances)]

    if len(test) == 0:
        raise ValueError(
            "No test samples found. Try another seed, increasing the test frac or a "
            "less stringent splitting strategy."
        )

    # Proceed with validation data
    train_val = df.copy()
    for i, e in enumerate(entities):
        train_val = train_val[~train_val[e].isin(test_entity_instances[i])]

    val_entity_instances = [
        train_val[e]
        .drop_duplicates()
        .sample(frac=val_frac / (1 - test_frac), replace=False, random_state=fold_seed)
        .values
        for e in entities
    ]
    val = train_val.copy()
    for entity, instances in zip(entities, val_entity_instances):
        val = val[val[entity].isin(instances)]

    if len(val) == 0:
        raise ValueError(
            "No validation samples found. Try another seed, increasing the test frac "
            "or a less stringent splitting strategy."
        )

    train = train_val.copy()
    for i, e in enumerate(entities):
        train = train[~train[e].isin(val_entity_instances[i])]

    return {
        "train": train.reset_index(drop=True),
        "valid": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }

class DTADataset(InMemoryDataset):
    def __init__(self, root, path, smiles_emb, target_emb, smiles_idx, smiles_graph, target_graph, smiles_len, target_len):

        super(DTADataset, self).__init__(root)
        self.path = path
        df = pd.read_csv(path)
        self.data = df

        self.target_graph_dict = {}
        uniprot_list = df['uniprot'].unique()
        for uniprot in uniprot_list:
            pro_graph =  torch.load('../kiba/graphs_ism_po'
                                     + '/'+ uniprot + '.pt', map_location='cpu')
            self.target_graph_dict[uniprot] = pro_graph
        self.drug_graph_dict = torch.load(r"../kiba/unimol_compounds.pt")

        self.smiles_emb = smiles_emb
        self.target_emb = target_emb
        self.smiles_len = smiles_len
        self.target_len = target_len

        self.process(df, smiles_emb, target_emb, smiles_idx, smiles_graph, target_graph, smiles_len, target_len)



    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['process.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, df, smiles_emb, target_emb, smiles_idx, smiles_graph, target_graph, smiles_len, target_len):
        if self.pre_filter is not None:
            self.data = [data for data in self.data if self.pre_filter(self.data)]

        if self.pre_transform is not None:
            self.data = [self.pre_transform(data) for data in self.data]
    pass


    def off_adj(self, adj, size):
        adj1 = adj.copy()
        for i in range(adj1.shape[0]):
            adj1[i][0] += size
            adj1[i][1] += size
        return adj1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        da = self.data.iloc[idx, :]
        uniprot = da['uniprot']
        target_graph = self.target_graph_dict[uniprot]
        sm = da.loc['compound_iso_smiles']

        drug_graph = self.drug_graph_dict[sm]


        target = da.loc['target_key']
        seq = da.loc['target_sequence']
        label = da.loc['affinity']


        smiles = self.smiles_emb[sm]
        protein = self.target_emb[seq]
        smiles_lengths = self.smiles_len[sm]
        protein_lengths = self.target_len[seq]
        Data = DATA(y=torch.FloatTensor([label]),
                    sm=sm,
                    target=target,
                    smiles=smiles,
                    protein=protein,
                    smiles_lengths=smiles_lengths,
                    protein_lengths=protein_lengths,
                    seq=seq
                    )
        return Data, drug_graph, target_graph