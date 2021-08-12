# yys_dataset_practice_1

"""
Author :Yingying song
Time : 2021/08/03
E-mail : 3113457266@qq
"""

import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data
from reps_to_graph import smi_to_graph


class CosmeticDataset(InMemoryDataset):
    def __init__(self, root=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.data, self.slices = torch.load(self.processed_paths[0])
        self._data_list = None

    '''copy authority text'''
    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    '''copy authority text'''
    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    @property
    def raw_file_names(self):
        return 'raw.csv'

    @property
    def processed_file_names(self):
        return 'processed.pt'

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.raw_paths[0], header=None)
        df.loc[:, 1:] = df.loc[:, 1:].astype(float)
        raw_data = df.values.tolist()# a list to represtent raw values

        data_list = []

        for _raw in tqdm(raw_data):
            smiles = _raw[0]
            target = _raw[1:]
            atom_fvs, bond_index, bond_fvs = smi_to_graph(smiles)

            x = torch.tensor(atom_fvs, dtype=torch.float)
            edge_index = torch.tensor(bond_index, dtype=torch.long)
            edge_attr = torch.tensor(bond_fvs, dtype=torch.float)
            y = torch.tensor(target, dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
