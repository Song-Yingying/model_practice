import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import Data


target = 0
dim = 100


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.featurizer = False
        # message passing phase
        self.lin0 = torch.nn.Linear(Data.num_features, dim)
        nn = Sequential(Linear(12, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)
        # readout phase
        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)
        # message passing
        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            '''deleting a dimensionality'''
            out = out.squeeze(0)
        # readout
        out = self.set2set(out, data.batch)
        # make GNN as feature extractor
        if self.featurizer and not self.training:
            return out
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        if not self.training:
            out = self.sigmoid(out)
        return out.view(-1)


class CosmeticNet(torch.nn.Module):
    def __init__(self):
        super(CosmeticNet, self).__init__()
        # message passing phase
        self.lin0 = torch.nn.Linear(Data.x, dim)
        nn = Sequential(Linear(12, 128), ReLU(), Linear(128, dim * dim))
        self.conv_ca = NNConv(dim, dim, nn, aggr='mean')
        self.gru_ca = GRU(dim, dim)
        # readout phase
        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(4 * dim, 2 * dim)
        self.lin2 = torch.nn.Linear(2 * dim, dim)
        self.lin3 = torch.nn.Linear(dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, Data):
        Data.x = Data.x.squeeze(-1)
        out = F.relu(self.lin0(Data.x))
        '''adding a dimensionality'''
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, Data.edge_index, Data.edge_attr))
            out, h = self.gru_ca(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, Data.x_batch)
        out = torch.cat((out, out), dim=1)
        out = F.relu(self.lin1(out))
        out = F.relu(self.lin2(out))
        out = self.lin3(out)
        if not self.training:
            out = self.sigmoid(out)
        return out.view(-1)
