import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader

from data import CosmeticDataset

target = 0
dim = 100

'''following the name of the keeping data folder'''
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
dataset = CosmeticDataset(root=path).shuffle()

# Normalize targets to mean = 0 and std = 1.
'''Attention the y of dimensionality'''
mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
# make sure the dimensionality of mean and std is dim 2
mean = mean.unsqueeze(0)
std = std.unsqueeze(0)
# set the dimensionality of data.y is dim 1
dataset.data.y = (dataset.data.y - mean) / std
dataset.data.y = dataset.data.y.squeeze(0)
mean, std = mean[:, target].item(), std[:, target].item()

# Split datasets.
test_dataset = dataset[:10]
val_dataset = dataset[10:30]
train_dataset = dataset[30:]
test_loader = DataLoader(test_dataset, batch_size=224, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=224,  shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=224, shuffle=True)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin0 = torch.nn.Linear(dataset.num_features, dim)
        nn = Sequential(Linear(12, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=5,
                                                       min_lr=0.00001)


def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        # transform the data to cpu, the step is invariable
        data = data.to(device)
        # choose suitable optimizer and process
        optimizer.zero_grad()
        # loss function is MSE Loss function
        loss = F.mse_loss(model(data), data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def test(loader):
    model.eval()
    error = 0

    # There are the official routine, can copy directly
    for data in loader:
        data = data.to(device)
        error += (model(data) * std - data.y * std).abs().sum().item()  # MAE
    return error / len(loader.dataset)


best_val_error = None
for epoch in range(1, 301):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train(epoch)
    val_error = test(val_loader)
    scheduler.step(val_error)

    if best_val_error is None or val_error <= best_val_error:
        test_error = test(test_loader)
        best_val_error = val_error

    print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
          'Test MAE: {:.7f}'.format(epoch, lr, loss, val_error, test_error))
