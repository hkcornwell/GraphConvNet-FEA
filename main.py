import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import gc

from torch_geometric.data import DataLoader
from data_util_indiv import get_data

from torch_geometric.nn.conv import GCNConv
#from torch_geometric.nn.conv import TAGConv
#from torch_geometric.nn.conv import ChebConv
#from torch_geometric.nn.conv import MFConv


gc.collect()
torch.cuda.empty_cache()

class Net(torch.nn.Module):
    def __init__(self, dataset, hidden_size):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset[0].num_node_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, 1) # output of size one because each node label is a scalar

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        # For regression, return the raw score (no softmax)
        return x


def test(data, train=True):
    model.eval()

    mse = []
    for d in data:
        pred = model(d).squeeze(1)
        mse.append(F.mse_loss(pred, d.y))

    return mse


def train(data, epochs, plot=False):
    train_performance, test_performance = list(), list()

    train_set = data[:6]
    test_set = data[6]

    train_order = [i for i in range(len(train_set))]

    for epoch in range(epochs):
        train_mse = []
        val_mse = []
        for k in range(len(train_set)):
            for i in train_order:
                model.train()
                optimizer.zero_grad()
                out = model(train_set[i])
                out = out.flatten()
                loss = F.mse_loss(out, train_set[i].y)        # use MSE loss for regression
                loss.backward()
                optimizer.step()

            train_mse += test([train_set[x] for x in train_order[:5]])
            val_mse += test([train_set[train_order[5]]])

            train_order = train_order[1:] + [train_order[0]]

        train_mse_avg = sum(train_mse)/len(train_mse)
        val_mse_avg = sum(val_mse)/len(val_mse)

        train_performance.append(train_mse_avg)
        test_performance.append(val_mse_avg)
        print('Epoch: {:03d}, Loss: {:.5f}, Train MSE: {:.5f}, Val MSE: {:.5f}'.
              format(epoch, 0, train_mse_avg, val_mse_avg))


    test_mse = test([test_set])
    print('Test MSE: {:.5f}'.format(test_mse[0]))


    if plot:
        # plot 2d visualization
        data_test = test_set.x
        x_loc = np.array(data_test[:, 0].cpu())
        y_loc = np.array(data_test[:, 1].cpu())
        model.eval()
        pred = model(test_set).squeeze(1)
        plt.title('Validation Result Visualization')
        plt.scatter(x_loc, y_loc, c=pred.cpu().detach().numpy())
        plt.show()
        plt.clf()

        # plot learning curve
        plt.plot(train_performance, label="Train")
        plt.plot(test_performance, label="Validation")
        plt.xlabel("# Epoch")
        plt.ylabel("MSE")
        plt.legend(loc='upper right')
        plt.show()

if __name__ == "__main__":
    lr = 0.01
    lr_decay = 5e-4
    wt_decay = 1e-7
    epochs = 101
    hidden_size = 16

    # generate dataset from files using data_util module
    data_list = get_data(path_train='data/train/', path_test='data/test/', plot=True)

    loader = DataLoader(data_list, batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(loader.dataset, hidden_size).to(device)
    data = [loader.dataset[x].to(device) for x in range(len(loader.dataset))]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wt_decay)

    train(data, epochs, plot=True)
