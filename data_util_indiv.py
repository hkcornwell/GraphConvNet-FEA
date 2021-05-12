import numpy as np
import csv
from os import listdir
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader



def find_csv_filenames( path_to_dir, prefix='graph', suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [path_to_dir + filename for filename in filenames if filename.endswith(suffix) and filename.startswith(prefix)]


def gather_data(graph_file, node_file, res_file):
    # gather graph data
    edges_list = []
    nodes_list = []
    res_list = []

    start_node_num = start_idx = len(nodes_list)-1
    with open(graph_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            for i in range(1, len(row)):
                edges_list.append([start_node_num + int(row[0]), start_node_num + int(row[i])])

    # gather node data (x/y location)
    with open(node_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            nodes_list.append([float(row[1]), float(row[2])])

    # gather stress result data (data labels)
    with open(res_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            res_list.append(float(row[1]))

    # convert data to tensors
    x = torch.as_tensor(nodes_list)
    y = torch.as_tensor(res_list)
    edge_index = torch.as_tensor(edges_list).T

    y_norm = y/torch.max(y)
    data_obj = Data(x, edge_index, y=y_norm)

    return data_obj


def get_data(path_train, path_test, plot=False):
    # get filenames for all graph, node, and results files in training directory
    graph_files = find_csv_filenames(path_train, 'graph')
    node_files = find_csv_filenames(path_train, 'node')
    res_files = find_csv_filenames(path_train, 'Res')

    # get filenames for all graph, node, and results files in testing directory
    graph_files += find_csv_filenames(path_test, 'graph')
    node_files += find_csv_filenames(path_test, 'node')
    res_files += find_csv_filenames(path_test, 'Res')

    data_list = []

    # train_length is the location of the last training index. Needed to generate masks.
    for scenario in range(len(graph_files)):
        data_list.append(gather_data(graph_files[scenario], node_files[scenario], res_files[scenario]))

    # loader = DataLoader(data_list, batch_size=32)

    # # generate train and test masks
    # train_mask = torch.zeros((len(nodes)), dtype=torch.bool)
    # test_mask = torch.zeros((len(nodes)), dtype=torch.bool)
    # train_mask[:train_length] = True
    # test_mask[train_length:] = True
    #
    # # convert data to tensors
    # x = torch.as_tensor(nodes)
    # y = torch.Tensor(res)
    # edge_index = torch.as_tensor(edges).T

    return data_list


if __name__ == "__main__":
    path_train = 'data/train/'
    path_test = 'data/test/'
    data = get_data(path_train, path_test)

    for i in range(len(data)):
        plt.clf()
        plt.title('Datapoint {} Visualization'.format(i))
        plt.scatter(data[i].x[:, 0], data[i].x[:, 1], c=data[i].y)
        plt.savefig('datapoint_vis/dp{}_actual.png'.format(i))

    # print(data)