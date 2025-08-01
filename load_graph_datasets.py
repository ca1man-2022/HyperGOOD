import torch
import os
import pickle
import pdb

import os.path as osp
import numpy as np
import pandas as pd
import scipy.sparse as sp

from torch_geometric.data import Data
from torch_sparse import coalesce  # 用于合并重复的边
# from randomperm_code import random_planetoid_splits
from sklearn.feature_extraction.text import CountVectorizer


def load_LE_dataset(path=None, dataset="ModelNet40", train_percent=0.025):
    # load edges, features, and labels.
    print('Loading {} dataset...'.format(dataset))

    file_name = f'{dataset}.content'
    p2idx_features_labels = osp.join(path, dataset, file_name)
    idx_features_labels = np.genfromtxt(p2idx_features_labels,
                                        dtype=np.dtype(str))
    # features = np.array(idx_features_labels[:, 1:-1])
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    #     labels = encode_onehot(idx_features_labels[:, -1])
    labels = torch.LongTensor(idx_features_labels[:, -1].astype(float))

    print('load features')

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # print('idx:',idx)
    idx_map = {j: i for i, j in enumerate(idx)}
    # print('idx_map:',idx_map)
    file_name = f'{dataset}.edges'
    p2edges_unordered = osp.join(path, dataset, file_name)
    edges_unordered = np.genfromtxt(p2edges_unordered,
                                    dtype=np.int32)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    print('load edges')
    print('edges:',edges)
    projected_features = torch.FloatTensor(np.array(features.todense()))

    # From adjacency matrix to edge_list
    edge_index = edges.T
    print('edge_index:', edge_index)
    #     ipdb.set_trace()
    assert edge_index[0].max() == edge_index[1].min() - 1

    # check if values in edge_index is consecutive. i.e. no missing value for node_id/he_id.
    assert len(np.unique(edge_index)) == edge_index.max() + 1

    num_nodes = edge_index[0].max() + 1
    print('data.num_nodes:', num_nodes)
    num_he = edge_index[1].max() - num_nodes + 1

    edge_index = np.hstack((edge_index, edge_index[::-1, :]))
    # ipdb.set_trace()

    # build torch data class
    data = Data(
        #             x = projected_features,
        x=torch.FloatTensor(np.array(features[:num_nodes].todense())),
        edge_index=torch.LongTensor(edge_index),
        y=labels[:num_nodes])

    # ipdb.set_trace()
    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates.
    total_num_node_id_he_id = len(np.unique(edge_index))
    data.edge_index, data.edge_attr = coalesce(data.edge_index,
                                               None,
                                               total_num_node_id_he_id,
                                               total_num_node_id_he_id)



    #     # generate train, test, val mask.
    n_x = num_nodes
    #     n_x = n_expanded
    num_class = len(np.unique(labels[:num_nodes].numpy()))
    val_lb = int(n_x * train_percent)
    percls_trn = int(round(train_percent * n_x / num_class))
    # data = random_planetoid_splits(data, num_class, percls_trn, val_lb)
    data.n_x = n_x
    # add parameters to attribute

    data.train_percent = train_percent
    data.num_hyperedges = num_he

    return data


def load_citation_dataset(path='../hyperGCN/data/', dataset='citation', train_percent=0.5):
    '''
    this will read the citation dataset from HyperGCN, and convert it edge_list to
    [[ -V- | -E- ]
     [ -E- | -V- ]]
    '''

    with open(osp.join(path, dataset, 'features.pickle'), 'rb') as f:
        features = pickle.load(f)
        features = features.todense()

    # then load node labels:
    with open(osp.join(path, dataset, 'labels.pickle'), 'rb') as f:
        labels = pickle.load(f)

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # The last, load hypergraph.
    with open(osp.join(path, dataset, 'hypergraph.pickle'), 'rb') as f:
        # hypergraph in hyperGCN is in the form of a dictionary.
        # { hyperedge: [list of nodes in the he], ...} 键是超边，值是包含在该超边中的节点列表
        hypergraph = pickle.load(f)

    print(f'number of hyperedges: {len(hypergraph)}')

    # 初始化邻接矩阵
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    edge_idx = num_nodes
    node_list = []
    edge_list = []
    # 遍历每个超边，将其拆分为普通边
    for he in hypergraph.values():
        cur_he = list(he)  # 当前超边中的节点列表
        cur_size = len(cur_he)  # 超边连接的节点数量
        node_list += list(cur_he)  # 将当前超边中的节点添加到节点列表 node_list 中

        # 将超边中的节点两两组合成普通边
        for i in range(cur_size):
            for j in range(i + 1, cur_size):
                node1 = cur_he[i]
                node2 = cur_he[j]
                adj_matrix[node1, node2] = 1
                adj_matrix[node2, node1] = 1  # 确保邻接矩阵是对称的
    
        '''
        假设当前的超边索引 edge_idx 为 10，超图中的某个超边 he 包含了三个节点 [1, 2, 3]。
        此时，对应的 edge_list 将会是 [10, 10, 10]，即包含了三个 10，因为这三个节点都属于同一个超边
        '''
        edge_list += [edge_idx] * cur_size

        edge_idx += 1
    '''
    [[1, 2, 3, 4, 5, 10, 10, 10, 11, 11],
    [10, 10, 10, 11, 11, 1, 2, 3, 4, 5]] 
    '''
    edge_index = np.array([node_list + edge_list,
                           edge_list + node_list], dtype=np.int_)
    edge_index = torch.LongTensor(edge_index)

    adj = adj_matrix
    # 构建普通图数据对象
    data = Data(
        x=features,
        edge_index=edge_index,
        y=labels,
        adj=adj
    )

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates.
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index,
                                               None,
                                               total_num_node_id_he_id,
                                               total_num_node_id_he_id)

    n_x = num_nodes
    #     n_x = n_expanded
    num_class = len(np.unique(labels.numpy()))
    val_lb = int(n_x * train_percent)
    percls_trn = int(round(train_percent * n_x / num_class))
    # data = random_planetoid_splits(data, num_class, percls_trn, val_lb)
    data.n_x = n_x
    # add parameters to attribute

    data.train_percent = train_percent
    data.num_hyperedges = len(hypergraph)

    return data

def load_other_dataset(path='../hyperGCN/data/', dataset='citation', train_percent=0.5):
    '''
    this will read the citation dataset from HyperGCN, and convert it edge_list to
    [[ -V- | -E- ]
     [ -E- | -V- ]]
    '''
    # print(f'Loading hypergraph dataset from hyperGCN: {dataset}')

    # first load node features:
    # .../data/AllSet_all_raw_data/cocitation/citeseer/features.pickle
    with open(osp.join(path, dataset, 'features.pickle'), 'rb') as f:
        features = pickle.load(f)
        # features = features.todense()

    # then load node labels:
    with open(osp.join(path, dataset, 'labels.pickle'), 'rb') as f:
        labels = pickle.load(f)

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')

    features = torch.FloatTensor(features)
    if (labels.dtype == object):
        labels = np.array(labels, dtype=int)
    labels = torch.LongTensor(labels)

    # The last, load hypergraph.
    with open(osp.join(path, dataset, 'hypergraph.pickle'), 'rb') as f:
        # hypergraph in hyperGCN is in the form of a dictionary.
        # { hyperedge: [list of nodes in the he], ...} 键是超边，值是包含在该超边中的节点列表
        hypergraph = pickle.load(f)

    print(f'number of hyperedges: {len(hypergraph)}')

    edge_idx = num_nodes
    node_list = []
    edge_list = []
    for he in hypergraph.keys():
        cur_he = hypergraph[he]  # 对于每个超边 he，从超图中获取相应的节点列表
        cur_size = len(cur_he)  # 超边连接的节点数量

        node_list += list(cur_he)  # 将当前超边中的节点添加到节点列表 node_list 中
        '''
        假设当前的超边索引 edge_idx 为 10，超图中的某个超边 he 包含了三个节点 [1, 2, 3]。
        此时，对应的 edge_list 将会是 [10, 10, 10]，即包含了三个 10，因为这三个节点都属于同一个超边
        '''
        edge_list += [edge_idx] * cur_size

        edge_idx += 1
    '''
    [[1, 2, 3, 4, 5, 10, 10, 10, 11, 11],
    [10, 10, 10, 11, 11, 1, 2, 3, 4, 5]] 
    '''
    edge_index = np.array([node_list + edge_list,
                           edge_list + node_list], dtype=np.int_)
    edge_index = torch.LongTensor(edge_index)

    data = Data(x=features,
                edge_index=edge_index,
                y=labels)

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates.
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index,
                                               None,
                                               total_num_node_id_he_id,
                                               total_num_node_id_he_id)

    n_x = num_nodes
    #     n_x = n_expanded
    num_class = len(np.unique(labels.numpy()))
    val_lb = int(n_x * train_percent)
    percls_trn = int(round(train_percent * n_x / num_class))
    # data = random_planetoid_splits(data, num_class, percls_trn, val_lb)
    data.n_x = n_x
    # add parameters to attribute

    data.train_percent = train_percent
    data.num_hyperedges = len(hypergraph)

    return data

def load_yelp_dataset(path='../data/raw_data/yelp_raw_datasets/', dataset='yelp',
                      name_dictionary_size=1000,
                      train_percent=0.025):
    '''
    this will read the yelp dataset from source files, and convert it edge_list to
    [[ -V- | -E- ]
     [ -E- | -V- ]]

    each node is a restaurant, a hyperedge represent a set of restaurants one user had been to.

    node features:
        - latitude, longitude
        - state, in one-hot coding.
        - city, in one-hot coding.
        - name, in bag-of-words

    node label:
        - average stars from 2-10, converted from original stars which is binned in x.5, min stars = 1
    '''
    print(f'Loading hypergraph dataset from {dataset}')

    # first load node features:
    # load longtitude and latitude of restaurant.
    latlong = pd.read_csv(osp.join(path, 'yelp_restaurant_latlong.csv')).values

    # city - zipcode - state integer indicator dataframe.
    loc = pd.read_csv(osp.join(path, 'yelp_restaurant_locations.csv'))
    state_int = loc.state_int.values
    city_int = loc.city_int.values

    num_nodes = loc.shape[0]
    state_1hot = np.zeros((num_nodes, state_int.max()))
    state_1hot[np.arange(num_nodes), state_int - 1] = 1

    city_1hot = np.zeros((num_nodes, city_int.max()))
    city_1hot[np.arange(num_nodes), city_int - 1] = 1

    # convert restaurant name into bag-of-words feature.
    vectorizer = CountVectorizer(max_features=name_dictionary_size, stop_words='english', strip_accents='ascii')
    res_name = pd.read_csv(osp.join(path, 'yelp_restaurant_name.csv')).values.flatten()
    name_bow = vectorizer.fit_transform(res_name).todense()

    features = np.hstack([latlong, state_1hot, city_1hot, name_bow])

    # then load node labels:
    df_labels = pd.read_csv(osp.join(path, 'yelp_restaurant_business_stars.csv'))
    labels = df_labels.values.flatten()

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # The last, load hypergraph.
    # Yelp restaurant review hypergraph is store in a incidence matrix.
    H = pd.read_csv(osp.join(path, 'yelp_restaurant_incidence_H.csv'))
    node_list = H.node.values - 1
    edge_list = H.he.values - 1 + num_nodes

    edge_index = np.vstack([node_list, edge_list])
    edge_index = np.hstack([edge_index, edge_index[::-1, :]])

    edge_index = torch.LongTensor(edge_index)

    data = Data(x=features,
                edge_index=edge_index,
                y=labels)

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates.
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index,
                                               None,
                                               total_num_node_id_he_id,
                                               total_num_node_id_he_id)

    n_x = num_nodes
    #     n_x = n_expanded
    num_class = len(np.unique(labels.numpy()))
    val_lb = int(n_x * train_percent)
    percls_trn = int(round(train_percent * n_x / num_class))
    # data = random_planetoid_splits(data, num_class, percls_trn, val_lb)
    data.n_x = n_x
    # add parameters to attribute

    data.train_percent = train_percent
    data.num_hyperedges = H.he.values.max()

    return data


def load_cornell_dataset(path='./data/raw_data/', dataset='amazon',
                         feature_noise=0.1,
                         feature_dim=None,
                         train_percent=0.025):
    '''
    this will read the yelp dataset from source files, and convert it edge_list to
    [[ -V- | -E- ]
     [ -E- | -V- ]]

    each node is a restaurant, a hyperedge represent a set of restaurants one user had been to.

    node features:
        - add gaussian noise with sigma = nosie, mean = one hot coded label.

    node label:
        - average stars from 2-10, converted from original stars which is binned in x.5, min stars = 1
    '''
    print(f'Loading hypergraph dataset from cornell: {dataset}')

    # first load node labels
    df_labels = pd.read_csv(osp.join(path, dataset, f'node-labels-{dataset}.txt'), names=['node_label'])

    num_nodes = df_labels.shape[0]
    labels = df_labels.values.flatten()

    # then create node features.
    num_classes = df_labels.values.max()
    features = np.zeros((num_nodes, num_classes))

    features[np.arange(num_nodes), labels - 1] = 1
    if feature_dim is not None:
        num_row, num_col = features.shape
        zero_col = np.zeros((num_row, feature_dim - num_col), dtype=features.dtype)
        features = np.hstack((features, zero_col))
    features = np.random.normal(features, feature_noise, features.shape)

    print(f'number of nodes:{num_nodes}, feature dimension: {features.shape[1]}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # The last, load hypergraph.
    # Corenll datasets are stored in lines of hyperedges. Each line is the set of nodes for that edge.
    p2hyperedge_list = osp.join(path, dataset, f'hyperedges-{dataset}.txt')
    node_list = []
    he_list = []
    he_id = num_nodes

    with open(p2hyperedge_list, 'r') as f:
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            cur_set = line.split(',')
            cur_set = [int(x) for x in cur_set]

            node_list += cur_set
            he_list += [he_id] * len(cur_set)
            he_id += 1
    # shift node_idx to start with 0.
    node_idx_min = np.min(node_list)
    node_list = [x - node_idx_min for x in node_list]

    edge_index = [node_list + he_list,
                  he_list + node_list]

    edge_index = torch.LongTensor(edge_index)

    data = Data(x=features,
                edge_index=edge_index,
                y=labels)

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates.
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index,
                                               None,
                                               total_num_node_id_he_id,
                                               total_num_node_id_he_id)

    n_x = num_nodes
    #     n_x = n_expanded
    num_class = len(np.unique(labels.numpy()))
    val_lb = int(n_x * train_percent)
    percls_trn = int(round(train_percent * n_x / num_class))
    # data = random_planetoid_splits(data, num_class, percls_trn, val_lb)
    data.n_x = n_x
    # add parameters to attribute

    data.train_percent = train_percent
    data.num_hyperedges = he_id - num_nodes

    return data


# if __name__ == '__main__':
#     import pdb

#     data = load_citation_dataset(path='./data/AllSet_all_raw_data/cocitation/', dataset='cora')
#     pdb.set_trace()

