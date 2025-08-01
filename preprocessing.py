import numpy as np
import torch
import copy
from collections import defaultdict, Counter

def perturb_hyperedges(data, prop, perturb_type='delete'):
    data_p = copy.deepcopy(data)
    edge_index = data_p.edge_index
    num_node = data.x.shape[0]
    e_idxs = edge_index[1,:] - num_node
    num_edge = (edge_index[1,:].max()) - (edge_index[1,:].min())
    if((perturb_type == 'delete') or (perturb_type == 'replace')):
        p_num = num_edge * prop
        p_num = int(p_num)
        chosen_edges = torch.as_tensor(np.random.permutation(int(num_edge.numpy()))).to(edge_index.device)
        chosen_edges = chosen_edges[:p_num]
        if(perturb_type == 'delete'):
            data_p.edge_index = delete_edges(edge_index, chosen_edges, e_idxs)
        else: # replace = add + delete
            data_p.edge_index = replace_edges(edge_index, chosen_edges, e_idxs, num_node)
    elif(perturb_type == 'add'):
        # p_num = num_edge * prop / (1 - prop)
        p_num = num_edge * prop
        p_num = int(p_num)
        data_p.edge_index = add_edges(edge_index, p_num, num_node)
    return data_p


def delete_edges(edge_index, chosen_edges, e_idxs):
    for i in range(chosen_edges.shape[0]):
        chosen_edge = chosen_edges[i]
        edge_index = edge_index[:, (e_idxs != chosen_edge)]
        e_idxs = e_idxs[(e_idxs != chosen_edge)]
    return edge_index

def replace_edges(edge_index, chosen_edges, e_idxs, num_node):
    edge_index = delete_edges(edge_index, chosen_edges, e_idxs)
    edge_index = add_edges_r(edge_index, chosen_edges, num_node)
    return edge_index

def add_edges_r(edge_index, chosen_edges, num_node):
    edge_idxs = [edge_index]
    for i in range(chosen_edges.shape[0]):
        new_edge = torch.as_tensor(np.random.choice(int(num_node), 16, replace=False)).to(edge_index.device)
        for j in range(new_edge.shape[0]):
            edge_idx_i = torch.zeros([2,1]).to(edge_index.device)
            edge_idx_i[0,0] = new_edge[j]
            edge_idx_i[1,0] = chosen_edges[i] + num_node
            edge_idxs.append(edge_idx_i)
    edge_idxs = torch.cat(edge_idxs, dim=1)
    return torch.tensor(edge_idxs, dtype=torch.int64)
def add_edges(edge_index, p_num, num_node):
    start_e_idx = edge_index[1,:].max() + 1
    edge_idxs = [edge_index]
    for i in range(p_num):
        new_edge = torch.as_tensor(np.random.choice(int(num_node.cpu().numpy()), 5, replace=False)).to(edge_index.device)
        for j in range(new_edge.shape[0]):
            edge_idx_i = torch.zeros([2,1]).to(edge_index.device)
            edge_idx_i[0,0] = new_edge[j]
            edge_idx_i[1,0] = start_e_idx
            edge_idxs.append(edge_idx_i)
        start_e_idx = start_e_idx + 1
    edge_idxs = torch.cat(edge_idxs, dim=1)
    return torch.tensor(edge_idxs, dtype=torch.int64)

def Add_Self_Loops(data):
    # update so we dont jump on some indices
    # Assume edge_index = [V;E]. If not, use ExtractV2E()

    edge_index = data.edge_index
    # print("Data type of edge_index array:", edge_index.dtype)
    num_nodes = data.n_x
    # num_nodes = data.n_x[0]
    # num_hyperedges = data.num_hyperedges[0]
    num_hyperedges = data.num_hyperedges
    #if not ((data.n_x[0] + data.num_hyperedges[0] - 1) == data.edge_index[1].max().item()):
    #    print('num_hyperedges does not match! 2')
    #    return

    hyperedge_appear_fre = Counter(edge_index[1].numpy())
    # store the nodes that already have self-loops
    skip_node_lst = []
    for edge in hyperedge_appear_fre:
        if hyperedge_appear_fre[edge] == 1:
            skip_node = edge_index[0][torch.where(
                edge_index[1] == edge)[0].item()]
            skip_node_lst.append(skip_node.item())

    new_edge_idx = edge_index[1].max() + 1
    print(edge_index[1].max())
    new_edges = torch.zeros(
        (2, num_nodes - len(skip_node_lst)), dtype=edge_index.dtype)
    # print("Data type of new_edges array:", new_edges.dtype)
    tmp_count = 0
    for i in range(num_nodes):
        if i not in skip_node_lst:
            new_edges[0][tmp_count] = i
            new_edges[1][tmp_count] = new_edge_idx
            new_edge_idx += 1
            tmp_count += 1

    data.totedges = num_hyperedges + num_nodes - len(skip_node_lst)
    edge_index = torch.cat((edge_index, new_edges), dim=1)
    # Sort along w.r.t. nodes
    _, sorted_idx = torch.sort(edge_index[0])
    data.edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)
    return data

def ExtractV2E(data):
    # Assume edge_index = [V|E;E|V]
    edge_index = data.edge_index
    edge_index = torch.tensor(edge_index)
#     First, ensure the sorting is correct (increasing along edge_index[0])
    _, sorted_idx = torch.sort(edge_index[0])
    edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)
    num_nodes = data.n_x
    num_hyperedges = data.num_hyperedges
    # print(data.edge_index)
    # print(data.edge_index[0].max().item())
    if not ((data.n_x+data.num_hyperedges-1) == data.edge_index[0].max().item()):
        print('num_hyperedges does not match! 1')
        return
    cidx = torch.where(edge_index[0] == num_nodes)[
        0].min()  # cidx: [V...|cidx E...]
    data.edge_index = edge_index[:, :cidx].type(torch.LongTensor)
    return data
# def ConstructH(data):
#     """
#     Construct incidence matrix H of size (num_nodes, num_hyperedges) from edge_index = [V;E]
#     """
#     edge_index = np.array(data.edge_index)
#     num_nodes = data.n_x
#     num_hyperedges = np.max(edge_index[1])-np.min(edge_index[1])+1
#     H = np.zeros((num_nodes, num_hyperedges))
#     cur_idx = 0
#     for he in np.unique(edge_index[1]):
#         nodes_in_he = edge_index[0][edge_index[1] == he]
#         H[nodes_in_he, cur_idx] = 1.
#         cur_idx += 1
#
# #     data.incident_mat = H
#     return H
def ConstructH(data):
    """
    Construct incidence matrix H of size (num_nodes, num_hyperedges) from edge_index = [V;E]
    """
    if not hasattr(data, 'totedges'):
        raise ValueError("The 'totedges' attribute is missing in the data object. "
                         "Please ensure it is set during data loading or preprocessing.")
    edge_index = np.array(data.edge_index)
    num_nodes = data.n_x
    num_hyperedges = int(data.totedges)
    H = np.zeros((num_nodes, num_hyperedges))
    cur_idx = 0
    for he in np.unique(edge_index[1]):
        nodes_in_he = edge_index[0][edge_index[1] == he]
        H[nodes_in_he, cur_idx] = 1.
        cur_idx += 1

#     data.incident_mat = H
    return H
# def ConstructH(data):
#     """
#     Construct incidence matrix H of size (num_nodes,num_hyperedges) from edge_index = [V;E]
#     """
# #     ipdb.set_trace()
#     edge_index = np.array(data.edge_index)
#     # Don't use edge_index[0].max()+1, as some nodes maybe isolated
#     num_nodes = data.x.shape[0]
#     num_hyperedges = np.max(edge_index[1])-np.min(edge_index[1])+1
#     H = np.zeros((num_nodes, num_hyperedges))
#     cur_idx = 0
#     for he in np.unique(edge_index[1]):
#         nodes_in_he = edge_index[0][edge_index[1] == he]
#         H[nodes_in_he, cur_idx] = 1.
#         cur_idx += 1
#
#     return H

def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True, balance=False):
    """ Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks"""
    """ randomly splits label into train/valid/test splits """
    # 如果不需要平衡数据集
    if not balance:
        # 如果忽略标签为负数的节点
        if ignore_negative:
            # 找出所有非负标签的节点索引
            labeled_nodes = torch.where(label != -1)[0]
        else:
            labeled_nodes = label
        # 获取有标签节点数量
        n = labeled_nodes.shape[0]
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)
        # 生成一个随机排列的索引数组
        perm = torch.as_tensor(np.random.permutation(n))

        # 将随机排列的索引数组分割成训练、验证和测试集的索引
        train_indices = perm[:train_num]
        val_indices = perm[train_num:train_num + valid_num]
        test_indices = perm[train_num + valid_num:]

        # 如果不忽略负值，则直接返回分割后的索引
        if not ignore_negative:
            return train_indices, val_indices, test_indices

        # 如果忽略负值，则根据分割后的索引从有标签的节点中获取对应的索引
        train_idx = labeled_nodes[train_indices]
        valid_idx = labeled_nodes[val_indices]
        test_idx = labeled_nodes[test_indices]


        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}

    # 平衡数据集
    else:
        #         ipdb.set_trace()
        indices = []
        # 对于每个类别，获取该类别的节点索引，随机排列并存储在indices列表中
        for i in range(label.max()+1):
            index = torch.where((label == i))[0].view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
        # 计算每个类别用于训练的节点数和验证集的节点数
        percls_trn = int(train_prop/(label.max()+1)*len(label))
        val_lb = int(valid_prop*len(label))
        # 根据计算出的训练和验证集的节点数，从每个类别中选择相应数量的节点作为训练集，剩余的节点作为验证集和测试集。
        train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]
        valid_idx = rest_index[:val_lb]
        test_idx = rest_index[val_lb:]
        # 构建包含训练、验证和测试集索引的字典
        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
    return split_idx