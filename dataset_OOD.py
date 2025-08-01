import torch
import pickle
import os
import os.path as osp
from torch_geometric.data import InMemoryDataset, Data
# from load_other_datasets import * 
from load_graph_datasets import * 
import matplotlib.pyplot as plt
import pdb


def save_data_to_pickle(data, p2root='../data/', file_name=None):
    surfix = 'star_expansion_dataset'
    if file_name is None:
        tmp_data_name = '_'.join(['Hypergraph', surfix])
    else:
        tmp_data_name = file_name
    p2he_StarExpan = osp.join(p2root, tmp_data_name)
    if not osp.isdir(p2root):
        os.makedirs(p2root)
    with open(p2he_StarExpan, 'bw') as f:
        pickle.dump(data, f)
    return p2he_StarExpan

def create_feat_noise_dataset(data, noise_level=0.1, control_noise=True):
    x = data.x
    n = data.num_nodes

    if control_noise:
        # 控制扰动等级
        idx = torch.randint(0, n, (n, 2))  # 随机选择节点对
        weight = torch.rand(n).unsqueeze(1)  # 随机权重
        x_new = x[idx[:, 0]] * weight + x[idx[:, 1]] * (1 - weight)  # 插值生成新特征
        x_new = x + noise_level * (x_new - x)  # 添加噪声
    else:
        idx = torch.randint(0, n, (n, 2))
        weight = torch.rand(n).unsqueeze(1) 
        x_new = x[idx[:, 0]] * weight + x[idx[:, 1]] * (1 - weight)

    # 创建新的 Data 对象，并复制原数据集的所有属性
    dataset = Data(
        x=x_new,
        edge_index=data.edge_index,
        y=data.y,
        n_x=data.n_x,  # 复制节点数
        train_percent=data.train_percent,  # 复制训练集比例
        num_hyperedges=data.num_hyperedges,  # 复制超边数
        # 其他属性也可以在这里复制
    )
    dataset.node_idx = torch.arange(n)  # 添加节点索引
    return dataset

def create_label_noise_dataset(data):
    # 随机修改50%节点的标签
    y = data.y
    n = data.num_nodes
    
    idx = torch.randperm(n)[:int(n * 0.7)]
    y_new = y.clone()
    y_new[idx] = torch.randint(0, y.max() + 1, (int(n * 0.7), ))
    
    dataset = Data(
        x=data.x,
        edge_index=data.edge_index,
        y=y_new,
        n_x=data.n_x, 
        train_percent=data.train_percent,  
        num_hyperedges=data.num_hyperedges, 
    )
    dataset.node_idx = torch.arange(n)
    return dataset

def get_hyperedge_size_distribution(edge_index):
    """
    计算原始数据中每个超边的节点数量分布（只看node->hyperedge部分）
    """
    # 筛选 前部分的边
    mask = edge_index[0] < edge_index[1] # 假设节点索引 < 超边索引
    node2hyperedge = edge_index[:, mask]
    
    # 每个超边出现多少次（即被多少节点连接）
    _, counts = torch.unique(node2hyperedge[1], return_counts=True)
    max_size = counts.max().item()
    size_counts = torch.bincount(counts, minlength=max_size+1)[1:]  # 跳过 size=0
    size_probs = size_counts.float() / size_counts.sum()
    return size_probs

def create_hypergraph_structure_perturbation(data, p_add=0.5, p_remove=0.7):
    edge_index = data.edge_index.clone()
    num_nodes = data.num_nodes
    edge0, edge1 = edge_index

    # 1. 选出 node → hyperedge 方向的边
    mask_v2e = edge0 < edge1  # node_id < hyperedge_id
    node2edge = edge_index[:, mask_v2e]

    # 获取所有超边
    hyperedge_ids = node2edge[1]
    num_hyperedges = hyperedge_ids.max().item() + 1

    # 2. 随机删除一些超边（mask）
    hyperedge_mask = torch.rand(num_hyperedges) > p_remove
    keep_mask = hyperedge_mask[hyperedge_ids]  # 根据超边 ID 筛选边
    kept_node2edge = node2edge[:, keep_mask]

    # 3. 构建完整的 kept_edge_index（包含双向）
    kept_edge_index = torch.cat([
        kept_node2edge,
        kept_node2edge[[1, 0]],  # 添加反向边 hyperedge → node
    ], dim=1)

    # 4. 统计 retained 超边大小分布
    size_probs = get_hyperedge_size_distribution(kept_edge_index)

    # 5. 新增若干个超边
    num_new = int(hyperedge_mask.sum().item() * p_add)
    new_edges = []
    for i in range(num_new):
        k = torch.multinomial(size_probs, 1).item() + 1
        nodes = torch.randperm(num_nodes)[:k]
        new_hyperedge_id = num_hyperedges + i
        edge_pairs = torch.stack([
            torch.cat([nodes, torch.full((k,), new_hyperedge_id)]),
            torch.cat([torch.full((k,), new_hyperedge_id), nodes])
        ], dim=0)
        new_edges.append(edge_pairs)

    # 6. 合并所有边
    if new_edges:
        new_edges_cat = torch.cat(new_edges, dim=1)
        final_edge_index = torch.cat([kept_edge_index, new_edges_cat], dim=1)
    else:
        final_edge_index = kept_edge_index

    return final_edge_index

def create_structure_dataset(data, p_remove=0.7, p_add=0.5):
    """
    生成结构扰动的超图数据集
    Args:
        p_remove: 删除原超边的概率 
        p_add: 新增超边的概率（相对于原超边数）
    """
    # 生成扰动后的超边索引
    perturbed_edge_index = create_hypergraph_structure_perturbation(
        data, p_add=p_add, p_remove=p_remove
    )

    # 构建扰动后的数据集
    dataset = Data(
        x=data.x,
        edge_index=perturbed_edge_index,
        y=data.y,
        n_x=data.n_x,
        train_percent=data.train_percent,
        num_hyperedges=torch.tensor([perturbed_edge_index[1].max() + 1]),
    )
    dataset.node_idx = torch.arange(data.num_nodes)
    
    return dataset


class dataset_Hypergraph(InMemoryDataset):
    def __init__(self, root='./data/pyg_data/hypergraph_dataset_updated/', name=None,
                 p2raw=None, train_percent=0.01, feature_noise=None,
                 transform=None, pre_transform=None, ood_noise_level=0.1, control_noise=True, ood_type=None):

        existing_dataset = ['20newsW100', 'ModelNet40', 'zoo',
                       'NTU2012', 'Mushroom',
                       'coauthor_cora', 'coauthor_dblp',
                       'cofriend_pokec', 'cocreate_twitch', 'cooccurence_actor', 'copurchasing_amazon',
                       'yelp', 'amazon-reviews', 'walmart-trips', 'house-committees',
                       'walmart-trips-100', 'house-committees-100',
                       'cora', 'citeseer', 'pubmed', 'senate-committees-100', 'congress-bills-100']
        if name not in existing_dataset:
            raise ValueError(f'name of hypergraph dataset must be one of: {existing_dataset}')
        else:
            self.name = name
        
        self.feature_noise = feature_noise
        self._train_percent = train_percent
        self.ood_noise_level = ood_noise_level
        self.ood_type = ood_type
        
        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(f'path to raw hypergraph dataset "{p2raw}" does not exist!')
        
        if not osp.isdir(root):
            os.makedirs(root)
            
        self.root = root
        self.myraw_dir = osp.join(root, self.name, 'raw')
        self.myprocessed_dir = osp.join(root, self.name)
        
        super(dataset_Hypergraph, self).__init__(osp.join(root, name), transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_percent = self.data.train_percent
        
    @property
    def raw_file_names(self):
        if self.feature_noise is not None:
            file_names = [f'{self.name}_noise_{self.feature_noise}']
        else:
            file_names = [self.name]
        return file_names

    # @property
    # def processed_file_names(self):
    #     if self.feature_noise is not None:
    #         return [f'data_noise_{self.feature_noise}.pt']
    #     else:
    #         return ['data.pt']
    @property
    def processed_file_names(self):
        if self.feature_noise is not None:
            return [f'data_noise_{self.feature_noise}.pt']
        elif self.ood_type is not None:
            return [
                f'data_ood_tr_{self.ood_type}.pt',
                f'data_ood_te_{self.ood_type}.pt'
            ]
        else:
            return ['data.pt']

    @property
    def num_features(self):
        return self.data.num_node_features

    def download(self):
        for name in self.raw_file_names:
            p2f = osp.join(self.myraw_dir, name)
            if not osp.isfile(p2f):
                if self.name in ['cora', 'citeseer', 'pubmed']:
                    tmp_data = load_citation_dataset(path=self.p2raw, dataset=self.name, train_percent=self._train_percent)
                    print('I AM HERE!!')
                elif self.name in ['pokec', 'twitch', 'actor', 'amazon']:
                    tmp_data = load_other_dataset(path=self.p2raw, dataset=self.name, train_percent=self._train_percent)
                elif self.name in ['coauthor_cora', 'coauthor_dblp']:
                    assert 'coauthorship' in self.p2raw
                    dataset_name = self.name.split('_')[-1]
                    tmp_data = load_citation_dataset(path=self.p2raw, dataset=dataset_name, train_percent=self._train_percent)
                elif self.name in ['amazon-reviews', 'walmart-trips', 'house-committees']:
                    if self.feature_noise is None:
                        raise ValueError(f'for cornell datasets, feature noise cannot be {self.feature_noise}')
                    tmp_data = load_cornell_dataset(path=self.p2raw, dataset=self.name, feature_noise=self.feature_noise, train_percent=self._train_percent)
                elif self.name in ['walmart-trips-100', 'house-committees-100', 'senate-committees-100', 'congress-bills-100']:
                    if self.feature_noise is None:
                        raise ValueError(f'for cornell datasets, feature noise cannot be {self.feature_noise}')
                    feature_dim = int(self.name.split('-')[-1])
                    tmp_name = '-'.join(self.name.split('-')[:-1])
                    tmp_data = load_cornell_dataset(path=self.p2raw, dataset=tmp_name, feature_dim=feature_dim, feature_noise=self.feature_noise, train_percent=self._train_percent)
                elif self.name == 'yelp':
                    tmp_data = load_yelp_dataset(path=self.p2raw, dataset=self.name, train_percent=self._train_percent)
                else:
                    tmp_data = load_LE_dataset(path=self.p2raw, dataset=self.name, train_percent=self._train_percent)
                    
                _ = save_data_to_pickle(tmp_data, p2root=self.myraw_dir, file_name=self.raw_file_names[0])
            else:
                pass

    def process(self):
        p2f = osp.join(self.myraw_dir, self.raw_file_names[0])
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def get_ood_datasets(self, ood_type):
        # Load the original data
        # p2f = osp.join(self.myraw_dir, self.raw_file_names[0])
        # with open(p2f, 'rb') as f:
        #     data = pickle.load(f)
        data = self.data
        print(f"[INFO] Constructing OOD datasets with ood_type={ood_type}")
        self.ood_type = ood_type
        
        # Create OOD datasets by adding noise
        if ood_type == 'feature':
            data_ood_tr = create_feat_noise_dataset(data, noise_level=self.ood_noise_level)
            data_ood_te = create_feat_noise_dataset(data, noise_level=self.ood_noise_level)
        elif ood_type == 'label':
            data_ood_tr = create_label_noise_dataset(data)
            data_ood_te = create_label_noise_dataset(data)
        elif ood_type == 'structure':
            data_ood_tr = create_structure_dataset(data, p_remove=0.7, p_add=0.5)
            data_ood_te = create_structure_dataset(data, p_remove=0.7, p_add=0.5)
        
        elif ood_type == 'cross':
            # 预定义的 cross-domain 配对字典：训练集 -> 测试集
            cross_map = {
                'cora': 'citeseer',
                'citeseer': 'pubmed',
                'pubmed': 'cora',
                'coauthor_cora': 'coauthor_dblp',
                'coauthor_dblp': 'coauthor_cora',
                'walmart-trips': 'house-committees',
                'house-committees': 'walmart-trips',
            }

            if self.name not in cross_map:
                raise ValueError(f"[CROSS-OOD ERROR] No target domain defined for dataset '{self.name}'.")

            target_name = cross_map[self.name]
            
            # 加载源数据（当前对象已经是 source 数据集）
            data_ood_tr = self.data

            # 加载目标数据（测试用）
            tgt_dataset = dataset_Hypergraph(
                root=self.root,
                name=target_name,
                p2raw=self.p2raw,
                train_percent=self._train_percent
            )
            p2tgt = osp.join(tgt_dataset.myraw_dir, tgt_dataset.raw_file_names[0])
            with open(p2tgt, 'rb') as f:
                data_ood_te = pickle.load(f)

        # 补全目标域特征维度，确保与 source 匹配
        if data_ood_te.x.size(1) < data_ood_tr.x.size(1):
            pad_dim = data_ood_tr.x.size(1) - data_ood_te.x.size(1)
            pad = torch.zeros(data_ood_te.x.size(0), pad_dim)
            data_ood_te.x = torch.cat([data_ood_te.x, pad], dim=1)
        elif data_ood_te.x.size(1) > data_ood_tr.x.size(1):
            data_ood_te.x = data_ood_te.x[:, :data_ood_tr.x.size(1)]

        
        # Save OOD datasets
        p2ood_tr = osp.join(self.myprocessed_dir, f'data_ood_tr_{ood_type}.pt')
        p2ood_te = osp.join(self.myprocessed_dir, f'data_ood_te_{ood_type}.pt')
        torch.save(self.collate([data_ood_tr]), p2ood_tr)
        torch.save(self.collate([data_ood_te]), p2ood_te)
        
        # Return IND and OOD datasets
        return self.data, data_ood_tr, data_ood_te

    def __repr__(self):
        return '{}()'.format(self.name)
    
