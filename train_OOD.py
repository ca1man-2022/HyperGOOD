'''
2025/3/12
Python 3.8.2
'''
#!/usr/bin/env python
# coding: utf-8

import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm
from preprocessing import *
from dataset_OOD import dataset_Hypergraph #, convert_hypergraph_to_graph

from load_other_datasets import *
from MSHR import *
from scipy.sparse.linalg import lobpcg
from scipy import sparse
from torch.nn.parameter import Parameter
import os.path as osp
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import pdb
from losses import UB_loss

import seaborn as sns


@torch.no_grad()
def scipy_to_torch_sparse(A):
    A = sparse.coo_matrix(A)
    row = torch.tensor(A.row)
    col = torch.tensor(A.col)
    index = torch.stack((row, col), dim=0)
    value = torch.Tensor(A.data)
    return torch.sparse_coo_tensor(index, value, A.shape)

def ChebyshevApprox(f, n):
    quad_points = 500
    c = np.zeros(n)
    a = np.pi / 2
    for k in range(1, n + 1):
        Integrand = lambda x: np.cos((k - 1) * x) * f(a * (np.cos(x) + 1))
        x = np.linspace(0, np.pi, quad_points)
        y = Integrand(x)
        c[k - 1] = 2 / np.pi * np.trapz(y, x)
    return c

def get_operator(L, DFilters, n, s, J, Lev):
    r = len(DFilters)
    c = [None] * r
    for j in range(r):
        c[j] = ChebyshevApprox(DFilters[j], n)
    a = np.pi / 2
    FD1 = sparse.identity(L.shape[0])
    d = dict()
    for l in range(1, Lev + 1):
        for j in range(r):
            T0F = FD1
            T1F = ((s ** (-J + l - 1) / a) * L) @ T0F - T0F
            d[j, l - 1] = (1 / 2) * c[j][0] * T0F + c[j][1] * T1F
            for k in range(2, n):
                TkF = ((2 / a * s ** (-J + l - 1)) * L) @ T1F - 2 * T1F - T0F
                T0F = T1F
                T1F = TkF
                d[j, l - 1] += c[j][k] * TkF
        FD1 = d[0, l - 1]
    return d

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_nodes, r, Lev, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        self.in_features = 2 * in_features if self.variant else in_features
        self.crop_len = (Lev - 1) * num_nodes
        self.Lev = Lev
        self.r = r
        self.out_features = out_features
        self.residual = residual
        self.num_nodes = num_nodes
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()
       

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, d_list, h0, lamda, alpha, l, gamma):
        device = next(self.parameters()).device
        adj = adj.to(torch.float32)
        theta = math.log(lamda / l + 1)
        # 分解
        x = torch.rand(self.Lev * self.r * self.num_nodes, 1).to(device) * torch.cat(d_list, dim=0).to_dense()
        # 重构
        x = torch.sparse.mm(torch.cat(d_list[self.Lev - 1:], dim=1), x[self.crop_len:, :])
        hi = gamma * torch.matmul(x, input) + (1 - gamma) * torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output + input
        return output

class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, num_nodes, r, Lev, nclass, dropout, lamda, alpha, gamma, variant, T, use_prop):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, num_nodes, r, Lev, variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.gamma = gamma
        self.T = T
        self.use_prop = use_prop
        
    def compute_energy(self, logits):
        energy = - self.T * torch.logsumexp(logits / self.T, dim=-1)
        return energy

    def propagate_energy(self, energy, adj, prop_layers=4, alpha=0.5):
        """SEP"""

        device = energy.device
        adj = adj.to(device)
        
        ne = adj.shape[1]
        nv = adj.shape[0]
        
        W = torch.ones(ne, dtype=torch.float32, device=device)  
        
        adj_dense = adj.to_dense()  
        
        D_v = torch.sum(adj_dense * W, dim=1)  
        D_e = torch.sum(adj_dense, dim=0)  
        
        D_v = torch.diag(D_v)  
        D_e = torch.diag(D_e) 
        energy = energy.unsqueeze(1)
       
        for _ in range(prop_layers):
            energy = alpha * energy + (1 - alpha) * torch.spmm(D_v.inverse(), torch.spmm(adj_dense, torch.spmm(D_e.inverse(), torch.spmm(adj_dense.T, energy))))
    
        return energy.squeeze(1)  

    def forward(self, x, adj, d_list):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(
                con(layer_inner, adj, d_list, _layers[0], self.lamda, self.alpha, i + 1, self.gamma))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        logits = self.fcs[-1](layer_inner)
        
        energy = self.compute_energy(logits)
        if self.use_prop:
            energy = self.propagate_energy(energy, adj)
        
        return F.log_softmax(logits, dim=1), energy

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = {}

    def add_result(self, run, result):
        if run not in self.results:
            self.results[run] = []
        assert len(result) == 3
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = [100 * torch.tensor(self.results[r]) for r in self.results]
            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))
            best_result = torch.tensor(best_results)
            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
            return best_result[:, 1], best_result[:, 3]

@torch.no_grad()
def evaluate(model, dataset_ind, dataset_ood_te, d_list, split_idx, eval_func):
    """
    评估模型在 IND 数据集和 OOD 数据集上的性能
    """
    model.eval()
    
    device = next(model.parameters()).device 
    dataset_ood_te.x = dataset_ood_te.x.to(device) 
    
    logits_ind, energy_ind = model(dataset_ind.x, adj, d_list)
    
    train_acc = eval_func(dataset_ind.y[split_idx['train']], logits_ind[split_idx['train']])
    valid_acc = eval_func(dataset_ind.y[split_idx['valid']], logits_ind[split_idx['valid']])
    test_acc = eval_func(dataset_ind.y[split_idx['test']], logits_ind[split_idx['test']])

    logits_ood, energy_ood = model(dataset_ood_te.x, adj, d_list)
    
    # 合并 IND 和 OOD 数据集的能量分数
    energy_all = torch.cat([energy_ind, energy_ood], dim=0)
    labels_all = torch.cat([
        torch.zeros_like(energy_ind), 
        torch.ones_like(energy_ood)    
    ], dim=0)
    
    auroc = roc_auc_score(labels_all.cpu().numpy(), energy_all.cpu().numpy())
    aupr = average_precision_score(labels_all.cpu().numpy(), energy_all.cpu().numpy())
    
    return train_acc, valid_acc, test_acc, auroc, aupr, energy_ind, energy_ood

def eval_acc(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()
    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    return float(np.sum(correct)) / len(correct)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    torch.cuda.current_device()
    torch.cuda._initialized = True
    
def loss_compute(model, dataset_ind, dataset_ood, criterion, device, args, split_idx, m_criterion = None, epoch=None):
    """计算损失函数，包括节点分类损失和能量正则化损失"""
    dataset_ind.x = dataset_ind.x.to(device)
    dataset_ood.x = dataset_ood.x.to(device)

    
    train_idx = split_idx['train']  
    ood_idx = dataset_ood.node_idx  
    
    logits_ind, energy_ind = model(dataset_ind.x, adj, d_list)
    sup_loss = criterion(logits_ind[train_idx], dataset_ind.y[train_idx])
    
    
    # Energy reg
    if args.use_reg:
        logits_ood, energy_ood = model(dataset_ood.x, adj, d_list)

        energy_ind = energy_ind[train_idx]
        energy_ood = energy_ood[ood_idx]
        
        # 截断以保证长度一致
        min_n = min(energy_ind.shape[0], energy_ood.shape[0])
        energy_ind = energy_ind[:min_n]
        energy_ood = energy_ood[:min_n]
        
        # 正则化损失
        # reg_loss = torch.mean(F.relu(energy_ind - args.m_in) ** 2 + F.relu(args.m_out - energy_ood) ** 2)
        reg_loss = torch.mean(torch.sigmoid(energy_ood - energy_ind))
        
        loss = sup_loss + args.h_reg * reg_loss
    else:
        loss = sup_loss
        
    # print(f"[DEBUG] Energy IND mean: {energy_ind.mean().item():.4f}, OOD mean: {energy_ood.mean().item():.4f}")
    
    if args.use_UB:
            if args.use_reg:
                mloss_in, _ = m_criterion(_features = logits_ind[train_idx],  labels = dataset_ind.y[train_idx], _features_out = logits_ood, epoch = epoch)
            else:
                mloss_in, _ = m_criterion(_features = logits_ind[train_idx],  labels = dataset_ind.y[train_idx], epoch = epoch)
            loss = loss +  args.lamda2 * mloss_in  
    


    return loss

if __name__ == '__main__':
    args = utils.parse_args()
    existing_dataset = ['20newsW100', 'ModelNet40', 'zoo',
                       'NTU2012', 'Mushroom',
                       'coauthor_cora', 'coauthor_dblp',
                       'cofriend_pokec', 'cocreate_twitch', 'cooccurence_actor', 'copurchasing_amazon',
                       'yelp', 'amazon-reviews', 'walmart-trips', 'house-committees',
                       'walmart-trips-100', 'house-committees-100',
                       'cora', 'citeseer', 'pubmed', 'senate-committees-100', 'congress-bills-100']

    synthetic_list = ['amazon-reviews', 'walmart-trips', 'house-committees', 'walmart-trips-100',
                      'house-committees-100', 'senate-committees-100', 'congress-bills-100']
    
    if args.cuda in [0,1,2,3,4,5,6,7]:
        device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
            
    if args.dname in existing_dataset:
        dname = args.dname
        f_noise = args.feature_noise
        ood_type = args.ood_type
        if (f_noise is not None) and dname in synthetic_list:
            p2raw = './data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname,
                                         feature_noise=f_noise,
                                         p2raw=p2raw,
                                         ood_noise_level=0.1,
                                         control_noise=False)
        else:
            if dname in ['cora', 'citeseer', 'pubmed']:
                p2raw = './data/AllSet_all_raw_data/cocitation/'
            elif dname in ['coauthor_cora', 'coauthor_dblp']:
                p2raw = './data/AllSet_all_raw_data/coauthorship/'
            elif dname in ['cofriend_pokec']:
                p2raw = './data/AllSet_all_raw_data/cofriendship/'
            elif dname in ['cocreate_twitch']:
                p2raw = './data/AllSet_all_raw_data/cocreate/'
            elif dname in ['cooccurence_actor']:
                p2raw = './data/AllSet_all_raw_data/cooccurence/'
            elif dname in ['copurchasing_amazon']:
                p2raw = './data/AllSet_all_raw_data/copurchasing/'
            elif dname in ['yelp']:
                p2raw = './data/AllSet_all_raw_data/yelp/'
            else:
                p2raw = './data/AllSet_all_raw_data/'
            # dataset = dataset_Hypergraph(name=dname, root='./data/pyg_data/hypergraph_dataset_updated/',
            #                              p2raw=p2raw)
            dataset = dataset_Hypergraph(root='./data/pyg_data/hypergraph_dataset_updated/', 
                                name=args.dname, 
                                p2raw=p2raw,
                                ood_noise_level=0.1,
                                control_noise=False,
                                ood_type=ood_type)
            # 获取IND和OOD数据集
        dataset_ind, dataset_ood_tr, dataset_ood_te = dataset.get_ood_datasets(ood_type)
        
        # graph_data = convert_hypergraph_to_graph(dataset_ind)
        
        # pdb.set_trace()
        
        data = dataset_ind
        num_features = len(dataset_ind.x[1])
        # data = dataset.data
        # num_features = dataset.num_features
        n_cls = dataset.num_classes
        if args.dname in ['yelp', 'walmart-trips', 'senate-committees', 'house-committees', 'walmart-trips-100', 'house-committees-100']:
            args.num_classes = len(data.y.unique())
            data.y = data.y - data.y.min()
        if not hasattr(data, 'n_x'):
            data.n_x = torch.tensor([data.x.shape[0]])
        if not hasattr(data, 'num_hyperedges'):
            data.num_hyperedges = torch.tensor(
                [data.edge_index[0].max()-data.n_x[0]+1])
        

    data = ExtractV2E(data)
    setup_seed(args.seed)
    x = data.x

    if args.add_self_loop:
        data = Add_Self_Loops(data)
        print("Added self-loop")

    else:
        print("Nope")
    print(ood_type)
    
    
    H = ConstructH(data)
    num_nodes = H.shape[0]
    L, adj = compute_L(H)
    L = sparse.coo_matrix(L, shape=(num_nodes, num_nodes))
    lobpcg_init = np.random.rand(num_nodes, 1)
    lambda_max, _ = lobpcg(L, lobpcg_init)
    lambda_max = lambda_max[0]

    FrameType = args.FrameType
    if FrameType == 'Haar':
        D1 = lambda x: np.cos(x / 2)
        D2 = lambda x: np.sin(x / 2)
        DFilters = [D1, D2]
        RFilters = [D1, D2]
    elif FrameType == 'Linear':
        D1 = lambda x: np.square(np.cos(x / 2))
        D2 = lambda x: np.sin(x) / np.sqrt(2)
        D3 = lambda x: np.square(np.sin(x / 2))
        DFilters = [D1, D2, D3]
        RFilters = [D1, D2, D3]
    elif FrameType == 'Quadratic':
        D1 = lambda x: np.cos(x / 2) ** 3
        D2 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2)), np.cos(x / 2) ** 2)
        D3 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2) ** 2), np.cos(x / 2))
        D4 = lambda x: np.sin(x / 2) ** 3
        DFilters = [D1, D2, D3, D4]
        RFilters = [D1, D2, D3, D4]
    else:
        raise Exception('Invalid FrameType')

    Lev = args.Lev
    s = args.s
    n = args.n
    J = np.log(lambda_max / np.pi) / np.log(s) + Lev - 1
    r = len(DFilters)
    d = get_operator(L, DFilters, n, s, J, Lev)
    d_list = list()
    for i in range(r):
        for l in range(Lev):
            d_list.append(scipy_to_torch_sparse(d[i, l]).to(device))

    learning_rate = args.lr
    weight_decay = args.wd
    nhid = args.nhid

    split_idx_lst = []
    for run in range(args.runs):
        split_idx = rand_train_test_idx(
            data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
        split_idx_lst.append(split_idx)



    logger = Logger(args.runs, args)
    criterion = nn.NLLLoss()
    eval_func = eval_acc

    train_acc_tensor = torch.zeros((args.runs, args.epochs))
    val_acc_tensor = torch.zeros((args.runs, args.epochs))
    test_acc_tensor = torch.zeros((args.runs, args.epochs))
    auroc_tensor = torch.zeros((args.runs, args.epochs))
    aupr_tensor = torch.zeros((args.runs, args.epochs))

    class EarlyStopping:
        def __init__(self, patience=5, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_val_loss = None
            self.early_stop = False

        def __call__(self, val_loss):
            if self.best_val_loss is None:
                self.best_val_loss = val_loss
            elif val_loss > self.best_val_loss - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_val_loss = val_loss
                self.counter = 0
                
    for run in tqdm(range(args.runs)):
        setup_seed(run)
        split_idx = split_idx_lst[run]
        train_idx = split_idx['train'].to(device)

        model = GCNII(nfeat=num_features,
                    nlayers=args.All_num_layers,
                    nhidden=args.nhid,
                    num_nodes=num_nodes,
                    r=r,
                    Lev=Lev,
                    nclass=n_cls,
                    dropout=args.dropout,
                    lamda=args.lamda,
                    alpha=args.alpha,
                    gamma=args.gamma,
                    variant=args.variant,
                    T=args.T,
                    use_prop=args.use_prop).to(device)

        L, adj = compute_L(H)

        x = x.to(device)

        adj = torch.from_numpy(adj).to(torch.float32).to_sparse().to(device)  
        d_list = [d.to(device) for d in d_list]
        
        m_criterion = UB_loss(args.lamda1)
        
        model = model.to(device)
        data = data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        early_stopping = EarlyStopping(patience=200, min_delta=1e-4)
        best_val = float('-inf')

        model.train()
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()

            loss = loss_compute(model, dataset_ind, dataset_ood_tr, criterion, device, args, split_idx, m_criterion=m_criterion, epoch=epoch)
            
            loss.backward()
            optimizer.step()

            result = evaluate(model, dataset_ind, dataset_ood_te, d_list, split_idx, eval_func)
            
            energy_ind_all = []
            energy_ood_all = []

            logger.add_result(run, result[:3])

            train_acc_tensor[run, epoch] = result[0]
            val_acc_tensor[run, epoch] = result[1]
            test_acc_tensor[run, epoch] = result[2]
            auroc_tensor[run, epoch] = result[3]
            aupr_tensor[run, epoch] = result[4]

            if epoch % args.display_step == 0 and args.display_step > 0:
                print(f'Epoch: {epoch:02d}, '
                    f'Train Acc: {100 * result[0]:.2f}%, '
                    f'Valid Acc: {100 * result[1]:.2f}%, '
                    f'Test Acc: {100 * result[2]:.2f}%, '
                    f'AUROC: {result[3]:.4f}, '
                    f'AUPR: {result[4]:.4f}')
                
            if epoch == args.epochs - 1 and run == 0:
                energy_ind_all = result[5]
                energy_ood_all = result[6]

            if (epoch + 1) % 100 == 0:
                torch.cuda.empty_cache()
            # early_stopping(result[4])
            # if early_stopping.early_stop:
            #     print("Early stopping triggered")
            #     break

    ### Save results ###
    best_val, best_test = logger.print_statistics()
    res_root = 'results'
    if not osp.isdir(res_root):
        os.makedirs(res_root)
    res_root = '{}/layer_{}'.format(res_root, args.All_num_layers)
    if not osp.isdir(res_root):
        os.makedirs(res_root)

    # 保存节点分类结果
    filename = f'{res_root}/{args.dname}_noise_{args.feature_noise}.csv'
    print(f"Saving node classification results to {filename}")
    with open(filename, 'a+') as write_obj:
        cur_line = f'best_val: {best_val.mean():.3f} ± {best_val.std():.3f}\n'
        cur_line += f'best_test: {best_test.mean():.3f} ± {best_test.std():.3f}\n'
        cur_line += f'\n'
        write_obj.write(cur_line)

    # 保存 OOD 检测结果
    ood_filename = f'{res_root}/{args.dname}_noise_{args.feature_noise}_ood.csv'
    print(f"Saving OOD detection results to {ood_filename}")
    with open(ood_filename, 'a+') as write_obj:
        best_auroc = auroc_tensor.max().item()  
        best_aupr = aupr_tensor.max().item()   
        cur_line = f'Best AUROC: {best_auroc:.4f}, Best AUPR: {best_aupr:.4f}\n'
        write_obj.write(cur_line)

    all_args_file = f'{res_root}/all_args_{args.dname}_noise_{args.feature_noise}.csv'
    with open(all_args_file, 'a+') as f:
        f.write(str(args))
        f.write('\n')

    res_root_2 = './storage'
    if not osp.isdir(res_root_2):
        os.makedirs(res_root_2)
    filename = f'{res_root_2}/{args.dname}_{args.feature_noise}_noise.pickle'
    data = {
        'train_acc_tensor': train_acc_tensor,
        'val_acc_tensor': val_acc_tensor,
        'test_acc_tensor': test_acc_tensor,
        'auroc_tensor': auroc_tensor,
        'aupr_tensor': aupr_tensor,
    }
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=4)

    

    if len(energy_ind_all) > 0 and len(energy_ood_all) > 0:
        plt.figure(figsize=(8, 6))
        
        energy_ind_all = energy_ind_all.cpu().numpy()
        energy_ood_all = energy_ood_all.cpu().numpy()

        sns.kdeplot(energy_ind_all, label='In-Distribution (IND)', fill=True, color='skyblue', linewidth=2)
        sns.kdeplot(energy_ood_all, label='Out-of-Distribution (OOD)', fill=True, color='salmon', linewidth=2)
        plt.title('Energy Distribution: IND vs OOD')
        plt.title('Energy Distribution: IND vs OOD')
        plt.xlabel('Energy Score')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()

        energy_fig_path = f'{res_root}/energy_distribution_{args.dname}_noise_{args.feature_noise}.png'
        print(f"Saving energy distribution plot to {energy_fig_path}")
        plt.savefig(energy_fig_path)
        plt.close()

    
    print('All done! Exit python code')
    quit()
