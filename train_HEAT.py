import warnings
warnings.filterwarnings('ignore')

import argparse
import time
import copy
import os

import numpy as np
import pandas as pd 
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import svds

import torch
from input_data import load_data
from preprocessing import general_train_test_split_edges,biased_train_test_split_edges, bidirectional_train_test_split_edges
from preprocessing import celltypespecific_train_test_split_edges, fake_train_test_split_edges #, dropout_train_test_split_edges
from layers import DirectedGCNConvEncoder, DirectedInnerProductDecoder
from layers import SingleLayerDirectedGCNConvEncoder
from layers import Heterogeneous_DirectedGCNConvEncoder
from models import DirectedGAE

from layers import InnerProductDecoder
from layers import DummyEncoder, DummyPairEncoder

from models import GAE
from sklearn.decomposition import PCA

import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
# from utils import sparse2tuple, plot_histogram,write_csv_matrix,connection_number_between_groups,generate_adj_new_long_edges,edges_enrichment_evaluation,randAdj_long_edges
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--seed',
                    nargs= '?',
                    default=1,
                    type=int)
parser.add_argument('--test_ratio',
                    nargs= '?',
                    default=0.1,
                    type=float)
parser.add_argument('--noise',
                    nargs= '?',
                    default=False,
                    type=bool)
parser.add_argument('--noise_disp',
                    nargs= '?',
                    default=2,
                    type=float)
parser.add_argument('--dropout_gene',
                    nargs= '?',
                    default=False,
                    type=bool)
parser.add_argument('--dropout_gene_ratio',
                    nargs= '?',
                    default=0.1,
                    type=float)
parser.add_argument('--dropout_value',
                    nargs= '?',
                    default=False,
                    type=bool)
parser.add_argument('--dropout_value_ratio',
                    nargs= '?',
                    default=0.1,
                    type=float)
parser.add_argument('--enrichment',
                    nargs= '?',
                    default=False,
                    type=bool)
parser.add_argument('--sensitivity',
                    nargs= '?',
                    default=False,
                    type=bool)
parser.add_argument('--sensitivity_long',
                    nargs= '?',
                    default=False,
                    type=bool)
parser.add_argument('--sensitivity_celltype_LR',
                    nargs= '?',
                    default=False,
                    type=bool)
parser.add_argument('--visualize',
                    nargs= '?',
                    default=False,
                    type=bool)


parser.add_argument('--dataset',
                    nargs= '?',
                    default='cora_ml',
                    type=str)
parser.add_argument('--celltype_nosie',
                    nargs= '?',
                    default='cora_ml',
                    type=str)
parser.add_argument('--celltype_nosie_ratio',
                    nargs= '?',
                    default=0.1,
                    type=float)
                    
parser.add_argument('--task',
                    nargs= '?',
                    default='task_1',
                    type=str)
parser.add_argument('--model',
                    nargs= '?',
                    default='digae',
                    type=str)
parser.add_argument('--learning_rate',
                    nargs= '?',
                    default=0.01,
                    type=float)
parser.add_argument('--epochs',
                    nargs= '?',
                    default=200,
                    type=int)
parser.add_argument('--hidden',
                    nargs= '?',
                    default=64,
                    type=int)
parser.add_argument('--dimension',
                    nargs= '?',
                    default=32,
                    type=int)
parser.add_argument('--alpha',
                    nargs= '?',
                    default=1.0,
                    type=float)
parser.add_argument('--beta',
                    nargs= '?',
                    default=0.0,
                    type=float)
parser.add_argument('--nb_run',
                    nargs= '?',
                    default=1,
                    type=int)
parser.add_argument('--prop_val',
                    nargs= '?',
                    default=5.0,
                    type=float)
parser.add_argument('--prop_test',
                    nargs= '?',
                    default=10.0,
                    type=float)
parser.add_argument('--verbose',
                    nargs= '?',
                    default=True,
                    type=bool)

parser.add_argument('--self_loops',
                    nargs= '?',
                    default=True,
                    type=bool)
parser.add_argument('--adaptive',
                    nargs= '?',
                    default=False,
                    type=bool)
parser.add_argument('--feature_vector_type',
                    nargs= '?',
                    const=None)
parser.add_argument('--feature_vector_size',
                    nargs= '?',
                    const=None,
                    type=int)
parser.add_argument('--directed',
                    nargs= '?',
                    default=True,
                    type=bool)
parser.add_argument('--logfile',
                    nargs='?',
                    default='logs.json',
                    type=str)
parser.add_argument('--validate',
                    nargs='?',
                    default=False,
                    type=bool)


args = parser.parse_args()


def train_single():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    loss.backward()
    optimizer.step()
    return float(loss)


def test_single(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)

def dummy_train_single():
    model.train()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    return float(loss)

def dummy_test_single(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)
    

def train_pair():
    model.train()
    optimizer.zero_grad()
    s, t = model.encode(u, v, train_pos_edge_index)
    # print('s: ',s.size()) #[2995, 32]
    # print('t: ',t.size()) #[2995, 32]
    loss = model.recon_loss(s, t, train_pos_edge_index)
    ################ add by huo
    # print(data)
    # Data(x=[2995, 2879], edge_weight=[8416], val_pos_edge_index=[2, 420], 
    # test_pos_edge_index=[2, 841], train_pos_edge_index=[2, 7155], train_neg_adj_mask=[2995, 2995], 
    # val_neg_edge_index=[2, 420], test_neg_edge_index=[2, 841], u=[2995, 2879], v=[2995, 2879])
    ################ 
    loss.backward()
    optimizer.step()
    return float(loss)

def test_pair(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        s, t = model.encode(u, v, train_pos_edge_index)
        # print('s: ',s)
        # print('t: ',t)
    return model.test(s, t, pos_edge_index, neg_edge_index)

def single_gene_occlusion(u_generate, v_generate, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        s, t = model.encode(u_generate, v_generate, train_pos_edge_index)
        # print('s: ',s)
        # print('t: ',t)
    return model.test(s, t, pos_edge_index, neg_edge_index)

def single_gene_occlusion_positive(u_generate, v_generate, pos_edge_index):
    model.eval()
    with torch.no_grad():
        s, t = model.encode(u_generate, v_generate, train_pos_edge_index)
        # print('s: ',s)
        # print('t: ',t)
    return model.test_posotive(s, t, pos_edge_index)

def dummy_train_pair():
    model.train()
    s, t = model.encode(u, v, train_pos_edge_index)
    loss = model.recon_loss(s, t, train_pos_edge_index)
    return float(loss)

def dummy_test_pair(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        s, t = model.encode(u, v, train_pos_edge_index)
    return model.test(s, t, pos_edge_index, neg_edge_index)


def svd_features(data, k):
    num_nodes = data.x.size(0)
    num_edges = data.edge_weight.size(0)
    
    indices  = data.train_pos_edge_index.clone().numpy()
    row, col = indices[0], indices[1]
    values   = np.ones(indices.shape[1])
    num_rows    = num_nodes
    num_columns = num_nodes

    adjacency_matrix = csc_matrix(coo_matrix((values, (row, col)), 
                                             shape=(num_rows, num_columns),
                                             dtype=float))


    u, s, vt        = svds(adjacency_matrix, k)
    sorting_indices = np.argsort(s)[::-1]
    s  = s[sorting_indices]
    u  = u[:, sorting_indices]
    vt = vt[sorting_indices, :]

    sqrt_s = np.sqrt(s)
    diag_sqrt_s = np.diag(sqrt_s) 
    u_hat       = np.dot(u, diag_sqrt_s)
    vt_hat      = np.dot(diag_sqrt_s, vt)
    v_hat       = vt_hat.T

    u_hat = torch.tensor(u_hat).float()
    v_hat = torch.tensor(v_hat).float()
    return u_hat, v_hat
    

def svd_randomized_features(data, k):
    num_nodes = data.x.size(0)
    num_edges = data.edge_weight.size(0)

    indices = data.train_pos_edge_index.clone()
    values  = torch.ones(indices.size(1))
    rows    = num_nodes
    columns = num_nodes

    adjacency_tensor = torch.sparse_coo_tensor(indices, values, (rows, columns))
    u, s, v = torch.svd_lowrank(adjacency_tensor, k)
    vh      = v.t()
    
    sqrt_s      = torch.sqrt(s)
    diag_sqrt_s = torch.diag(sqrt_s) 
    u_hat       = torch.matmul(u, diag_sqrt_s)
    vh_hat      = torch.matmul(diag_sqrt_s, vh)
    v_hat       = vh_hat.t()    
    return u_hat, v_hat


def random_features(data, k):
    num_nodes = data.x.size(0)
    u_hat = torch.rand(num_nodes, k)
    v_hat = torch.rand(num_nodes, k)
    return u_hat, v_hat

def normal_features(data, k):
    num_nodes = data.x.size(0)
    u_hat = torch.randn(num_nodes, k)
    v_hat = torch.randn(num_nodes, k)
    return u_hat, v_hat
    
def identity_features(data, k=None):
    num_nodes = data.x.size(0)
    x = torch.eye(num_nodes)
    u_hat = x.clone()
    v_hat = x.clone()
    return u_hat, v_hat


def random_ones_features(data, k):
     num_nodes = data.x.size(0)

     u_hat = torch.zeros(num_nodes, k)
     u_idx = [np.random.randint(0, k-1) for _ in range(num_nodes)]
     for i in range(num_nodes):
        u_hat[i, u_idx[i]] = 1.0

     v_hat = torch.zeros(num_nodes, k)
     v_idx = [np.random.randint(0, k-1) for _ in range(num_nodes)]
     for i in range(num_nodes):
        v_hat[i, v_idx[i]] = 1.0

     return u_hat, v_hat

def read_LR(LR_df):    
    interaction = LR_df['interaction_name'].values
    Ligand_list = []
    Receptor_list = []
    for interaction_i in interaction:
        interaction_i = interaction_i.split('_')
        # Ligand_list.append(interaction_i[0])
        # Receptor_list.append(interaction_i[1:]) 
        Ligand_list.append(interaction_i[0])
        Receptor_list.extend(interaction_i[1:])        
    return Ligand_list, Receptor_list


dummy_run = False
 
seed = 1 
seed = args.seed
np.random.seed(seed)
# # # random.seed(seed)
torch.manual_seed(seed) #cpu
# torch.cuda.manual_seed_all(seed)  #并行gpu
# torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致
# torch.backends.cudnn.benchmark = True   #训练集变化不大时使训练加速


if args.verbose:
    print("Loading data...")

#############################################################################################################
# python train.py --dataset=cora_ml --model=digae --alpha=0.0 --beta=0.2 --epochs=200 --nb_run=5 
# --logfile=digae_cora_ml_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True

# python train.py --dataset=HBC --model=digae --alpha=0.0 --beta=0.2 --epochs=200 --nb_run=5 
# --logfile=digae_cora_HBC_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True

# python train.py --dataset=MERFISH --model=digae --alpha=0.0 --beta=0.2 --epochs=200 --nb_run=5 
# --logfile=digae_merfish_ml_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True

# python train.py --dataset=HBC --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_cora_HBC_grid_search.json 
# --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --feature_vector_type=LR

# Ligand genes as u, receptor genes as v , set dimension of s and t the same with input
# python train.py --dataset=HBC --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_cora_HBC_grid_search.json 
# --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --feature_vector_type=LR

device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_name    = args.dataset
out_channels    = args.dimension
hidden_channels = args.hidden
print('dataset_name: ', dataset_name)

feature_vector_type =  args.feature_vector_type
if feature_vector_type == 'svd':
    compute_features = svd_features
elif feature_vector_type == 'svd_randomized':
    compute_features = svd_randomized_features
elif feature_vector_type == 'random':
    compute_features = random_features
elif feature_vector_type == 'normal':
    compute_features = normal_features
elif feature_vector_type == 'identity':
    compute_features = identity_features
elif feature_vector_type == 'random_ones':
    compute_features = random_ones_features
    
feature_vector_size  = args.feature_vector_size

if args.model in ['gae',
                  'source_target']:
    train_func = train_single
    test_func  = test_single
elif args.model in ['dummy']:
    train_func = dummy_train_single
    test_func  = dummy_test_single
elif args.model in ['dummy_pair']:
    train_func = dummy_train_pair
    test_func  = dummy_test_pair
else:
    train_func = train_pair
    test_func  = test_pair
    

val_ratio  = args.prop_val / 100. # 0.05 
# test_ratio = args.prop_test / 100. # 0.1
test_ratio = args.test_ratio

print('directed: ',args.directed)
loaded_data     = load_data(dataset_name, directed=args.directed)
print('loaded_data: ')
print(loaded_data)

mean_roc  = []
mean_ap   = []
mean_time = []

###### add noise to gene expression
if args.noise:
    print('add noise to gene expression....')
    exp_noise = loaded_data.x #(1981, 5595)
    for gene_index in range(np.shape(loaded_data.x)[1]):
        rand= np.random.normal(0,args.noise_disp) # 2, 2.7, 3
        exp_noise[:,gene_index] = loaded_data.x[:,gene_index] * np.power(2, rand)
    loaded_data.x = exp_noise

##### different ratio of genes were randomly removed
if args.dropout_gene:
    print('different ratio of genes were randomly removed....')
    exp_dropout = loaded_data.x #(1981, 5595)
    sample_size = loaded_data.x.size(1) - int(loaded_data.x.size(1)*args.dropout_gene_ratio)
    rs = random.sample(range(0,loaded_data.x.size(1)),sample_size)
    rs.sort()
    loaded_data.x = exp_dropout[:,rs]

##### different ratio of the non-zero values were randomly selected and forced to zero
if args.dropout_value:
    print('different ratio of the non-zero values were randomly selected and forced to zero....')
    exp_dropout = loaded_data.x #(1981, 5595)
    exp = exp_dropout.numpy()
    nonzero_index = np.argwhere(exp!=0)
    nonzero_num = np.shape(nonzero_index)[0]
    sample_size = int(nonzero_num*args.dropout_value_ratio)
    rs = random.sample(range(0,nonzero_num),sample_size)
    rs_index = nonzero_index[rs,:]

    for rs_i in rs_index:
        exp_dropout[rs_i[0],rs_i[1]] = 0.
    loaded_data.x = exp_dropout

print('nb_run: ',args.nb_run) # 5 times of split
for i in range(args.nb_run):
    if args.verbose:
        print("Masking test edges...")

    print(args.task)
    #### split dataset
    if args.task == 'task_1':
        data = loaded_data.clone()
        # data.train_mask = data.val_mask = data.test_mask = data.y = None
        data.train_mask = data.val_mask = data.test_mask = None
        data            = general_train_test_split_edges(data,
                                                         val_ratio=val_ratio,
                                                         test_ratio=test_ratio,
                                                         directed=args.directed)
    elif args.task == 'task_2':
        data = loaded_data.clone()
        data.train_mask = data.val_mask = data.test_mask = data.y = None
        data            = biased_train_test_split_edges(data,
                                                        val_ratio=val_ratio,
                                                        test_ratio=test_ratio,
                                                        directed=args.directed)
    elif args.task == 'task_3':
        data = loaded_data.clone()
        data.train_mask = data.val_mask = data.test_mask = data.y = None
        data            = bidirectional_train_test_split_edges(data,
                                                               val_ratio=val_ratio,
                                                               test_ratio=test_ratio,
                                                               directed=args.directed)     
    # specify specific cell type as test edges
    elif args.task == 'celltypespecific':
        data = loaded_data.clone()
        data.train_mask = data.val_mask = data.test_mask = None
        data            = celltypespecific_train_test_split_edges(data,
                                                               val_ratio=val_ratio,
                                                               test_ratio=test_ratio,
                                                               directed=args.directed)  
    # fake
    elif args.task == 'fake':
        test_ratio = 0.1
        data = loaded_data.clone()
        data.train_mask = data.val_mask = data.test_mask = None
        data            = fake_train_test_split_edges(data,
                                                      fake_ratio=test_ratio,
                                                      directed=args.directed)
    # noise
    elif args.task == 'noise':
        data = loaded_data.clone()
        data.train_mask = data.val_mask = data.test_mask = None
        data            = noise_train_test_split_edges(data,
                                                       val_ratio=val_ratio,
                                                       test_ratio=test_ratio,
                                                       directed=args.directed)
    # dropout
    elif args.task == 'dropout':
        data = loaded_data.clone()
        data.train_mask = data.val_mask = data.test_mask = None
        data            = dropout_train_test_split_edges(data,
                                                         val_ratio=val_ratio,
                                                         test_ratio=test_ratio,
                                                         directed=args.directed)
    else:
        raise ValueError('Undefined task!')

    print('original cell types: ',data.y)
    ######## modified by jinxian: add noise to cell type information, 20250616
    if args.celltype_nosie == 'random_flip':
        data_path = 'generated_data/'
        data_name = dataset_name
        data_file = data_path + data_name +'/' + 'cell_type_with_noise/random_flip/' + 'cell_types_random_flip_' + str(args.celltype_nosie_ratio) + '.npy'
        cell_type_indeces = np.load(data_file, allow_pickle=True)
        cell_type = cell_type_indeces.astype(np.int32)
        labels   = torch.from_numpy(cell_type).long()
        data.y = labels

    if args.celltype_nosie == 'random_generate':
        data_path = 'generated_data/'
        data_name = dataset_name
        data_file = data_path + data_name +'/' + 'cell_type_with_noise/random_generate/' + 'cell_type_random_generate_seed42' + '.npy'
        cell_type_indeces = np.load(data_file, allow_pickle=True)
        cell_type = cell_type_indeces.astype(np.int32)
        labels   = torch.from_numpy(cell_type).long()
        data.y = labels
    print('modified cell types', data.y)


    data                 = data.to(device)
    train_pos_edge_index = data.train_pos_edge_index.to(device)
    if args.validate is True:
        test_pos_edge_index  = data.val_pos_edge_index.to(device)
        test_neg_edge_index  = data.val_neg_edge_index.to(device)
    else:
        test_pos_edge_index  = data.test_pos_edge_index.to(device)
        test_neg_edge_index  = data.test_neg_edge_index.to(device)

    if feature_vector_type in ['svd', 'svd_randomized', 'random', 'normal', 'random_ones']:
        in_channels  = feature_vector_size
        u, v = compute_features(data, in_channels)
        data.u = u
        data.v = v
        data.x = torch.cat([data.u, data.v], dim=1)
    elif feature_vector_type in ['identity']:
        in_channels = data.x.size(0)
        u, v = compute_features(data, in_channels)
        data.u = u
        data.v = v
        data.x = torch.cat([data.u, data.v], dim=1)
    # use ligand-receptor genes
    elif feature_vector_type in ['LR']:
        print('feature_vector_type is LR...')
        data.u = data.u_feat
        data.v = data.v_feat       
    else:
        data.u = data.x.clone()
        data.v = data.x.clone() 


    u = data.u.to(device)
    v = data.v.to(device)

    print('u,v: ',u.shape, v.shape)

    ###### modified by jinxian 20250423 
    # x = data.x.to(device)
    data = data.cpu()

    in_channels = data.v.shape[1]

    # if feature_vector_type == 'LR':
    #     out_channels =  in_channels
    print('model:',args.model)

    ##### define the model type
    if args.model == 'digae':
        encoder = DirectedGCNConvEncoder(in_channels, hidden_channels, out_channels, alpha=args.alpha, beta=args.beta,
                                         self_loops=args.self_loops,
                                         adaptive=args.adaptive)
        decoder = DirectedInnerProductDecoder()
        model   = DirectedGAE(encoder, decoder)
        model   = model.to(device)

    #### heterogeneous digae
    elif args.model == 'HEAT_digae':
        print(data.y)
        encoder = Heterogeneous_DirectedGCNConvEncoder(in_channels, hidden_channels, out_channels, data.y, alpha=args.alpha, beta=args.beta,
                                         self_loops=args.self_loops,
                                         adaptive=args.adaptive)
        decoder = DirectedInnerProductDecoder()
        model   = DirectedGAE(encoder, decoder)
        model   = model.to(device)

        
    elif args.model == 'digae_single_layer':
        encoder = SingleLayerDirectedGCNConvEncoder(in_channels, out_channels, alpha=args.alpha, beta=args.beta,
                                                    self_loops=args.self_loops,
                                                    adaptive=args.adaptive)
        decoder = DirectedInnerProductDecoder()
        model   = DirectedGAE(encoder, decoder)
        model   = model.to(device)


    elif args.model == 'dummy':
        encoder = DummyEncoder()
        decoder = InnerProductDecoder()
        model   = GAE(encoder, decoder)
        model   = model.to(device)
        dummy_run = True

    elif args.model == 'dummy_pair':
        encoder = DummyPairEncoder()
        decoder = DirectedInnerProductDecoder()
        model   = DirectedGAE(encoder, decoder)
        model   = model.to(device)
        dummy_run =True
    else:
        raise ValueError('Undefined model!')


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
 
    if args.verbose:
        print("Training...")

    # Flag to compute total running time
    t_start = time.time()
    ####################### start to train model...
    print('epochs: ',args.epochs) #200
    auc_list = []
    ap_list = []
    acc_list = []
    f1_list = []
    epoch_list = []
    for epoch in range(args.epochs):
        # Flag to compute running time for each epoch
        t = time.time()
        loss = train_func()
        avg_cost = loss
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
        "time=", "{:.5f}".format(time.time() - t))

        ############ test dataset 
        # 与 train.py / train_github.py 保持一致，返回 auc, ap, acc, f1 四个指标
        auc, ap, acc, f1, adj_pred, s, t = test_func(test_pos_edge_index, test_neg_edge_index)
        print('auc: ', auc, 'ap: ', ap, 'acc:', acc, 'f1:', f1)
        auc_list.append(auc)
        ap_list.append(ap)
        acc_list.append(acc)
        f1_list.append(f1)
        epoch_list.append(epoch)

    mean_time.append(time.time() - t_start)
    # 保存四个评价指标到 csv
    df_iter = pd.DataFrame({
        'auc': auc_list,
        'auprc': ap_list,
        'acc': acc_list,
        'f1': f1_list,
        'epoch': epoch_list
    })
    if args.noise:
        filename = 'results/AUC/'+dataset_name+'/'+dataset_name+'_noise_'+str(args.noise_disp)+'_iter'+str(i)+'.csv'
    elif args.dropout_gene:
        filename = 'results/AUC/'+dataset_name+'/'+dataset_name+'_dropout_gene_'+str(args.dropout_gene_ratio)+'_iter'+str(i)+'.csv'
    elif args.dropout_value:
        filename = 'results/AUC/'+dataset_name+'/'+dataset_name+'_dropout_value_'+str(args.dropout_value_ratio)+'_iter'+str(i)+'.csv'
    elif args.celltype_nosie:
        if args.celltype_nosie == 'random_flip':
            filename = 'results/AUC/'+dataset_name+'/'+dataset_name+'_celltype_nosie_random_flip_'+str(args.celltype_nosie_ratio)+'_iter'+str(i)+'.csv'
        if args.celltype_nosie == 'random_generate':
            filename = 'results/AUC/'+dataset_name+'/'+dataset_name+'_celltype_nosie_random_generate' +'_iter'+str(i)+'.csv'    
    else:
        filename = 'results/AUC/'+dataset_name+'/'+dataset_name+'_missingratio'+str(test_ratio)+'_iter'+str(i)+'.csv'

    df_iter.to_csv(filename)

    if args.verbose:
        print("Testing model...")
    # auc, ap = test_func(test_pos_edge_index, test_neg_edge_index)
    ############ save model
    model_dir = 'model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"目录 {model_dir} 已创建")

    model_name = 'model/'+'model_iter_'+str(i)+'.pth'
    torch.save(model, model_name)
    # 最终测试同样返回四个指标
    auc, ap, acc, f1, adj_pred, s, t = test_func(test_pos_edge_index, test_neg_edge_index)
    roc_score, ap_score = auc, ap
    mean_roc.append(roc_score)
    mean_ap.append(ap_score)
    print(adj_pred)
    print("AUC score: ", roc_score,
      "\nAP scores : ", ap_score, "\n \n")


# if adaptive...
print('=' * 60)
print(args.adaptive)
print(args.model)
print('=' * 60)
    
# Report final results
print("\nTest results for", args.model,
      "model on", args.dataset, "on", args.task, "\n",
      "___________________________________________________\n")

print("AUC scores\n", mean_roc)
print("Mean AUC score: ", np.mean(mean_roc),
      "\nStd of AUC scores: ", np.std(mean_roc), "\n \n")

print("AP scores \n", mean_ap)
print("Mean AP score: ", np.mean(mean_ap),
      "\nStd of AP scores: ", np.std(mean_ap), "\n \n")

print("Running times\n", mean_time)
print("Mean running time: ", np.mean(mean_time),
      "\nStd of running time: ", np.std(mean_time), "\n \n")
