import warnings
warnings.filterwarnings('ignore')

import argparse
import time
import copy

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
 
seed = 0 #10
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

    x = data.x.to(device)

    in_channels          = data.v.shape[1]

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
        # 与 train.py 保持一致，返回 auc, ap, acc, f1 四个指标
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

    # # ### load best model to do later analysis
    # # model = torch.load('model/model_best.pth')
    # # auc, ap, adj_pred, s, t = test_func(test_pos_edge_index, test_neg_edge_index)
    # # roc_score, ap_score = auc, ap
    # # mean_roc.append(roc_score)
    # # mean_ap.append(ap_score)
    # # print(adj_pred)
    # # print("AUC score: ", roc_score,
    # #   "\nAP scores : ", ap_score, "\n \n")
    # if roc_score > 0.9:
    #     # sender_matrix = s.numpy()
    #     # receiver_matrix = t.numpy()
    #     # df_sender = pd.DataFrame(data=sender_matrix,columns=data.LR_name)
    #     # df_receiver = pd.DataFrame(data=receiver_matrix,columns=data.LR_name)
    #     # df_sender.to_csv('results/matrix/sender_iter_'+str(i)+'.csv')
    #     # df_receiver.to_csv('results/matrix/receiver_iter_'+str(i)+'.csv')
    #     # save_name = 'results/matrix/adj_pred_'+str(i)+'.npy'
    #     # np.save(save_name,adj_pred)


    ################################################################# # 3.3. 连接可视化
    if args.visualize:
        adj = data.adj.todense()
        adj_reconstructed = adj_pred
        adj_reconstructed[adj_reconstructed>=0.5]=1
        adj_reconstructed[adj_reconstructed<0.5]=0
        adj_reconstructed = adj_reconstructed.numpy()

        cell_label_df = pd.read_csv('data/' + dataset_name + '/cell_type.csv')
        cell_label = cell_label_df.values
        coord_df = pd.read_csv('data/' + dataset_name + '/coord.csv')
        coord = coord_df.values

        id_subgraph, _ = ranked_partial(adj, adj_reconstructed, coord, [10,15])  #返回的是[(diff,[id_list]),(diff,[id_list])...]这种形式
                                                                                                    #adj_rec1:[10,15], adj_rec2:[3,5]
        rank = 0
        for item in id_subgraph:
            cell_type_subgraph = cell_label[item[1],:][:,[0,1]]
            cell_type_subgraph[:,0] = np.array(list(range(cell_type_subgraph.shape[0]))) + 1  #需要对X重新生成细胞的id，这里以1开始
            coord_subgraph = coord[item[1],:]
            adj_reconstructed_subgraph = adj_reconstructed[item[1],:][:,item[1]]
            rank += 1
            adjacency_visualization(cell_type_subgraph, coord_subgraph, adj_reconstructed_subgraph, filename='spatial_network_rank'+str(rank)+'_diff'+str('%.3f'%item[0]))

    #################################################################
    if args.enrichment:
        cell_label_df = pd.read_csv('data/' + dataset_name + '/cell_type.csv')
        cell_label = cell_label_df.values
        coord_df = pd.read_csv('data/' + dataset_name + '/coord.csv')
        coord = coord_df.values
        adj = data.adj.todense()
        # threshold
        adj_pred[adj_pred>=0.5]=1
        adj_pred[adj_pred<0.5]=0
        adj_pred = adj_pred.numpy()

        adj_diff = adj - adj_pred
        adj_diff = (adj_diff == -1).astype('int')
        adj_diff = sp.csr_matrix(adj_diff)

        dist_matrix_rongyu = pdist(coord, 'euclidean')
        dist_matrix = squareform(dist_matrix_rongyu)

        new_edges = sparse2tuple(sp.triu(sp.csr_matrix(adj_diff)))[0]
        all_new_edges_dist = dist_matrix[new_edges[:,0].tolist(),new_edges[:,1].tolist()]
        plot_histogram(all_new_edges_dist, xlabel='distance', ylabel='density', filename='all_new_edges_distance', color="coral")
        write_csv_matrix(dist_matrix*adj_diff, 'results/enrichment/'+dataset_name+'/all_new_edges_dist_matrix'+'_iter'+str(i))

        # -over/-under representation of particular types of interactions
        cutoff_distance = np.percentile(all_new_edges_dist,99)
        print('cutoff_distance: ',cutoff_distance)

        connection_number, _ = connection_number_between_groups(adj, cell_label[:,1])
        write_csv_matrix(connection_number, 'results/enrichment/'+dataset_name+'/connection_number_between_groups'+'_iter'+str(i))

        adj_new_long_edges = generate_adj_new_long_edges(dist_matrix, new_edges, all_new_edges_dist, cutoff_distance)
        write_csv_matrix(adj_new_long_edges.todense(), 'results/enrichment/'+dataset_name+'/adj_new_long_edges'+'_iter'+str(i))

        print('------permutations calculating------')
        # cell_type_name = [np.unique(cell_label[cell_label[:,1]==i,2])[0] for i in np.unique(cell_label[:,1])]
        # test_result, _, _, _ = edges_enrichment_evaluation(adj, cell_label[:,1], cell_type_name, edge_type='all edges',N=1000)
        # write_csv_matrix(test_result, 'results/all_edges_enrichment_evaluation_original', colnames=['cell type A','cell type B','average_connectivity','significance'])

        cell_type_name = [np.unique(cell_label[cell_label[:,1]==i,2])[0] for i in np.unique(cell_label[:,1])]
        test_result, _, _, _ = edges_enrichment_evaluation(adj_pred, cell_label[:,1], cell_type_name, edge_type='all edges',N=1000)
        write_csv_matrix(test_result, 'results/enrichment/'+dataset_name+'/all_edges_enrichment_evaluation'+'_iter'+str(i), colnames=['cell type A','cell type B','average_connectivity','significance'])
        test_result, _, _, _ = edges_enrichment_evaluation(adj_new_long_edges.toarray(), cell_label[:,1], cell_type_name, edge_type='long edges', dist_matrix=dist_matrix, cutoff_distance=cutoff_distance,N=1000)
        write_csv_matrix(test_result, 'results/enrichment/'+dataset_name+'/long_edges_enrichment_evaluation'+'_iter'+str(i), colnames=['cell type A','cell type B','connection_number','significance'])

    if args.sensitivity_celltype_LR:
        if dataset_name == 'MERFISH':
            ############## select Microglia-Astrocyte edges as test edges
            adj_reconstructed = adj_pred
            adj_reconstructed[adj_reconstructed>=0.5]=1
            adj_reconstructed[adj_reconstructed<0.5]=0
            adj_reconstructed = adj_reconstructed.numpy()

            adj_reconstructed = sp.csr_matrix(adj_reconstructed)
            coo = adj_reconstructed.tocoo()
            indices = np.vstack((coo.row, coo.col))
            indices = torch.from_numpy(indices).long()

            cell_label_df = pd.read_csv('data/' + dataset_name + '/cell_type.csv')
            cell_label = cell_label_df['Cell_class_name'].values

            cell_index  = np.arange(x.size(0))
            lst_celltype0 = cell_index[np.array(cell_label=='Microglia')].tolist()
            lst_celltype1 = cell_index[np.array(cell_label=='Astrocyte')].tolist()
            temp0 = torch.isin(indices[0,:],torch.tensor(lst_celltype0))
            temp1 = torch.isin(indices[1,:],torch.tensor(lst_celltype1))
            test = temp0 & temp1
            test_pos_edge_index = indices[:,test]

            ############## start to compute 
            roc_score_orig, ap_score_orig, adj_pred_orig, _, _ = single_gene_occlusion(u, v, test_pos_edge_index, test_neg_edge_index)

            CellChatDB_LR = np.load('data/MERFISH/CellChatDB_LR.npy')

            df_info = pd.read_csv('generated_data/MERFISH/counts.csv')
            genes = df_info.columns.values
            mapping = {}
            for i in range(len(genes)):
                mapping[genes[i]] = i 

            # Calculate the test score for each LR in a loop
            single_gene_roc_score = dict()
            single_gene_ap_score = dict()
            for k in range(0,len(CellChatDB_LR)):
                ligand = CellChatDB_LR[k].split('-')[0]
                receptor = CellChatDB_LR[k].split('-')[1]
                ligand_index = mapping[ligand]
                receptor_index = mapping[receptor]

                col_all_roc_score = []
                col_all_ap_score = []
                for j in range(30): #30
                    u_occlu = copy.deepcopy(u)
                    v_occlu = copy.deepcopy(v)
                    np.random.shuffle(u_occlu[:,ligand_index])
                    np.random.shuffle(v_occlu[:,receptor_index])
                    roc_score, ap_score, _, _, _  = single_gene_occlusion(u_occlu, v_occlu, test_pos_edge_index, test_neg_edge_index)
                    col_all_roc_score.append(roc_score)
                    col_all_ap_score.append(ap_score)
                    del u_occlu, v_occlu
                gene_name = CellChatDB_LR[k] 
                print(gene_name)
                single_gene_roc_score.update({gene_name: col_all_roc_score})
                single_gene_ap_score.update({gene_name: col_all_ap_score})

            occlu_roc = {}
            occlu_ap = {}
            for k,v in single_gene_roc_score.items():
                occlu_roc[k] = np.mean(np.array(v))

            for k,v in single_gene_ap_score.items():
                occlu_ap[k] = np.mean(np.array(v))

            # Get gene sensitivity
            occlu_deta_ap = {}
            occlu_deta_roc = {}
            for k,v in occlu_ap.items():
                occlu_deta_ap[k] = np.abs(float(ap_score_orig) - np.array(v).mean())
            for k,v in occlu_roc.items():
                occlu_deta_roc[k] = np.abs(float(roc_score_orig) - np.array(v).mean())

            ## sort genes
            genes_list = occlu_deta_ap.keys()
            delta_ap = occlu_deta_ap.values()
            pd_delta_ap = pd.DataFrame({'genes_list':genes_list,'delta_ap':delta_ap})
            pd_delta_ap = pd_delta_ap.sort_values('delta_ap',ascending=False)
            file_name = 'results/deltaAUC/'+dataset_name+'_delta_ap_celltypeLR_iter'+str(i)+'.csv'
            pd_delta_ap.to_csv(file_name)

            genes_list = occlu_deta_roc.keys()
            delta_roc = occlu_deta_roc.values()
            pd_delta_roc = pd.DataFrame({'genes_list':genes_list,'delta_roc':delta_roc})
            pd_delta_roc = pd_delta_roc.sort_values('delta_roc',ascending=False)
            file_name = 'results/deltaAUC/'+dataset_name+'_delta_roc_celltypeLR_iter'+str(i)+'.csv'
            pd_delta_roc.to_csv(file_name)


    if args.sensitivity_long:
        cell_label_df = pd.read_csv('data/' + dataset_name + '/cell_type.csv')
        cell_label = cell_label_df.values
        coord_df = pd.read_csv('data/' + dataset_name + '/coord.csv')
        coord = coord_df.values
        adj = data.adj.todense()
        # threshold
        adj_pred[adj_pred>=0.5]=1
        adj_pred[adj_pred<0.5]=0
        adj_pred = adj_pred.numpy()

        adj_diff = adj - adj_pred
        adj_diff = (adj_diff == -1).astype('int')
        adj_diff = sp.csr_matrix(adj_diff)

        dist_matrix_rongyu = pdist(coord, 'euclidean')
        dist_matrix = squareform(dist_matrix_rongyu)

        new_edges = sparse2tuple(sp.triu(sp.csr_matrix(adj_diff)))[0]
        all_new_edges_dist = dist_matrix[new_edges[:,0].tolist(),new_edges[:,1].tolist()]

        cutoff_ratio = 90
        cutoff_distance = np.percentile(all_new_edges_dist,cutoff_ratio)
        print('cutoff_distance: ',cutoff_distance)

        adj_new_long_edges = generate_adj_new_long_edges(dist_matrix, new_edges, all_new_edges_dist, cutoff_distance)

        coo = adj_new_long_edges.tocoo()
        indices = np.vstack((coo.row, coo.col))
        indices = torch.from_numpy(indices).long()
        print('change test_pos_edge to long distance edges...')
        print(test_pos_edge_index)
        test_pos_edge_index = indices
        print(test_pos_edge_index)

    ############################################## get_sensitivity
    if args.sensitivity  or args.sensitivity_long:
        if args.sensitivity:
            roc_score_orig, ap_score_orig, adj_pred_orig, _, _ = single_gene_occlusion(u, v, test_pos_edge_index, test_neg_edge_index)
        elif args.sensitivity_long:
            roc_score_orig, ap_score_orig, adj_pred_orig, _, _ = single_gene_occlusion(u, v, test_pos_edge_index,test_neg_edge_index) #single_gene_occlusion_positive

        # compute for each gene:
        if dataset_name == 'HDST_ob' or dataset_name == 'HDST_cancer':
            df_info = pd.read_csv('data/'+ dataset_name + '/counts.csv')
            genes = df_info.columns.values

            # Calculate the test score for each gene in a loop
            single_gene_roc_score = dict()
            single_gene_ap_score = dict()
            for k in range(0,u.shape[1]):
            # for i in range(0,3):
                col_all_roc_score = []
                col_all_ap_score = []
                for j in range(30): #30
                    u_occlu = copy.deepcopy(u)
                    v_occlu = copy.deepcopy(v)
                    np.random.shuffle(u_occlu[:,k])
                    np.random.shuffle(v_occlu[:,k])
                    if args.sensitivity:
                        roc_score, ap_score, _, _, _ = single_gene_occlusion(u_occlu, v_occlu, test_pos_edge_index, test_neg_edge_index)
                    elif args.sensitivity_long:
                        roc_score, ap_score, _, _, _ = single_gene_occlusion(u_occlu, v_occlu, test_pos_edge_index,test_neg_edge_index) #single_gene_occlusion_positive
                    col_all_roc_score.append(roc_score)
                    col_all_ap_score.append(ap_score)
                    del u_occlu, v_occlu
                gene_name = genes[k] 
                print(gene_name)
                single_gene_roc_score.update({gene_name: col_all_roc_score})
                single_gene_ap_score.update({gene_name: col_all_ap_score})

            occlu_roc = {}
            occlu_ap = {}
            for k,v in single_gene_roc_score.items():
                occlu_roc[k] = np.mean(np.array(v))

            for k,v in single_gene_ap_score.items():
                occlu_ap[k] = np.mean(np.array(v))

            # Get gene sensitivity
            occlu_deta_ap = {}
            occlu_deta_roc = {}
            for k,v in occlu_ap.items():
                occlu_deta_ap[k] = np.abs(float(ap_score_orig) - np.array(v).mean())
            for k,v in occlu_roc.items():
                occlu_deta_roc[k] = np.abs(float(roc_score_orig) - np.array(v).mean())

            ## sort genes
            genes_list = occlu_deta_ap.keys()
            delta_ap = occlu_deta_ap.values()
            pd_delta_ap = pd.DataFrame({'genes_list':genes_list,'delta_ap':delta_ap})
            pd_delta_ap = pd_delta_ap.sort_values('delta_ap',ascending=False)
            file_name = 'results/deltaAUC/'+dataset_name+'_delta_ap_iter'+str(i)+'percent'+str(cutoff_ratio)+'.csv'
            pd_delta_ap.to_csv(file_name)

            genes_list = occlu_deta_roc.keys()
            delta_roc = occlu_deta_roc.values()
            pd_delta_roc = pd.DataFrame({'genes_list':genes_list,'delta_roc':delta_roc})
            pd_delta_roc = pd_delta_roc.sort_values('delta_roc',ascending=False)
            file_name = 'results/deltaAUC/'+dataset_name+'_delta_roc_iter'+str(i)+'percent'+str(cutoff_ratio)+'.csv'
            pd_delta_roc.to_csv(file_name)


        if dataset_name == 'MERFISH' or dataset_name == 'seqFISH':
            if dataset_name == 'MERFISH':
                df_info = pd.read_csv('generated_data/MERFISH/counts.csv')
                genes = df_info.columns.values
            if dataset_name == 'seqFISH':
                df_info = pd.read_csv('data/seqFISH/counts.csv')
                genes = df_info.columns.values            

            # Calculate the test score for each gene in a loop
            single_gene_roc_score = dict()
            single_gene_ap_score = dict()
            for k in range(0,u.shape[1]):
            # for k in range(0,3):
                col_all_roc_score = []
                col_all_ap_score = []
                for j in range(30): #30
                    u_occlu = copy.deepcopy(u)
                    v_occlu = copy.deepcopy(v)
                    np.random.shuffle(u_occlu[:,k])
                    np.random.shuffle(v_occlu[:,k])
                    roc_score, ap_score, _, _, _  = single_gene_occlusion(u_occlu, v_occlu, test_pos_edge_index, test_neg_edge_index)
                    col_all_roc_score.append(roc_score)
                    col_all_ap_score.append(ap_score)
                    del u_occlu, v_occlu
                gene_name = genes[k] 
                print(gene_name)
                single_gene_roc_score.update({gene_name: col_all_roc_score})
                single_gene_ap_score.update({gene_name: col_all_ap_score})

            occlu_roc = {}
            occlu_ap = {}
            for k,v in single_gene_roc_score.items():
                occlu_roc[k] = np.mean(np.array(v))

            for k,v in single_gene_ap_score.items():
                occlu_ap[k] = np.mean(np.array(v))

            # Get gene sensitivity
            occlu_deta_ap = {}
            occlu_deta_roc = {}
            for k,v in occlu_ap.items():
                occlu_deta_ap[k] = np.abs(float(ap_score_orig) - np.array(v).mean())
            for k,v in occlu_roc.items():
                occlu_deta_roc[k] = np.abs(float(roc_score_orig) - np.array(v).mean())

            ## sort genes
            genes_list = occlu_deta_ap.keys()
            delta_ap = occlu_deta_ap.values()
            pd_delta_ap = pd.DataFrame({'genes_list':genes_list,'delta_ap':delta_ap})
            pd_delta_ap = pd_delta_ap.sort_values('delta_ap',ascending=False)
            file_name = 'results/deltaAUC/'+dataset_name+'_delta_ap_iter'+str(i)+'.csv'
            pd_delta_ap.to_csv(file_name)

            genes_list = occlu_deta_roc.keys()
            delta_roc = occlu_deta_roc.values()
            pd_delta_roc = pd.DataFrame({'genes_list':genes_list,'delta_roc':delta_roc})
            pd_delta_roc = pd_delta_roc.sort_values('delta_roc',ascending=False)
            file_name = 'results/deltaAUC/'+dataset_name+'_delta_roc_iter'+str(i)+'.csv'
            pd_delta_roc.to_csv(file_name)


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



# # ############################################################ save log file 
# # from datetime import datetime
# # import json

# # now       = datetime.now()
# # date_time = now.strftime("%m/%d/%Y, %H:%M:%S")

# # log = {
# #     'dataset'       : args.dataset,
# #     'task'          : args.task,
# #     'model'         : args.model,
# #     'learning_rate' : args.learning_rate,
# #     'epochs'        : args.epochs,
# #     'hidden'        : args.hidden,
# #     'dimension'     : args.dimension,
# #     'alpha'         : args.alpha,
# #     'beta'          : args.beta,
# #     'nb_run'        : args.nb_run,
# #     'prop_val'      : args.prop_val,
# #     'prop_test'     : args.prop_test,

# #     'directed'            : args.directed,  
# #     'feature_vector_type' : args.feature_vector_type,
# #     'feature_vector_size' : args.feature_vector_size,
# #     'validate'            : args.validate,
    
# #     'date_time'     : date_time,
# #     'auc_mean'      : np.mean(mean_roc),
# #     'auc_std'       : np.std(mean_roc),
# #     'ap_mean'       : np.mean(mean_ap),
# #     'ap_std'        : np.std(mean_ap),
# #     'time_mean'     : np.mean(mean_time),
# #     'time_std'      : np.std(mean_time)
# #     }
    

# # logfile = args.logfile

# # try:
# #     data = json.load(open(logfile))
    
# #     # convert data to list if not
# #     if type(data) is dict:
# #         data = [data]
# # except:
# #     data = []
    
# # # append new item to data list
# # data.append(log)

# # # write list to file
# # with open(logfile, 'w') as outfile:
# #     json.dump(data, outfile, indent=4, sort_keys=True)
