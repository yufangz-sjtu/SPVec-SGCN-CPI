import numpy as np
import json
import pdb
import scipy.sparse
from sklearn.preprocessing import StandardScaler
import os
import yaml
import scipy.sparse as sp
from copy import deepcopy
import random
from graphsaint.globals import *
from time import sleep
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


def feature_prepare(dict_disease_similarity, dict_mirna_similarity,train_datapair,test_datapair,task,file_name):     
    assert task in  ['tp','td','tm','tn']
    print('task:{}!!!!'.format(task))
    
    train_datapair = list(train_datapair)
    print('训练集数量：',len(train_datapair))
    
    train_datapair.extend(list(test_datapair))
    datapair = deepcopy(train_datapair)
    print(len(datapair),"测试集数量：",len(test_datapair))
    
    
    
    """--------------------------生成features-------------------------------------------"""
    similarity= []
    features = []
    for pairs in tqdm(list(datapair)):
        disease_idx = pairs[0]
        mirna_idx = pairs[1]
        #print(disease_idx,mirna_idx)
        similarity= []
        similarity.extend(list(dict_disease_similarity.item()[disease_idx]))
        #print(len(similarity))
        similarity.extend(list(dict_mirna_similarity.item()[mirna_idx]))
        #print(len(similarity))
        features.append(similarity)
        #print(len(features))
        feats = np.array(features)
        sleep(0.01)
        
    # ---- normalize feats ----
    train_nodes = len(datapair) - len(test_datapair)  
    train_feats = feats[:train_nodes]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)
    # -------------------------
    sleep(0.5)
    print ('features:',type(feats),feats.shape)
    
    np.save(file_name,feats)
    return feats
    
    



def data_prepare(train_datapair,test_datapair,train_label,test_label,task,task_neighbors_edge,feats,seed):
    """
    Load the various data files residing in the `prefix` directory.
    Files to be loaded:
        adj_full.npz        sparse matrix in CSR format, stored as scipy.sparse.csr_matrix
                            The shape is N by N. Non-zeros in the matrix correspond to all
                            the edges in the full graph. It doesn't matter if the two nodes
                            connected by an edge are training, validation or test nodes.
                            For unweighted graph, the non-zeros are all 1.
        adj_train.npz       sparse matrix in CSR format, stored as a scipy.sparse.csr_matrix
                            The shape is also N by N. However, non-zeros in the matrix only
                            correspond to edges connecting two training nodes. The graph
                            sampler only picks nodes/edges from this adj_train, not adj_full.
                            Therefore, neither the attribute information nor the structural
                            information are revealed during training. Also, note that only
                            a x N rows and cols of adj_train contains non-zeros. For
                            unweighted graph, the non-zeros are all 1.
        role.json           a dict of three keys. Key 'tr' corresponds to the list of all
                              'tr':     list of all training node indices
                              'va':     list of all validation node indices
                              'te':     list of all test node indices
                            Note that in the raw data, nodes may have string-type ID. You
                            need to re-assign numerical ID (0 to N-1) to the nodes, so that
                            you can index into the matrices of adj, features and class labels.
        class_map.json      a dict of length N. Each key is a node index, and each value is
                            either a length C binary list (for multi-class classification)
                            or an integer scalar (0 to C-1, for single-class classification).
        feats.npz           a numpy array of shape N by F. Row i corresponds to the attribute
                            vector of node i.

    Inputs:
        prefix              string, directory containing the above graph related files
        normalize           bool, whether or not to normalize the node features

    Outputs:
        adj_full            scipy sparse CSR (shape N x N, |E| non-zeros), the adj matrix of
                            the full graph, with N being total num of train + val + test nodes.
        adj_train           scipy sparse CSR (shape N x N, |E'| non-zeros), the adj matrix of
                            the training graph. While the shape is the same as adj_full, the
                            rows/cols corresponding to val/test nodes in adj_train are all-zero.
        feats               np array (shape N x f), the node feature matrix, with f being the
                            length of each node feature vector.
        class_map           dict, where key is the node ID and value is the classes this node
                            belongs to.
        role                dict, where keys are: 'tr' for train, 'va' for validation and 'te'
                            for test nodes. The value is the list of IDs of nodes belonging to
                            the train/val/test sets.
    """

    assert task in  ['tp','td','tm','tn']
    print('task:{}!!!!'.format(task))
    print(task_neighbors_edge)
    
    train_label = list(train_label)
    train_label.extend(list(test_label))
    #class_map = deepcopy(train_label) #label
    class_map = dict(zip(range(len(train_label)),train_label))
    
    train_datapair = list(train_datapair)
    print('训练集数量：',len(train_datapair))
    
    train_datapair.extend(list(test_datapair))
    datapair = deepcopy(train_datapair)
    print(len(datapair),"测试集数量：",len(test_datapair))
    
    
    
    
    '''-----------------------------------生成role--------------------------------------''' 
    
    tr_num = len(datapair)-len(test_datapair)
    random_index = list(range(tr_num))
    
    random.seed(seed)
    random.shuffle(random_index)
    
    
    k_folds = 5
    CV_size = int(tr_num / k_folds)
    temp = np.array(random_index[:tr_num - tr_num %
                                 k_folds]).reshape(k_folds, CV_size,  -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
        random_index[tr_num - tr_num % k_folds:]
    
    tr_index = list(np.array(temp[1:]).reshape(1,CV_size*4))
    #print(tr_index[0])
    val_index = list(np.array(temp[0]).reshape(1,CV_size))
    te_index = list(range(len(datapair)-len(test_datapair),len(datapair))) #list of all test node indices
    print('tr:',len(tr_index[0]),'va',len(val_index[0]),'te',len(te_index))
    
    role ={}
    role['te'] = te_index
    role['va'] = val_index[0]
    role['tr'] = tr_index[0]
    
    """---------------------------生成邻接矩阵-------------------------------------------"""
    adj_full = sp.load_npz(task_neighbors_edge)
    
    adj_train = np.matrix(adj_full.todense(), copy=True)
    
    index = np.hstack((val_index[0],te_index)) 
    print(index.shape)
    
    for idx in list(index):
        #print(idx)
    
        adj_train[idx] = 0
        adj_train[:,idx] =0
    
    adj_train = sp.csr_matrix(adj_train)  
    
    print("全部link的数量：",adj_full.sum(),"  训练link的数量： ",adj_train.sum())
    
    feats = np.load(feats)
    print ('features:',type(feats),feats.shape) 
    
    return adj_full,adj_train,role,class_map,feats


def process_graph_data(adj_full,adj_train,role,class_map,feats):
    """
    setup vertex property map for output classes, train/val/test masks, and feats
    """
    num_vertices = adj_full.shape[0]
    if isinstance(list(class_map.values())[0],list):
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_vertices, num_classes))
        for k,v in class_map.items():
            class_arr[k] = v
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = np.zeros((num_vertices, num_classes))
        offset = min(class_map.values())
        for k,v in class_map.items():
            class_arr[k][v-offset] = 1
    return adj_full, adj_train, feats, class_arr, role


def parse_layer_yml(arch_gcn,dim_input):
    """
    Parse the *.yml config file to retrieve the GNN structure.
    """
    num_layers = len(arch_gcn['arch'].split('-'))
    # set default values, then update by arch_gcn
    bias_layer = [arch_gcn['bias']]*num_layers
    act_layer = [arch_gcn['act']]*num_layers
    aggr_layer = [arch_gcn['aggr']]*num_layers
    dims_layer = [arch_gcn['dim']]*num_layers
    order_layer = [int(o) for o in arch_gcn['arch'].split('-')]
    return [dim_input]+dims_layer,order_layer,act_layer,bias_layer,aggr_layer


def parse_n_prepare(flags):
    with open(flags.train_config) as f_train_config:
        train_config = yaml.load(f_train_config,Loader=yaml.FullLoader)
    arch_gcn = {
        'dim': -1,
        'aggr': 'concat',
        'loss': 'softmax',
        'arch': '1',
        'act': 'I',
        'bias': 'norm'
    }
    arch_gcn.update(train_config['network'][0])
    train_params = {
        'lr': 0.01,
        'weight_decay': 0.,
        'norm_loss': True,
        'norm_aggr': True,
        'q_threshold': 50,
        'q_offset': 0
    }
    train_params.update(train_config['params'][0])
    train_phases = train_config['phase']
    for ph in train_phases:
        assert 'end' in ph
        assert 'sampler' in ph
   
    return train_params,train_phases,arch_gcn





def log_dir(f_train_config,prefix,git_branch,git_rev,timestamp):
    import getpass
    log_dir = args_global.dir_log+"/log_train/" + prefix.split("/")[-1]
    log_dir += "/{ts}-{model}-{gitrev:s}/".format(
            model='graphsaint',
            gitrev=git_rev.strip(),
            ts=timestamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if f_train_config != '':
        from shutil import copyfile
        copyfile(f_train_config,'{}/{}'.format(log_dir,f_train_config.split('/')[-1]))
    return log_dir

def sess_dir(dims,train_config,prefix,git_branch,git_rev,timestamp):
    import getpass
    log_dir = "saved_models/" + prefix.split("/")[-1]
    log_dir += "/{ts}-{model}-{gitrev:s}-{layer}/".format(
            model='graphsaint',
            gitrev=git_rev.strip(),
            layer='-'.join(dims),
            ts=timestamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return sess_dir


def adj_norm(adj, deg=None, sort_indices=True):
    """
    Normalize adj according to the method of rw normalization.
    Note that sym norm is used in the original GCN paper (kipf),
    while rw norm is used in GraphSAGE and some other variants.
    Here we don't perform sym norm since it doesn't seem to
    help with accuracy improvement.

    # Procedure:
    #       1. adj add self-connection --> adj'
    #       2. D' deg matrix from adj'
    #       3. norm by D^{-1} x adj'
    if sort_indices is True, we re-sort the indices of the returned adj
    Note that after 'dot' the indices of a node would be in descending order
    rather than ascending order
    """
    diag_shape = (adj.shape[0],adj.shape[1])
    D = adj.sum(1).flatten() if deg is None else deg
    norm_diag = sp.dia_matrix((1/D,0),shape=diag_shape)
    adj_norm = norm_diag.dot(adj)
    if sort_indices:
        adj_norm.sort_indices()
    return adj_norm




##################
# PRINTING UTILS #
#----------------#

_bcolors = {'header': '\033[95m',
            'blue': '\033[94m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'bold': '\033[1m',
            'underline': '\033[4m'}


def printf(msg,style=''):
    if not style or style == 'black':
        print(msg)
    else:
        print("{color1}{msg}{color2}".format(color1=_bcolors[style],msg=msg,color2='\033[0m'))
