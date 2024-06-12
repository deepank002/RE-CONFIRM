import sys

if len(sys.argv) != 6:
    print("usage: have to provide five arguments in total")
    sys.exit(1)

gnn = sys.argv[1]           # type of gnn 
method = sys.argv[2]        # method of explainability
task = sys.argv[3]          # different metric algorithms
dataset = sys.argv[4]       # dataset type
single_site = sys.argv[5]   # only single site used ("yes" or "no") 

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import pandas as pd
from utils.helper import *
from utils.dataset import *

import warnings
warnings.filterwarnings("ignore")

setup(gnn, method, dataset, single_site)

set_seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nROI = 264 # power 264 atlas
num_features = nROI
hidden_dims = [128, 64]
num_layers = 2
output_size = 2
params = [gnn, num_features, hidden_dims, num_layers, output_size]

global directory, path, label_encoded, corr_matrix, ones, zeros, nSubj, val_split_idx, all_subject_ids 

if single_site == "no" and dataset == "abide":
    directory = "/all_sites"
    path = '/home/data/ABIDE-I/processed_corr_mat'
    val_split_idx = 493 

if single_site == "yes" and dataset == "abide":
    directory = "/single_site"
    path = '/home/data/ABIDE-I/only_NYU'
    val_split_idx = 74 

if single_site == "no" and dataset == "adhd":
    directory = "/all_sites"

if single_site == "yes" and dataset == "adhd":
    directory = "/single_site"

if dataset == "adhd":
    label_encoded, corr_matrix, ones, zeros, nSubj, val_split_idx = setup_ADHD(nROI)

if dataset == "abide":
    label_encoded, corr_matrix, ones, zeros, nSubj, all_subject_ids = setup_ABIDE(path, nROI)

if task == "train":
    from tasks.training import * 

    graphdata_subj, _ = load_data(nSubj, label_encoded, corr_matrix)
    folder_path = "./gnn/" + gnn + "/dataset/" + dataset + directory + "/ckpts"

    training(nSubj, graphdata_subj, val_split_idx, device, folder_path, params)

if task == "scores":
    from tasks.scores import *
    
    graphdata_subj, _ = load_data(nSubj, label_encoded, corr_matrix)
    ckpt_paths = pd.read_csv("./gnn/" + gnn + "/dataset/" + dataset + directory + "/ckpts" + "/results.csv")
    fpath = './gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/init/' + method
    
    baseline_scores(dataset, directory, graphdata_subj, method, ckpt_paths, ones, zeros, nSubj, nROI, params, fpath, device, flag=1)

if task == "mprc":
    from tasks.scores import *
    
    graphdata_subj, _ = load_data(nSubj, label_encoded, corr_matrix)
    ckpt_paths = None
    fpath = './gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/mprc/' + method
    
    baseline_scores(dataset, directory, graphdata_subj, method, ckpt_paths, ones, zeros, nSubj, nROI, params, fpath, device, flag=0)

if task == "drc":
    from tasks.training import *
    from tasks.scores import *
    
    np.random.shuffle(label_encoded)
    graphdata_subj, _ = load_data(nSubj, label_encoded, corr_matrix)
    folder_path = "./gnn/" + gnn + "/dataset/" + dataset + directory + "/drc_ckpts"
    #training(nSubj, graphdata_subj, val_split_idx, device, folder_path, params)
    
    ckpt_paths = pd.read_csv("./gnn/" + gnn + "/dataset/" + dataset + directory + "/drc_ckpts" + "/results.csv")
    fpath = './gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/drc/' + method
    
    baseline_scores(dataset, directory, graphdata_subj, method, ckpt_paths, ones, zeros, nSubj, nROI, params, fpath, device, flag=1)

if task == "fidelity":
    from tasks.fidelity import *

    ckpt_paths = pd.read_csv("./gnn/" + gnn + "/dataset/" + dataset + directory + "/ckpts" + "/results.csv")
    orig_data = np.load('./gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/init/' + method + '/all_subject_scores.npy')
    fpath = './gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/fidelity/' + method
    
    topk = input("Enter the number of top features you wish to analyze: ")
    fidelity(ckpt_paths, orig_data, fpath, nROI, nSubj, label_encoded, corr_matrix, topk, params, device, "plus", flag=1)
    fidelity(ckpt_paths, orig_data, fpath, nROI, nSubj, label_encoded, corr_matrix, topk, params, device, "minus", flag=1)

if task == "fidelity_individual":
    from tasks.fidelity_individual import *

    ckpt_paths = pd.read_csv("./gnn/" + gnn + "/dataset/" + dataset + directory + "/ckpts" + "/results.csv")
    mask = np.load('./gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/init/' + method + '/all_masks.npy')
    graphdata_subj, _ = load_data(nSubj, label_encoded, corr_matrix)
    fpath = './gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/fidelity/' + method

    #topk = input("Enter the number of top features you wish to analyze: ")
    topk = 20
    fidelity_individual(gnn, graphdata_subj, mask, ckpt_paths, fpath, nROI, nSubj, label_encoded, corr_matrix, topk, params, device, "plus", flag=1)
    fidelity_individual(gnn, graphdata_subj, mask, ckpt_paths, fpath, nROI, nSubj, label_encoded, corr_matrix, topk, params, device, "minus", flag=1)

if task == "stability":
    from tasks.stability import *
    
    ckpt_paths = pd.read_csv("./gnn/" + gnn + "/dataset/" + dataset + directory + "/ckpts" + "/results.csv")
    graphdata_subj, _ = load_data(nSubj, label_encoded, corr_matrix)
    fpath = './gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/stability/' + method
    
    stability(dataset, directory, ckpt_paths, graphdata_subj, method, nSubj, nROI, params, fpath, device, flag=1)

if task == "invariance":
    from tasks.training import *
    from tasks.scores import *
    
    set_seed(12345678)

    graphdata_subj, _ = load_data(nSubj, label_encoded, corr_matrix)
    folder_path = "./gnn/" + gnn + "/dataset/" + dataset + directory + "/ii_ckpts"
    #training(nSubj, graphdata_subj, val_split_idx, device, folder_path, params)
    
    ckpt_paths = pd.read_csv("./gnn/" + gnn + "/dataset/" + dataset + directory + "/ii_ckpts" + "/results.csv")
    fpath = './gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/ii/' + method
    
    baseline_scores(dataset, directory, graphdata_subj, method, ckpt_paths, ones, zeros, nSubj, nROI, params, fpath, device, flag=1)

if task == "ambivert":
    from tasks.ambivert import *
    
    mask = np.load('./gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/init/' + method + '/all_masks.npy')
    fpath = './gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/ambivert/' + method
    amb_score = np.load('./assets/' + dataset + '_' + directory[1:] + '/' + 'z_score_node_ambivert_degree.npy')
    
    #ambivert(mask, all_subject_ids, 'NYU', fpath)
    ambivert(mask, amb_score, fpath)

if task == "hdr":
    from tasks.homophilic_ratio import *
    
    mask = np.load('./gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/init/' + method + '/all_masks.npy')
    #_, adj_matrix = load_data(nSubj, label_encoded, corr_matrix)
    fpath = './gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/hdr/' + method
    
    #compute_homophilic_ratio(adj_matrix, mask, "cityblock", fpath)
    compute_homophilic_ratio(nSubj, nROI, corr_matrix, mask, "cityblock", fpath)
    
if task == "homfid":
    from tasks.fidelity_individual import *

    ckpt_paths = pd.read_csv("./gnn/" + gnn + "/dataset/" + dataset + directory + "/ckpts" + "/results.csv")
    mask = np.load('./gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/hdr/' + method + '/homophilic_ratio_all_subjects.npy')
    graphdata_subj, _ = load_data(nSubj, label_encoded, corr_matrix)
    fpath = './gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/homfid/' + method
    
    #topk = input("Enter the number of top features you wish to analyze: ")
    topk = 20
    fidelity_individual(gnn, graphdata_subj, mask, ckpt_paths, fpath, nROI, nSubj, label_encoded, corr_matrix, topk, params, device, "plus", flag=1, scaling="no")
    fidelity_individual(gnn, graphdata_subj, mask, ckpt_paths, fpath, nROI, nSubj, label_encoded, corr_matrix, topk, params, device, "minus", flag=1, scaling="no")

if task == "modr":
    from tasks.modular_ratio import *
    
    mask = np.load('./gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/init/' + method + '/all_masks.npy')
    fpath = './gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/modr/' + method

    modular_ratio(mask, fpath, nROI)
    
if task == "hac":
    from tasks.hub_assortativity import *
    
    mask = np.load('./gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/init/' + method + '/all_masks.npy')
    #_, adj_matrix = load_data(nSubj, label_encoded, corr_matrix)
    fpath = './gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/hac/' + method
    ambivert_score = np.load('./assets/' + dataset + '_' + directory[1:] + '/' + 'z_score_node_ambivert_degree.npy')

    #hub_assortativity_coeff(mask, adj_matrix, all_subject_ids, 'NYU', fpath)
    #hub_assortativity_coeff(mask, corr_matrix, all_subject_ids, 'NYU', fpath)
    hub_assortativity_coeff(mask, corr_matrix, fpath, ambivert_score)

if task == "hubs":
    from tasks.hub_scores import *

    subj_modules = np.load('./assets/subject_modules.npy')

    generate_hub_scores(dataset, directory, nSubj, nROI, corr_matrix, subj_modules)

if task == "morehac":
    from tasks.hub_assortativity import *
    
    mask = np.load('./gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/init/' + method + '/all_masks.npy')
    #_, adj_matrix = load_data(nSubj, label_encoded, corr_matrix)
    fpath = './gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/morehac/' + method
    
    mod_path = './assets/' + dataset + '_' + directory[1:] + '/' + 'z_score_node_ambivert_degree.npy'
    conn_path = './assets/' + dataset + '_' + directory[1:] + '/' + 'node_participation_coeff.npy'
    mod_score = np.load(mod_path)
    conn_score = np.load(conn_path)
    netw_hub = mod_score + conn_score
    netw_name = mod_path.split('/')[-1].split('.')[0] + '_' + conn_path.split('/')[-1].split('.')[0]
    
    #hub_assortativity_coeff(mask, adj_matrix, all_subject_ids, 'NYU', fpath)
    #hub_assortativity_coeff(mask, corr_matrix, all_subject_ids, 'NYU', fpath, netw_hub, netw_name)
    hub_assortativity_coeff(mask, corr_matrix, fpath, netw_hub, flag=1, netw_name=netw_name)