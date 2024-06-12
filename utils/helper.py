import gc
import networkx as nx
import numpy as np
import os
import random
import scipy.sparse as sp
import torch

from networkx.linalg.graphmatrix import adjacency_matrix as adj_mat
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import convert, to_undirected, from_scipy_sparse_matrix
from torch_geometric.utils.subgraph import k_hop_subgraph
from typing import List

from utils.nx_modified import *


def setup(gnn, method, dataset, single_site="yes"):
    path = "gnn"
    os.makedirs(path, exist_ok=True)

    if single_site == "no":
        parent_folder = "./gnn/" + gnn + "/dataset/" + dataset + "/all_sites"
    else:
        parent_folder = "./gnn/" + gnn + "/dataset/" + dataset + "/single_site"
    os.makedirs(parent_folder, exist_ok=True)

    subfolders = ["ckpts", "drc_ckpts", "ii_ckpts", "scores"]
    for subfolder in subfolders:
        subfolder_path = os.path.join(parent_folder, subfolder)
        os.makedirs(subfolder_path, exist_ok=True)

    # Define additional subfolders for scores folder
    additional_subfolders = ["drc", "mprc", "fidelity", "ii", "stability", "init", "ambivert", "hdr", "homfid", "modr", "hac", "morehac"]

    # Create the additional subfolders within scores folder
    scores_folder_path = os.path.join(parent_folder, "scores")
    for subfolder in additional_subfolders:
        subfolder_path = os.path.join(scores_folder_path, subfolder)
        os.makedirs(subfolder_path, exist_ok=True)
    
        # Create a folder named 'method' if it does not exist
        method_folder_path = os.path.join(subfolder_path, method)
        if not os.path.exists(method_folder_path):
            os.makedirs(method_folder_path)

    print("Setup finished")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(1)
    random.seed(1)
    print("Initialised PyTorch successfully")


def load_data(nSubj, label_encoded, corr_matrix, start=0, step=1):
    graphdata_subj = []
    adj_matrix = np.zeros_like(corr_matrix)
    for i in range(start, nSubj, step):
        graphdata_ts = []
        x = torch.tensor(corr_matrix[i,:,:], dtype=torch.float)
        y = torch.tensor([label_encoded[i]])
        adj_mat = np.where(corr_matrix[i,:,:] >= 0.4, 1, 0)
        adj_matrix[i,:,:] = adj_mat
        edge_index, edge_weight = from_scipy_sparse_matrix(sp.csr_matrix(adj_mat))
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
        graphdata_ts.append(data)
        dataloader = DataLoader(graphdata_ts, batch_size=1, shuffle=False)
        graphdata_subj.append(dataloader)
        del graphdata_ts, dataloader
        gc.collect()
    print("Loaded data successfully")
    return graphdata_subj, adj_matrix


def train(train_dataset, train_labels, model, device, criterion, optimizer, batch_size=4):
    model.train()
    for i in range(0, len(train_dataset), batch_size):
        all_subj_outs = []
        temp_dataset = train_dataset[i:i+batch_size]
        temp_labels = train_labels[i:i+batch_size]
        for j in range(len(temp_dataset)):
            train_loader = [data.to(device) for data in temp_dataset[j]]
            out = model(train_loader)
            all_subj_outs.append(out)
        concat_outs = torch.cat(all_subj_outs, dim=0)
        loss = criterion(concat_outs, temp_labels)
        loss.backward()   
        optimizer.step()
        optimizer.zero_grad()


def accuracy(dataset, labels, model, device, batch_size=4):
    model.eval()
    correct = 0
    for i in range(0, len(dataset), batch_size):
        all_subj_outs = []
        temp_dataset = dataset[i:i+batch_size]
        temp_labels = labels[i:i+batch_size]
        for j in range(len(temp_dataset)):
            train_loader = [data.to(device) for data in temp_dataset[j]]
            out = model(train_loader)
            all_subj_outs.append(out)
        concat_outs = torch.cat(all_subj_outs, dim=0)
        pred = concat_outs.argmax(dim=1)
        correct += int((pred == temp_labels).sum())
    return correct / len(labels)


def save_model(model, acc, epoch, path):
    path = path + "/_Model_Val_" + str(acc) + "_" + str(epoch) +".pth"
    torch.save(model.state_dict(), path)
    return path


def accuracy_probs(model, dataset, labels, device, batch_size=4):
    correct = 0
    all_pred_probs = []
    all_preds = []
    for i in range(0, len(dataset), batch_size):
        all_subj_outs = []
        temp_dataset = dataset[i:i+batch_size]
        temp_labels = labels[i:i+batch_size]
        for j in range(len(temp_dataset)):
            train_loader = [data.to(device) for data in temp_dataset[j]]
            out = model(train_loader)
            all_subj_outs.append(out)
        concat_outs = torch.cat(all_subj_outs, dim=0)
        all_pred_probs.append(concat_outs)
        pred = concat_outs.argmax(dim=1)
        all_preds.append(pred)
        correct += int((pred == temp_labels).sum())
    stacked_probs = torch.cat(all_pred_probs, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    #return stacked_probs, correct / len(labels), pred
    return stacked_probs, correct / len(labels), all_preds
    

#**********************************************************************************************************************
#    Title: Evaluating explainability for graph neural networks (https://www.nature.com/articles/s41597-023-01974-x)
#    Author: Chirag Agarwal, Owen Queen, Himabindu Lakkaraju & Marinka Zitnik
#    Availability: https://github.com/mims-harvard/GraphXAI
#**********************************************************************************************************************
def rewire_edges(edge_index: torch.Tensor, num_nodes: int,
                G: nx.Graph = None, data: Data = None,
                node_idx: int = None, num_hops: int = 3,
                rewire_prob: float = 0.01, seed: int = 912):
    """
    Rewire edges in the graph.

    If subset is None, rewire the whole graph.
    Otherwise, rewire the edges within the induced subgraph of subset of nodes.
    """
    # Get the k-hop subgraph of node_idx if specified, and compute nswap
    if node_idx is None:
        subset = None
        m = edge_index.shape[1]
        nswap = round(m*rewire_prob)
    else:
        subset, sub_edge_index, _, _ = k_hop_subgraph(node_idx, num_hops, edge_index)
        m = sub_edge_index.shape[1]
        nswap = round(m*rewire_prob)
        nswap = nswap if nswap else 1
    # Convert to networkx graph for rewiring edges
    # import ipdb; ipdb.set_trace()
    if data is None:
        data = Data(edge_index=edge_index, num_nodes=num_nodes)
    if G is None:
        G = convert.to_networkx(data, to_undirected=True)
    else:
        G = G.copy()
    rewired_G = swap(G, subset, nswap=nswap, max_tries=1000*nswap)  # , seed=seed)
    # Quick way to get edge index from networkx graph:
    rewired_edge_index = to_undirected(torch.as_tensor(list(rewired_G.edges)).t().long())
    # rewired_adj_mat = adj_mat(rewired_G)
    # rewired_edge_index = convert.from_scipy_sparse_matrix(rewired_adj_mat)[0]
    return rewired_edge_index