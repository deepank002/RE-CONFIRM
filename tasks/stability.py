import copy
import numpy as np
import random
import torch
from sklearn.metrics.pairwise import cosine_similarity

from tasks.scores import *

from utils.model import *
from utils.helper import *


def stability(dataset, directory, ckpt_paths, graphdata_subj, method, nSubj, nROI, params, fpath, device, flag=1):
    path = ckpt_paths['Val_Checkpoint_Path'][ckpt_paths['Test_Accuracy']==ckpt_paths['Test_Accuracy'].max()].to_list()[0]
    gnn_model = load_model_weights(params, device, path, flag)
    gnn_model.eval()
    gnn_layer = gnn_model.gnn_convs[0]

    all_subject_score = 0
    all_masks = torch.zeros(nSubj, nROI, nROI)
    all_indices = [num for num in range(nROI)]
    node_index = torch.tensor(all_indices)
    
    graphdata_copy = copy.deepcopy(graphdata_subj)
    
    if method == "pgexp":
        device = "cpu"
    
    for i in range(len(graphdata_subj)):
        gnn_data = [data for data in graphdata_subj[i]][0]
        explainer = create_explainer(method, gnn_layer)
        noise = torch.randn_like(gnn_data.x) * 1.0  # Add Gaussian noise
        gnn_data.x += noise
        gnn_data.edge_index = rewire_edges(gnn_data.edge_index, num_nodes=1)
        gnn_explanation = init_explainer(dataset, directory, method, explainer, gnn_layer, gnn_data, device, node_index)
        if method == "pgexp" or method == "attnexp":
            gnn_node_mask = gnn_explanation.x
        elif method == "gbackprop" or method == "ig":
            gnn_node_mask = gnn_explanation[0]
        else:
            gnn_node_mask = gnn_explanation.node_mask
        all_masks[i, :, :] = gnn_node_mask
        all_subject_score += gnn_node_mask.sum(dim=0)

    all_subject_score /= nSubj
    
    randomize = random.sample(range(len(graphdata_subj)), len(graphdata_subj))
    
    shuffleddata1, shuffleddata2 = [], []
    for i in randomize:
        shuffleddata1.append(graphdata_subj[i])
        shuffleddata2.append(graphdata_copy[i])
    
    all_labels = [[data.y for data in shuffleddata1[i]] for i in range(len(shuffleddata1))]
    labels1 = torch.tensor([item[0] for item in all_labels])
    del all_labels
    gc.collect()

    all_labels = [[data.y for data in shuffleddata2[i]] for i in range(len(shuffleddata2))]
    labels2 = torch.tensor([item[0] for item in all_labels])
    del all_labels
    gc.collect()
    
    _, _, noisy_probs = accuracy_probs(gnn_model.to(device), shuffleddata1, labels1.to(device), device)
    _, _, orig_probs = accuracy_probs(gnn_model.to(device), shuffleddata2, labels2.to(device), device)
    
    print(noisy_probs)
    print()
    print(orig_probs)
    print()
    if torch.allclose(labels1, labels2):
        print(labels1)
    
    np.save(fpath + '/all_masks.npy', all_masks.cpu().numpy())
    np.save(fpath + '/all_subject_scores.npy', all_subject_score.cpu().numpy())
    np.save(fpath + '/noisy_predictions.npy', noisy_probs.cpu().numpy())
    np.save(fpath + '/noiseless_predictions.npy', orig_probs.cpu().numpy())
    np.save(fpath + '/all_labels.npy', labels1.cpu().numpy())
