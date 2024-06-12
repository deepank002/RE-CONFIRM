import copy
import gc
import numpy as np
import torch
import torch.nn.functional as F

from tasks.scores import *

from utils.model import *
from utils.helper import *


def fidelity_individual(gnn, graphdata_subj, mask, ckpt_paths, fpath, nROI, nSubj, label_encoded, corr_matrix, topk, params, device, sign, flag=1, scaling="yes"):
    path = ckpt_paths['Val_Checkpoint_Path'][ckpt_paths['Test_Accuracy']==ckpt_paths['Test_Accuracy'].max()].to_list()[0]
    
    topk = int(topk)
    nROI -= topk
    new_corr_matrix = np.zeros((nSubj, nROI, nROI))
    
    fp, opred, npred = [], [], []
    for i in range(len(mask)):
        if scaling == "yes":
            temp = mask[i]
            orig_data = np.sum(temp, axis=0)
            min_val = np.min(orig_data)
            max_val = np.max(orig_data)
            data = (orig_data - min_val) / (max_val - min_val)
        else:
            data = mask[i]

        #sorted_indices = np.argsort(data)[::-1]
        sorted_indices = np.argsort(np.abs(data))[::-1]
        if sign == "plus":
            topROIs = sorted_indices[0:topk]
        if sign == "minus":
            topROIs = sorted_indices[-topk:]

        gnn_model = load_model_weights(params, device, path, flag)
        gnn_model.eval()
   
        new_gnn_model = copy.deepcopy(gnn_model)
        for name, param in new_gnn_model.named_parameters():
            if param.requires_grad:
                if gnn == "gcn":
                    if name == "gnn_convs.0.lin.weight":
                        columns_to_keep = [i for i in range(param.data.shape[1]) if i not in topROIs]
                        param.data = param.data[:, columns_to_keep]
                if gnn == "sage":
                    if name == "gnn_convs.0.lin_l.weight" or name == "gnn_convs.0.lin_r.weight":
                        columns_to_keep = [i for i in range(param.data.shape[1]) if i not in topROIs]
                        param.data = param.data[:, columns_to_keep]
                if gnn == "cheb":
                    if name == "gnn_convs.0.lins.0.weight" or name == "gnn_convs.0.lins.1.weight":
                        columns_to_keep = [i for i in range(param.data.shape[1]) if i not in topROIs]
                        param.data = param.data[:, columns_to_keep]
                if gnn == "gat":
                    if name == "gnn_convs.0.lin_src.weight":
                        columns_to_keep = [i for i in range(param.data.shape[1]) if i not in topROIs]
                        param.data = param.data[:, columns_to_keep]

        new_gnn_model.eval()
        
        temp_matrix = corr_matrix[i,:,:]
        masks = np.ones(temp_matrix.shape[0], dtype=bool)
        masks[topROIs] = False
        new_corr_matrix[i,:,:] = temp_matrix[masks][:, masks]
   
        new_graphdata_subj, _ = load_data(i-1, label_encoded, new_corr_matrix, start=i, step=-1)

        all_test_labels = [[data.y for data in new_graphdata_subj[i]] for i in range(len(new_graphdata_subj))]
        test_labels = torch.tensor([item[0] for item in all_test_labels])
        del all_test_labels
        gc.collect()
        
        orig_graphdata = [graphdata_subj[i]]
        all_test_labels = [[data.y for data in orig_graphdata[i]] for i in range(len(orig_graphdata))]
        orig_test_labels = torch.tensor([item[0] for item in all_test_labels])
        del all_test_labels
        gc.collect()

        orig_prob, _, orig_pred = accuracy_probs(gnn_model, orig_graphdata, orig_test_labels.to(device), device)
        orig_prob_softmax = F.softmax(orig_prob, dim=1)
        
        new_prob, _, new_pred = accuracy_probs(new_gnn_model, new_graphdata_subj, test_labels.to(device), device)
        new_prob_softmax = F.softmax(new_prob, dim=1)
        
        opred.append(orig_pred.cpu().detach().numpy()[0])
        npred.append(new_pred.cpu().detach().numpy()[0])
        difference = np.abs(orig_prob_softmax.cpu().detach().numpy()-new_prob_softmax.cpu().detach().numpy())
        fp.append(difference[0][0])

    np.save(fpath + '/fidelity_individual_score_' + sign + '_' + str(topk) + '.npy', np.array(fp))
    np.save(fpath + '/original_prediction_labels_' + sign + '_' + str(topk) + '.npy', np.array(opred))
    np.save(fpath + '/new_prediction_labels_' + sign + '_' + str(topk) + '.npy', np.array(npred))
