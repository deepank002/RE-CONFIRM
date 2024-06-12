import copy
import numpy as np
import torch
from torch_geometric.explain import GNNExplainer, Explainer, CaptumExplainer, PGExplainer, AttentionExplainer
from captum.attr import GuidedBackprop, IntegratedGradients
#from torch_geometric.explain.metric import fidelity, unfaithfulness

from utils.model import *


def create_explainer(method, gnn_layer):
    if method == "gnnexp":
        explainer = Explainer(model=gnn_layer, algorithm=GNNExplainer(epochs=30), explanation_type='model',
                              model_config=dict(mode='binary_classification', task_level='graph', return_type='raw'),
                              node_mask_type='attributes', edge_mask_type='object')
    """if method == "ig":
        explainer = Explainer(model=gnn_layer, algorithm=CaptumExplainer('IntegratedGradients'), explanation_type='model',
                              model_config=dict(mode='binary_classification', task_level='graph', return_type='raw'),
                              node_mask_type='attributes', edge_mask_type='object')"""
    if method == "pgexp":
        explainer = Explainer(model=gnn_layer, algorithm=PGExplainer(epochs=30), explanation_type='phenomenon',
                              model_config=dict(mode='binary_classification', task_level='graph', return_type='raw'),
                              node_mask_type=None, edge_mask_type='object')
    if method == "attnexp":
        explainer = Explainer(model=gnn_layer, algorithm=AttentionExplainer(), explanation_type='model',
                              model_config=dict(mode='binary_classification', task_level='graph', return_type='raw'),
                              node_mask_type=None, edge_mask_type='object')
    if method == "gbackprop":
        explainer = GuidedBackprop(gnn_layer)
    if method == "ig":
        explainer = IntegratedGradients(gnn_layer.to("cpu"))
    return explainer


def init_explainer(dataset, directory, method, explainer, gnn_layer, gnn_data, device, node_index):
    if method == "gnnexp":
        gnn_explanation = explainer(gnn_data.x.to(device), gnn_data.edge_index.to(device))
    """if method == "ig":
        gnn_explanation = explainer(gnn_data.x.to(device), gnn_data.edge_index.to(device), index=node_index)"""
    if method == "attnexp":
        gnn_explanation = explainer(gnn_data.x.to(device), gnn_data.edge_index.to(device))
    if method == "pgexp":
        device = "cpu"
        for epoch in range(100):
            loss = explainer.algorithm.train(epoch, gnn_layer.to(device), gnn_data.x.to(device), gnn_data.edge_index.to(device), 
                                             target=gnn_layer(gnn_data.x.to(device), gnn_data.edge_index.to(device)).to(device))
        gnn_explanation = explainer(gnn_data.x.to(device), gnn_data.edge_index.to(device), target=gnn_data.y.to(device))
    if method == "gbackprop":
        if gnn_data.y.shape == torch.Size([1]):
            target = [1] * 264
        if gnn_data.y.shape == torch.Size([0]):
            target = [0] * 264
        x = gnn_data.x
        edge_index = gnn_data.edge_index
        gnn_explanation = explainer.attribute(inputs=(x.to(device), edge_index.to(device)), target=target)
    if method == "ig":
        device = "cpu"
        baselines = np.load('./assets/baselines_' + dataset + '_' + directory[1:] + '.npy')
        baselines = torch.tensor(baselines)
        if gnn_data.y.shape == torch.Size([1]):
            target = [1] * 264
        if gnn_data.y.shape == torch.Size([0]):
            target = [0] * 264
        x = gnn_data.x
        gnn_explanation = explainer.attribute(inputs=(x.to(device), x.to(device)), target=target, baselines=(baselines.to(device), baselines.to(device)), n_steps=5)
    return gnn_explanation


def load_model_weights(params, device, path, flag):
    gnn, num_features, hidden_dims, num_layers, output_size = params
    if flag != 0:
        gnn_model = GraphModel(gnn, num_features, hidden_dims, num_layers, output_size).to(device)
        gnn_model.load_state_dict(torch.load(path))
    else:
        gnn_model = GraphModel(gnn, num_features, hidden_dims, num_layers, output_size).to(device)    
    return gnn_model
    

def baseline_scores(dataset, directory, graphdata_subj, method, ckpt_paths, ones, zeros, nSubj, nROI, params, fpath, device, flag=1):
    if flag != 0:
        path = ckpt_paths['Val_Checkpoint_Path'][ckpt_paths['Test_Accuracy']==ckpt_paths['Test_Accuracy'].max()].to_list()[0]
    else:
        path = None
    gnn_model = load_model_weights(params, device, path, flag)
    gnn_model.eval()
    gnn_layer = gnn_model.gnn_convs[0]
        
    print("Loaded model weights successfully")

    all_subject_score = 0
    all_masks = torch.zeros(nSubj, nROI, nROI)
    all_indices = [num for num in range(nROI)]
    node_index = torch.tensor(all_indices)
    
    #pos_fid, neg_fid, unfaith = [], [], []
    edge_mask, prediction, target, edge_index = {}, {}, {}, {}
    for i in range(len(graphdata_subj)):
        gnn_data = [data for data in graphdata_subj[i]][0]
        explainer = create_explainer(method, gnn_layer)
        gnn_explanation = init_explainer(dataset, directory, method, explainer, gnn_layer, gnn_data, device, node_index)
        #(fplus, fminus) = fidelity(explainer, gnn_explanation)
        #unf = unfaithfulness(explainer, gnn_explanation)
        #pos_fid.append(fplus)
        #neg_fid.append(fminus)
        #unfaith.append(unf)
        gnn_node_mask = torch.zeros(nROI, nROI)
        if method == "pgexp" or method == "attnexp":
            edge_mask = gnn_explanation.edge_mask
            edge_index = gnn_explanation.edge_index
            for e in range(len(edge_mask)):
                n1, n2 = edge_index[:, e]
                gnn_node_mask[n1, n2] = edge_mask[e]
            """
            gnn_node_mask = gnn_explanation.x
            edge_mask[i+1] = gnn_explanation.edge_mask.cpu().numpy()
            prediction[i+1] = gnn_explanation.prediction.cpu().numpy()
            target[i+1] = gnn_explanation.target.cpu().numpy()
            edge_index[i+1] = gnn_explanation.edge_index.cpu().numpy()
            """
        elif method == "gbackprop" or method == "ig":
            gnn_node_mask = gnn_explanation[0]
        else:
            gnn_node_mask = gnn_explanation.node_mask
        all_masks[i, :, :] = gnn_node_mask
        all_subject_score += gnn_node_mask.sum(dim=0)

    """print(sum(pos_fid))
    print(sum(neg_fid))
    print(sum(unfaith))
    np.save("pos_fid.npy", np.array(pos_fid))
    np.save("neg_fid.npy", np.array(neg_fid))
    np.save("unfaith.npy", np.array(unfaith))"""
    
    all_subject_score /= nSubj
    
    tdc = all_masks[zeros]
    tdc_mean = torch.mean(tdc, dim=0)
    tdc_std = torch.std(tdc, dim=0)
    disease_subj = all_masks[ones]
    disease_subj_mean = torch.mean(disease_subj, dim=0)
    disease_subj_std = torch.std(disease_subj, dim=0)
    
    np.save(fpath + '/all_masks.npy', all_masks.cpu().numpy())
    
    """
    np.save(fpath + '/edge_mask.npy', np.array([edge_mask], dtype='object'))
    np.save(fpath + '/prediction.npy', np.array([prediction], dtype='object'))
    np.save(fpath + '/target.npy', np.array([target], dtype='object'))
    np.save(fpath + '/edge_index.npy', np.array([edge_index], dtype='object'))
    """

    np.save(fpath + '/all_subject_scores.npy', all_subject_score.cpu().numpy())
    np.save(fpath + '/tdc_node_mask_mean_scores.npy', tdc_mean.cpu().numpy())
    np.save(fpath + '/disease_subj_node_mask_mean_scores.npy', disease_subj_mean.cpu().numpy())
    np.save(fpath + '/tdc_node_mask_std_scores.npy', tdc_std.cpu().numpy())
    np.save(fpath + '/disease_subj_node_mask_std_scores.npy', disease_subj_std.cpu().numpy())
    print("Saved all scores successfully")
