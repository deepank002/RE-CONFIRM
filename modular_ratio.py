import numpy as np
import re

from collections import defaultdict
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances


def modular_ratio(mask, fpath, nROI, distance_metric="manhattan"):
    file_path = '/home/MICCAI/src/assets/power264NodeNames.txt'
    with open(file_path, 'r') as file:
        rois = file.readlines()
    rois = [element.strip() for element in rois]
    rois = [re.sub(r'_\d+$', '', my_string) for my_string in rois]
    indices_dict = defaultdict(list)
    for idx, element in enumerate(rois):
        indices_dict[element].append(idx)
    
    mod_ratio_all = np.zeros((len(mask), nROI))
    for i in range(len(mask)):
        if distance_metric == "manhattan":
            pairwise_dist_mat = manhattan_distances(mask[i,:,:])

        else:
            pairwise_dist_mat = euclidean_distances(mask[i,:,:])

        #denom = np.mean(np.unique(pairwise_dist_mat[pairwise_dist_mat != 0]))
        
        #mod_ratio = {}
        for module, nodes in indices_dict.items():
            nodes.sort()
            mod_mat = mask[i,:,:][nodes]
            if distance_metric == "manhattan":
                pairwise_mod_mat = manhattan_distances(mod_mat)
            else:
                pairwise_mod_mat = euclidean_distances(mod_mat)
            
            for count, index in enumerate(nodes):
                mask_zero_num = np.ma.masked_equal(pairwise_mod_mat[count,:], 0)
                avg_mod_dist = np.mean(mask_zero_num)
                mask_zero_den = np.ma.masked_equal(pairwise_dist_mat[index,:], 0)
                avg_nod_dist = np.mean(mask_zero_den)
                mod_ratio_all[i, index] = avg_mod_dist/avg_nod_dist
            
            #numer = np.mean(np.unique(pairwise_mod_mat[pairwise_mod_mat != 0]))
            
            #mod_ratio[module] = numer/denom
        
        #mod_ratio_all.append(mod_ratio)
    
    np.save(fpath + '/modular_ratio_all_subjects.npy', mod_ratio_all)
    