import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def ambivert(mask, amb_score, fpath):
    corr_list = []
    spearman_list = []
    for i in range(len(mask)):
        temp = mask[i]
        orig_data = np.sum(temp, axis=0)
        #min_val = np.min(orig_data)
        #max_val = np.max(orig_data)
        #data = (orig_data - min_val) / (max_val - min_val)
        
        """
        subject_id = all_subject_ids[i]
        subject_id = site_name + '_' + subject_id[0]
        
        matching_rows = amb[amb[0] == subject_id]
        tlist1 = matching_rows.values.tolist()
        amb_score = np.array(tlist1[0][1:])
        """
        
        corr = np.corrcoef(orig_data, amb_score[i,:])
        spearman = spearmanr(orig_data, amb_score[i,:])
        corr_list.append(corr)
        spearman_list.append(spearman.statistic)
    
    corr_array = np.array(corr_list)
    spearman_array = np.array(spearman_list)

    np.save(fpath + '/corr_ambivert.npy', corr_array[:,1,0])
    np.save(fpath + '/spearman_ambivert.npy', spearman_array)
    
    #print("mean correlation: ", np.mean(corr_array))
    #print("standard deviation correlation: ", np.std(corr_array))
