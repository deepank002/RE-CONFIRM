import os
import re
import numpy as np

from nilearn.connectome import ConnectivityMeasure

def setup_ADHD(nROI):
    #sites = ['NI', 'NYU', 'OHSU', 'PKU']
    sites = ['NYU']
    all_outputs = []
    outputs = np.empty((0, 2))
    label_encoded = np.array([])
    for site in sites:
        temp_outputs = np.load('/home/data/ADHD_' + site + '_Y.npy')
        temp_label_encoded = np.argmax(temp_outputs, axis=1)
        all_outputs.append(temp_outputs)
        label_encoded = np.append(label_encoded, temp_label_encoded)
        #break

    for arr in all_outputs:
        outputs = np.vstack([outputs, arr])

    label_encoded = label_encoded.astype(np.int64)
    #np.save("all_labels_encoded_adhd.npy", label_encoded)
    ones = np.where(label_encoded == 1)
    #np.save("diseased_labels_encoded_adhd.npy", ones)
    zeros = np.where(label_encoded == 0)
    #np.save("control_labels_encoded_adhd.npy", zeros)
    
    time_series = np.array([])
    for site in sites:
        temp_time_series = np.load('/home/data/ADHD_' + site + '_TS.npy', allow_pickle=True)
        for i in range(len(temp_time_series)):
            temp_time_series[i] = temp_time_series[i].T
        time_series = np.append(time_series, temp_time_series)
        #break

    nSubj = len(outputs)

    corr_measure = ConnectivityMeasure(kind="correlation")
    corr_matrix = np.zeros((nSubj, nROI, nROI))

    for i in range(nSubj):
        corr_matrix[i,:,:] = corr_measure.fit_transform([time_series[i]])[0]

    #val_split_idx = 521 # all sites combined
    val_split_idx = 237 # only site NYU
    
    return label_encoded, corr_matrix, ones, zeros, nSubj, val_split_idx


def setup_ABIDE(path, nROI):
    files = os.listdir(path)
    files.sort()
    diseased_files = os.listdir(path + '/' + files[0])
    normal_files = os.listdir(path + '/' + files[1])
    
    ds_matrix = np.zeros((len(diseased_files), nROI, nROI))
    ds = []
    dsd = {}
    for count, file in enumerate(diseased_files):
        subj_id = re.findall(r'\d+', file)
        temp = np.load(path + '/' + files[0] + '/' + file)
        ds_matrix[count, :, :] = temp
        ds.append(1) # diseased -> label 1
        dsd[count] = subj_id
    
    nm_matrix = np.zeros((len(normal_files), nROI, nROI))
    nm = []
    nmd = {}
    for count, file in enumerate(normal_files):
        subj_id = re.findall(r'\d+', file)
        temp = np.load(path + '/' + files[1] + '/' + file)
        nm_matrix[count, :, :] = temp
        nm.append(0) # control -> label 0
        nmd[count] = subj_id

    corr_matrix = np.concatenate((ds_matrix, nm_matrix), axis=0)
    label_encoded = np.array(ds+nm)
    all_subject_ids = {}
    for key, value in dsd.items():
        all_subject_ids[key] = value
    for key, value in nmd.items():
        all_subject_ids[key + max(dsd.keys()) + 1] = value   
    
    non_nan_list = []
    for i in range(len(corr_matrix)):
        if np.isnan(corr_matrix[i,:,:]).sum() == 0:
            non_nan_list.append(i)

    corr_matrix = corr_matrix[non_nan_list]
    label_encoded = label_encoded[non_nan_list]
    #np.save("all_labels_encoded_abide.npy", label_encoded)
    ones = np.where(label_encoded == 1)
    #np.save("diseased_labels_encoded_abide.npy", ones)
    zeros = np.where(label_encoded == 0)
    #np.save("control_labels_encoded_abide.npy", zeros)
    nSubj = len(label_encoded)
    all_subject_ids = {key: value for key, value in all_subject_ids.items() if key in non_nan_list}
    all_subject_ids = {index: value for index, value in enumerate(all_subject_ids.values(), start=0)}
    
    return label_encoded, corr_matrix, ones, zeros, nSubj, all_subject_ids