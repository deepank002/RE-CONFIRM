import gc
import os
import pandas as pd
import random
import sys
import torch

from utils.model import *
from utils.helper import *


def training(nSubj, graphdata_subj, val_split_idx, device, folder_path, params):
    split = int(nSubj*0.8)
    shuffledata = random.sample(graphdata_subj, len(graphdata_subj))
    train_dataset = shuffledata[:split]
    test_dataset = shuffledata[split:]
    new_train = train_dataset[:val_split_idx]
    new_val = train_dataset[val_split_idx:]
    
    gnn, num_features, hidden_dims, num_layers, output_size = params
    model = GraphModel(gnn, num_features, hidden_dims, num_layers, output_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    print("Initialised model successfully")

    all_train_labels = [[data.y for data in new_train[i]] for i in range(len(new_train))]
    train_labels = torch.tensor([item[0] for item in all_train_labels])
    del all_train_labels
    gc.collect()

    all_val_labels = [[data.y for data in new_val[i]] for i in range(len(new_val))]
    val_labels = torch.tensor([item[0] for item in all_val_labels])
    del all_val_labels
    gc.collect()

    all_test_labels = [[data.y for data in test_dataset[i]] for i in range(len(test_dataset))]
    test_labels = torch.tensor([item[0] for item in all_test_labels])
    del all_test_labels
    gc.collect()

    print("Created labels successfully")

    number_epochs = 30

    df = pd.DataFrame(columns=['Epoch', 'Train_Accuracy', 'Val_Accuracy', 'Test_Accuracy', 'Val_Checkpoint_Path'])
    df = df._append([{}]*number_epochs, ignore_index=True)

    for counts, epoch in enumerate(range(1, number_epochs+1)):
        df['Epoch'][counts] = epoch
        train(new_train, train_labels.to(device), model, device, criterion, optimizer)
        train_acc = accuracy(new_train, train_labels.to(device), model, device)
        df['Train_Accuracy'][counts] = train_acc
        val_acc = accuracy(new_val, val_labels.to(device), model, device)
        df['Val_Accuracy'][counts] = val_acc
        path = save_model(model, val_acc, epoch, folder_path)
        df['Val_Checkpoint_Path'][counts] = path

    print("Finished model training and validation")

    fpath = folder_path
    files = os.listdir(fpath)

    for counts, path in enumerate(files):
        testmodel = GraphModel(gnn, num_features, hidden_dims, num_layers, output_size).to(device)
        testmodel.load_state_dict(torch.load(fpath + "/" + path))
        testmodel.eval()
        _, test_acc, _ = accuracy_probs(testmodel, test_dataset, test_labels.to(device), device)
        df['Test_Accuracy'][counts] = test_acc
        del testmodel, test_acc
        torch.cuda.empty_cache()
        gc.collect()

    df.to_csv(fpath + '/results.csv')

    print("Finished model testing")
