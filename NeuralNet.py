# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:30:37 2022

@author: arvidro
"""
# Batch size (Normal way) would correspond to time steps (e.g. there are x time steps per epoch)
# 
import numpy as np
import torch
import torch.autograd as autograd
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.jit import script, trace
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler

input_names = ("Inertia[GW]", "Total_production", "Share_Wind", "Share_Conv","month", "hour")
output_names = ("Inertia[GW]","Total_production", "Share_Wind", "Share_Conv","month", "hour")
datafile_path = "Data/CleanedTrainingset16-22.csv"
scale_columns = ["Inertia[GW]","Total_production", "Share_Wind", "Share_Conv",
                 "month", "hour"]
def to_supervised(df, n_in, n_out, input_names, output_names):
    cols = []
    names = []
    for name in input_names:
        for i in range(n_in, 0, -1):
            cols.append(df[name].shift(i))
            names.append(name + "(t-{})".format(i))
    for name in output_names:
        for i in range(0, n_out):
            cols.append(df[name].shift(-i))
            if i == 0:
                names.append(name + "(t)")
            else:
                names.append(name + "(t+{})".format(i))
    supervised = pd.concat(cols, axis=1)
    supervised.columns = names
    supervised.dropna(inplace=True)
    return supervised

def get_split_sets(split=(0.8,0.05,0.15), scaler=MinMaxScaler(feature_range=(0,1)),
             data_path=datafile_path, scale_columns=scale_columns,
             input_names=input_names, output_names=output_names, n_samples=24, n_out=1):
    # Perform a train-val-test split, scale data and return datasets
    dataset = pd.read_csv(datafile_path, index_col=0)
    dataset[scale_columns] = scaler.fit_transform(dataset[scale_columns])
    
    supervised = to_supervised(dataset, n_samples, n_out, input_names, output_names)
    l = len(supervised)
    r_train = [0, round(split[0] * l)]
    r_val = [r_train[1], r_train[1] + round(split[1] * l)]
    r_test = [r_val[1], r_val[1] + round(split[2] * l)]
    
    
    train_set = supervised.iloc[r_train[0]:r_train[1]]
    val_set = supervised.iloc[r_val[0]:r_val[1]]
    test_set = supervised.iloc[r_test[0]:]
    
    return  train_set, val_set, test_set, scaler
    
def df_to_dataset(df, n_samples, n_out, n_features):
    features = df.to_numpy()[:, :n_samples*n_features]
    targets = df.to_numpy()[:, :n_samples*n_features:]
    ds = InertiaDataset(features, targets, n_samples, n_features, n_out, n_features)
    return ds    
    
class InertiaDataset(Dataset):
    
    def __init__(self, input_data, target_data, sequence_length=24,
                 in_features=6, output_length=1, target_features=6):
        self.input_data = input_data
        self.target_data = target_data
        self.sequence_length = sequence_length
        self.output_length = output_length
        self.in_features = in_features
        self.target_features = target_features
    
    def __len__(self):
        # The length of the dataset is determined by how long the sequences are 
        return self.input_data.shape[0]
    
    def __getitem__(self, idx):
        s_len = self.sequence_length
        o_len = self.output_length
        o_feat = self.target_features
        n_feat = self.in_features
        input_sequence = torch.zeros(s_len, n_feat)
        target_sequence = torch.zeros(o_len, o_feat)
        for i in range(self.in_features):
            input_sequence[:, i] = tensor(self.input_data[idx, i * s_len:
                                                          (i + 1) * s_len])
        for i in range(self.target_features):
            target_sequence[:, i] = tensor(self.target_data[idx, i*o_len:(i + 1) * o_len])
        return input_sequence, target_sequence


class MultilayerLSTM(nn.Module):
    
    def __init__(self, n_features=6, n_hidden=128, n_layers=2, drpout_lvl=0.2):
        super(MultilayerLSTM, self).__init__()
        
class Stacked_LSTM(nn.Module):
    # Attempt with LSTMCell
    def __init__(self, n_inputs=6, hidden1=128, hidden2=128, dropout_lvl=0.1):
        
        super(Stacked_LSTM, self).__init__()
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.lstm1 = nn.LSTMCell(n_inputs, self.hidden1)
        self.lstm2 = nn.LSTMCell(self.hidden1, self.hidden2)
        self.dropout = nn.Dropout(dropout_lvl)
        self.BN1 = nn.BatchNorm1d(hidden1)
        self.BN2 = nn.BatchNorm1d(hidden2)
        self.linear = nn.linear(self.hidden2, n_inputs) # To be able to forecast without measured input need to predict all inputs
    
    def forward(self, input_seq, future_preds=0):
        outputs, n_batch = [], input_seq.size(0)
        h_t = torch.zeros(n_batch, self.hidden1, dtype=torch.float32)
        c_t = torch.zeros(n_batch, self.hidden1, dtype=torch.float32)
        h_t2 = torch.zeros(n_batch, self.hidden2, dtype=torch.float32)
        c_t2 = torch.zeros(n_batch, self.hidden2, dtype=torch.float32)
        
        # torch.nn.init.xavier_normal_(h_t)
        # torch.nn.init.xavier_normal_(c_t)
        # torch.nn.init.xavier_normal_(h_t2)
        # torch.nn.init.xavier_normal_(c_t2)
        for input_t in input_seq.split(1, dim=1):
            input_t = input_t.squeeze(dim=1) # Because dataloader gives batch first have to do it like this
            h_t, c_t = self.lstm1(input_t, (h_t,c_t))
            h_t, c_t = self.dropout(h_t), self.dropout(c_t)
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t2, c_t2 = self.dropout(h_t2), self.dropout(c_t2)
            output = self.linear(h_t2)
            outputs.append(output)
        
        for i in range(future_preds):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)
        return outputs

def training_loop(model, train_loader, val_loader, optimizer, criterion,
                  learning_rate = 1e-3, n_epochs=10):
    training_loss = []
    validation_loss =[]
    for epoch in range(n_epochs):
        model.train()
        training_loss = 0
        # Training loop
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        
        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        
        

