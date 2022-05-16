# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:30:37 2022

@author: arvidro
"""
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
output_names = ("Inertia[GW]",)
datafile_path = "Data/CleanedTrainingset16-22.csv"
def to_supervised(df, n_in, n_out, input_names, output_names):
    input_cols = []
    target_cols = []
    inputs_names = []
    target_names = []
    for name in input_names:
        for i in range(n_in, 0, -1):
            input_cols.append(df[name].shift(i))
            inputs_names.append(name + "(t-{})".format(i))
    for name in output_names:
        for i in range(0, n_out):
            target_cols.append(df[name].shift(-i))
            if i == 0:
                target_names.append(name + "(t)")
            else:
                target_names.append(name + "(t+{})".format(i))
    input_df = pd.concat(input_cols, axis=1)
    input_df.columns = inputs_names
    input_df.dropna(inplace=True)
    target_df = pd.concat(target_cols, axis=1)
    target_df.columns = target_names
    target_df.dropna(inplace=True)
    return input_df, target_df

class InertiaDataset(Dataset):
    
    def __init__(self, input_data, target_data, sequence_length=24,
                 in_features=6, output_length=1, target_features=1):
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
        n_feat = self.in_features
        input_sequence = torch.zeros(s_len, n_feat)
        target = tensor(self.target_data[idx]) # Try to predict one step at a time
        for i in range(self.in_features):
            input_sequence[:, i] = tensor(self.input_data[idx, i * s_len:
                                                          (i + 1) * s_len])
        return input_sequence, target

class Stacked_LSTM(nn.Module):
    
    def __init__(self, n_inputs, hidden1=100, hidden2=100, dropout_lvl=0.1):
        
        super(Stacked_LSTM, self).__init__()
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.lstm1 = nn.LSTMCell(n_inputs, self.hidden1)
        self.lstm2 = nn.LSTMCell(self.hidden1, self.hidden2)
        self.dropout = nn.Dropout(dropout_lvl)
        self.linear = nn.linear(self.hidden2, 1)
    
    def forward(self, input_seq, future_preds=0):
        outputs, n_samples = [], input_seq.size(0)
        h_t = torch.zeros(n_samples, self.hidden1, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.hidden1, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.hidden2, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.hidden2, dtype=torch.float32)
        
        torch.nn.init.xavier_normal_(h_t)
        torch.nn.init.xavier_normal_(c_t)
        torch.nn.init.xavier_normal_(h_t2)
        torch.nn.init.xavixavier_normal_(c_t2)
        for input_t in input_seq.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t,c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
            
        
#data = pd.read_csv("Data/CleanedTrainingset16-22.csv", index_col=0)
        