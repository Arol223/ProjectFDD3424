# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:30:37 2022

@author: arvidro
"""

import torch.autograd as autograd
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.jit import script, trace
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

input_names = ("Inertia[GW]", "Total_production", "Share_Wind", "Share_Conv","month", "hour")
output_names = ("Inertia[GW]")
datafile_path = "Data/CleanedTrainingset16-22.csv"
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
    output_df = pd.concat(cols, axis=1)
    output_df.columns = names
    output_df.dropna(inplace=True)
    return output_df

class InertiaDataset(Dataset):
    
    def __init__(self, inertia_data, sequence_length=24, output_length=5):
        self.data = inertia_data
        self.sequence_length = sequence_length
        self.output_length = output_length
    
    def __len__(self):
        # The length of the dataset is determined by how long the sequences are 
        return self.data.shape[0] - self.sequence_length
    
    def __getitem__(self, index):
        return self.data[index:index+self.sequence_length, :]

class Stacked_LSTM(nn.Module):
    
    def __init__(self, n_inputs, hidden1=100, hidden2=100):
        
        super(Stacked_LSTM, self).__init__()
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.lstm1 = nn.LSTMCell(n_inputs, self.hidden1)
        self.lstm2 = nn.LSTMCell(self.hidden1, self.hidden2)
        self.dropout = nn.Dropout(0.25)
        
        
        