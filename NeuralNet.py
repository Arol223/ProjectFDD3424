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
from matplotlib import pyplot as plt

input_names = ("Inertia[GW]", "Total_production", "Share_Wind", "Share_Conv","day", "hour")
output_names = ("Inertia[GW]","Total_production", "Share_Wind", "Share_Conv","day", "hour")
datafile_path = "Data/CleanedTrainingset16-22_new_.csv"
scale_columns = ["Inertia[GW]","Total_production", "Share_Wind", "Share_Conv",
                 "day", "hour"]
norm_columns = ["Inertia[GW]","Total_production", "Share_Wind", "Share_Conv"]

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

def hrs_to_time_of_day(hours):
    return np.sin(hours*2*np.pi/24)

def mnths_to_time_of_year(mnths):
    return np.sin(mnths*2*np.pi/12)

def days_to_time_of_year(days):
    return np.sin(days*2*np.pi/365)

def get_split_sets(
        split=(0.8,0.05,0.15), scaler=MinMaxScaler(feature_range=(-1,1)), 
        normalizer=StandardScaler(), data_path=datafile_path,
        scale_columns=scale_columns, input_names=input_names,
        output_names=output_names, n_samples=24, n_out=1
             ):
    # Perform a train-val-test split, scale data and return datasets
    dataset = pd.read_csv(datafile_path, index_col=0)
    dataset.drop("month", axis=1, inplace=True)
    dataset["day"] = days_to_time_of_year(dataset["day"])
    
    dataset["hour"] = hrs_to_time_of_day(dataset["hour"])
    dataset[scale_columns] = scaler.fit_transform(dataset[scale_columns])
    dataset[norm_columns] = normalizer.fit_transform(dataset[norm_columns])
    
    supervised = to_supervised(dataset, n_samples, n_out, input_names, output_names)
    l = len(supervised)
    r_train = [0, round(split[0] * l)]
    r_val = [r_train[1], r_train[1] + round(split[1] * l)]
    r_test = [r_val[1], r_val[1] + round(split[2] * l)]
    
    
    train_set = supervised.iloc[r_train[0]:r_train[1]]
    val_set = supervised.iloc[r_val[0]:r_val[1]]
    test_set = supervised.iloc[r_test[0]:]
    
    return  train_set, val_set, test_set, scaler, normalizer
    
def df_to_dataset(df, n_samples, n_out, n_features):
    features = df.to_numpy()[:, :n_samples*n_features]
    targets = df.to_numpy()[:, n_samples*n_features:]
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
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.drpout_lvl = drpout_lvl
        self.dropout = nn.Dropout(drpout_lvl)
    
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            batch_first=True,
            num_layers=n_layers,
            dropout=drpout_lvl
            )
        self.linear = nn.Linear(
            in_features=self.n_hidden,
            out_features=self.n_features
            )
    def forward(self, X):
        batch_size = X.shape[0]
        h0 = torch.zeros(self.n_layers, batch_size, self.n_hidden).requires_grad_()
        c0 = torch.zeros(self.n_layers, batch_size, self.n_hidden).requires_grad_()
    
        output, (hn, cn) = self.lstm(X, (h0, c0))
        
        out = self.dropout(hn[-1,:,:])
        out = self.linear(hn[-1,:,:]) # First dim of hn is num_layers. Want only the last hidden state.
        return out
        
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
def train_model(data_loader, model, criterion, optimizer, mode="norm"):
    n_batches = len(data_loader)
    tot_loss = 0
    model.train()
    if mode == "norm":
        for i, (X, y) in enumerate(data_loader):
            out = model(X)
            loss = criterion(out, y.squeeze(dim=1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
    elif mode == "LBFGS":
        X, y = next(iter(data_loader)) # LBFGS uses full batch training
        def closure():
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y.squeeze(dim=1))
            loss.backward()
            return loss
        optimizer.step(closure)
        
    avg_loss = tot_loss / n_batches
    return avg_loss

def validate_model(data_loader, model, criterion):
    
    n_batches = len(data_loader)
    tot_loss = 0
    
    model.eval()
    
    with torch.no_grad():
        for X, y in data_loader:
            out = model(X)
            tot_loss += criterion(out, y.squeeze(dim=1)).item()
    avg_loss = tot_loss / n_batches
    
    return avg_loss 

def training_loop(model, train_loader, val_loader, optimizer, criterion,
                  n_epochs=10, early_stopping=True, mode="norm"):
    training_loss = []
    validation_loss =[]
    final_epoch = 0
    for epoch in range(n_epochs):
        print("Starting {}th epoch".format(epoch + 1))
        loss = train_model(train_loader, model, criterion, optimizer,
                           mode=mode)
        training_loss.append(loss)
        val_loss = validate_model(val_loader, model, criterion)
        validation_loss.append(val_loss)
        final_epoch = epoch
        # model.train()
        # training_loss = 0
        # # Training loop
        # for features, targets in train_loader:
        #     optimizer.zero_grad()
        #     outputs = model(features)
        #     loss = criterion(outputs, targets)
        #     loss.backward()
        #     optimizer.step()
        #     training_loss += loss.item()
        
        # # Validation loop
        # model.eval()
        # val_loss = 0
        # with torch.no_grad():
        #     for features, targets in val_loader:
        #         outputs = model(features)
        #         loss = criterion(outputs, targets)
        #         val_loss += loss.item()
        if early_stopping:
            if len(validation_loss) >= 5:
                median = np.median(validation_loss[-5:])
                
                if validation_loss[epoch] > median:
                    print("Early stopping because validation loss increased")
                    print("I trained for {} epochs".format(epoch + 1))
                    break 
        
    return training_loss, validation_loss, final_epoch
            
def predict(test_loader, model):
    
    output_pred = tensor([])
    out_true = tensor([])
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            y_hat = model(X)
            output_pred = torch.cat((output_pred, y_hat), 0)
            out_true = torch.cat((out_true, y), 0)
    return output_pred, out_true

def load_model(path, n_layers, n_hidden, drpout_lvl):
    model = MultilayerLSTM(n_layers=n_layers, n_hidden=n_hidden, drpout_lvl=drpout_lvl)
    model.load_state_dict(torch.load(path))
    return model
def save_model(model, name):
    torch.save(model.state_dict(), "Models/" + name + ".pt")
    
def test_model(test_loader, model, out_features=output_names):
    yhat_col = "Forecasted Inertia"
    y_col = "Actual Inertia"
    test_df = pd.DataFrame()
    y_hat, y = predict(test_loader, model)
    y = y.squeeze(dim=1)
    yh_df = pd.DataFrame()
    y_df = pd.DataFrame()
    for i, name in enumerate(out_features):
        yh_df["Predicted " + name] = y_hat[:, i].numpy()
        y_df["Actual " + name] = y[:, i].numpy()
    return yh_df, y_df

def prepare_evaluation(
        model_name, model_params, test_name=None, plot_range=24*30,
        set_params=[24, 1, 6]
                       ):
    if test_name is None:
        test_name = model_name
    _, _, test, scaler, normalizer = get_split_sets()
    model_path = "Models/" + model_name + ".pt"
    test_set = df_to_dataset(test, *set_params)
    model = load_model(model_path, **model_params)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    yh, y = test_model(test_loader, model)
    yh.index = test.index
    y.index = test.index
    actual_cols = ["Actual " + x for x in norm_columns]
    pred_cols = ["Predicted " + x for x in norm_columns]
    y[actual_cols] = normalizer.inverse_transform(y[actual_cols])
    yh[pred_cols] = normalizer.inverse_transform(yh[pred_cols])
    y.iloc[:] = scaler.inverse_transform(y.iloc[:])
    yh.iloc[:] = scaler.inverse_transform(yh.iloc[:])
    total_test_df = pd.concat([yh,y], axis=1)
    total_test_df.to_csv("Predictions/" + test_name + ".csv")
    sub = total_test_df.iloc[-1 - plot_range:-1]
    return total_test_df, sub
    
def main():
    seed = torch.initial_seed()
    model_params = {"n_hidden":256, "drpout_lvl":0.2, "n_layers":2}
    train, val, test, scaler, normalizer = get_split_sets(n_samples=24)
    test = df_to_dataset(test, 24, 1, 6)
    train = df_to_dataset(train, 24, 1, 6)
    val = df_to_dataset(val, 24, 1, 6)
    
    #BATCH_SIZE_TRAIN = 256
    BATCH_SIZE_TRAIN = len(train) # Use this for LBFGSar
    n_epochs = 5
    early_stopping = False
    train_loader = DataLoader(train, BATCH_SIZE_TRAIN, shuffle=False)
    val_loader = DataLoader(val, batch_size=8, shuffle=False)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)
    learning_rate = 5e-4
    model = MultilayerLSTM(**model_params)
    
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.LBFGS(model.parameters())
    criterion = nn.MSELoss()
    
    training_loss, validation_loss, epochs = training_loop(
        model, train_loader, val_loader, optimizer, criterion,
        n_epochs=n_epochs, early_stopping=early_stopping, mode='LBFGS'
        )
    mdl_param_str = str(model_params).replace(" ", '').replace(':', '_').replace('{','').replace('}','')
    model_name = ("Model_" + mdl_param_str +"_"+ "_Batch_size_"
               + str(BATCH_SIZE_TRAIN) + "Epochs_" + str(epochs)
               + "_Seed_" + str(seed))
    save_model(
        model, model_name
               )    
    
    
    return training_loss, validation_loss, model_name, model_params
    
if __name__=='__main__':
    training_loss, validation_loss, model_name, model_params = main()

# model_name = "Model_'n_hidden'_256,'drpout_lvl'_0.2,'n_layers'_2__Batch_size_256Epochs_8_Seed_167566065688400"
# model_params = {'n_hidden': 256, 'drpout_lvl': 0.2, 'n_layers': 2}
# tot, sub = prepare_evaluation(model_name, model_params)

# model = load_model("Models/FirstTrainedModel.pt")
#train, val, test, scaler = get_split_sets()

# test_set = df_to_dataset(test, 24, 1, 6)

# test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

# yh, y = test_model(test_loader, model)

         


