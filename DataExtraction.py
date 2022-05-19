# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:21:04 2022

@author: arvidro
"""

import pandas as pd
import numpy as np
import matplotlib
import copy as cp
from functools import reduce 
from matplotlib import pyplot as plt


production_data_path = "Data/ProductionData/"
inertia_data_path = "Data/InertiaData/"
suffix = ".csv"

countries = ("Sweden", "Norway", "Denmark", "Finland")

years = ('2022', '2021', '2020', '2019', '2018', '2017', '2016')

months = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12')

keys = ['Biomass[MW]', 'FossilGas[MW]', 'FossilHardcoal[MW]', 'FossilOil[MW]',
       'FossilPeat[MW]', 'HydroPumpedStorage[MW]',
       'HydroRun-of-riverandpoundage[MW]', 'HydroWaterReservoir[MW]',
       'Nuclear[MW]', 'Other[MW]', 'Otherrenewable[MW]', 'Solar[MW]',
       'Waste[MW]', 'WindOffshore[MW]', 'WindOnshore[MW]']

wind_keys = ('WindOffshore[MW]', 'WindOnshore[MW]')
conventional_keys = ('Biomass[MW]', 'FossilGas[MW]', 'FossilHardcoal[MW]', 'FossilOil[MW]',
       'FossilPeat[MW]', 'HydroPumpedStorage[MW]',
       'HydroRun-of-riverandpoundage[MW]', 'HydroWaterReservoir[MW]',
       'Nuclear[MW]', 'Other[MW]','Waste[MW]',)

training_labels = ['Inertia[GW]',
       'Total_production', 'Share_Wind', 'Share_Conv']

def get_datasets(countries=countries, years=years,path=production_data_path, suffix=suffix):
    
    # reading csv files into dataframes
    dataframes = {year:{country:pd.read_csv(path + country + year + suffix)
                           for country in countries} for year in years}
    # Cleaning the data a bit
    for year in dataframes.values(): 
        for df in year.values():
            col_subset = []
            for key in df.keys():
                new_key = key.replace(" ", "")
                new_key = new_key.replace("-ActualAggregated", '')
                
                df.rename(columns={key:new_key}, inplace=True)
                if '[MW]' in new_key:
                    df[new_key] = pd.to_numeric(df[new_key], errors='coerce')
                    col_subset.append(new_key)
            df.drop(labels='Area', axis=1,inplace=True)
            #df.fillna(value=0, inplace=True)
            df.dropna(axis=0, how='all', inplace=True, subset=col_subset) # Drop dates with no data
            df.dropna(axis=1, how='all', inplace=True) # drop columns with no values
            df['MTU'] = pd.to_datetime(df['MTU'].str.split('-').str[0],dayfirst=True) # Convert timestamp str to datetime
            df.rename(columns={'MTU':'Time'}, inplace=True)
        
        # df['MTU'].apply(str.split, args=('-'))
        # df['MTU'].apply(list.pop)
        # df['MTU'].apply(str)
        # df['MTU'].apply(str.replace, args=('[',''))
        # df['MTU'].apply(str.replace, args=(']',''))
        # df['MTU'].apply(pd.to_datetime)
    return dataframes

def get_tot_by_year(year, data):
    # Takes a dictionary on the form output by get_datasets
    
    country_data = data[str(year)]
    
    dfs = [country_data[key] for key in country_data.keys()]
    
    tot = reduce(lambda a, b: a.set_index('Time').add(b.set_index('Time'), fill_value=0).reset_index(), dfs)
    return tot

#df = get_datasets()  
def get_inertia_data(year, path=inertia_data_path):
     df = []
     drop_tags = ["Start time UTC", "End time UTC", "End time UTC+02:00"]
     for month in months:
         try:
             path = inertia_data_path + "Inertia" + month + year + suffix
             df.append(pd.read_csv(inertia_data_path + "Inertia" + month + year + suffix))
         except FileNotFoundError:
             print("Tried to open non-existent file, probably don't have data for {}{}".format(month,year))
             print("Path used: {}".format(path))
             break
     for month in df:
         month.drop(labels=drop_tags, axis=1, inplace=True)
         month.rename(columns={"Start time UTC+02:00":"Time", 
                               'Kinetic energy of the Nordic power system - real time data':"Inertia[GW]"}, inplace=True)
     df = pd.concat(df,axis=0)
     df.reset_index()
     df["Time"] = pd.to_datetime(df["Time"], dayfirst=False)
     return df


def get_full_year(year, inertia_frame):
    prod = pd.read_csv(production_data_path + "Nordic{}".format(year) + suffix)
    prod.drop(labels=["Unnamed: 0"],inplace=True, axis=1)
    prod.rename(columns={'MTU':'Time'},inplace=True)
    prod['Time'] = pd.to_datetime(prod['Time'],dayfirst=False)
    prod_inert = prod.merge(inertia_frame, how='outer', on='Time')
    #Drop rows with all except three nan. Corresponds to removing rows where no production data is available
    prod_inert.dropna(how='all',thresh=3,inplace=True)  
    return prod_inert

def interpolate_missing(data):
    # Assumes data is a pandas DataFrame
    pass
    
def find_na(df, key):
    
    return df[key].isna().sum()



def scatter_year(df, year, label):
    
    wind = df[label].to_numpy()
    inertia = df["Inertia[GW]"].to_numpy()
    
    normal_w = reject_outliers(wind)
    wind = wind[normal_w]
    inertia = inertia[normal_w]
    normal_i = reject_outliers(inertia)
    wind = wind[normal_i]
    inertia = inertia[normal_i]
    wind = wind - np.nanmean(wind)
    wind = wind/np.nanstd(wind)
    
    inertia = inertia - np.nanmean(inertia)
    inertia = inertia/np.nanstd(inertia)
    
    plt.figure()
    plt.scatter(inertia, wind)
    plt.xlabel("Normalised Inertia")
    plt.ylabel("Normalised {} production".format(label))

    wind_title = "{} vs Inertia{}".format(label, year)
    plt.title(wind_title)
    plt.show()
    
def reject_outliers(data, m=3):
    normal_inds = [i for i,val in enumerate(abs(data - np.nanmean(data)) < m * np.nanstd(data)) if val]
    
    return normal_inds

def get_total_production(df, keys=keys):
    df["Total_production"] = 0
    for key in keys:
        try:
            df["Total_production"] = df["Total_production"].add(df[key], fill_value=0)
        except KeyError:
            print(key + " Not in DataFrame")
    df["Share_Wind"] = 0
    for key in wind_keys:
        df["Share_Wind"] = df["Share_Wind"].add(df[key], fill_value=0)
    df["Share_Wind"] = df["Share_Wind"].div(df["Total_production"], fill_value=0)
    
    df["Share_Conv"] = 0 # Share conventional generation
    for key in conventional_keys:
        try:
            df["Share_Conv"] = df["Share_Conv"].add(df[key], fill_value=0)
        except KeyError:
            print(key + " Not in DataFrame")
    df["Share_Conv"] = df["Share_Conv"].div(df["Total_production"], fill_value=0)
#tot = get_tot_by_year(2022, df)

        #print(year.keys())
#dataframes["Sweden"]["2022"].plot()

def get_training_data(df, labels=training_labels):
    try:
        df.set_index("Time", inplace=True)
    except KeyError:
        print("Index already set to time")
    df.index = pd.to_datetime(df.index)
    training_df = df.filter(labels, axis=1)
    
    training_df["month"] = training_df.index.month
    training_df["day"] = training_df.index.day_of_year
    training_df["hour"] = training_df.index.hour
    # for label in labels:
    #     mean = training_df[label].mean()
    #     std = training_df[label].std()
    #     training_df[label] = (training_df[label] - mean)/std
    #     training_df[label + "_mean"] = mean
    #     training_df[label + "_std"] = std
    return training_df
# def add_countries(country1, country2)

#     df_sum = pd.DataFrame()

def load_dataset(year):
    df = pd.read_csv("Data/ProdInertia{}.csv".format(year))
    df["Time"] = pd.to_datetime(df["Time"], yearfirst=True)
    #df.drop("Unnamed: 0", axis=1, inplace=True)
    return df

years = np.arange(2016, 2023)

data = [load_dataset for year in years]

