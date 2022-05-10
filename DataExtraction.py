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
data_path = "Data/"
suffix = ".csv"

countries = ("Sweden", "Norway", "Denmark", "Finland")

years = ('2022', '2021', '2020', '2019', '2018', '2017', '2016')



def get_datasets(countries=countries, years=years,path=data_path, suffix=suffix):
    
    # reading csv files into dataframes
    dataframes = {year:{country:pd.read_csv(data_path + country + year + suffix)
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
            df['MTU'] = pd.to_datetime(df['MTU'].str.split('-').str[0]) # Convert timestamp str to datetime
        
        
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
    
    tot = reduce(lambda a, b: a.set_index('MTU').add(b.set_index('MTU'), fill_value=0), dfs)
    return tot

df = get_datasets()       

tot = get_tot_by_year(2022, df)

        #print(year.keys())
#dataframes["Sweden"]["2022"].plot()


# def add_countries(country1, country2)

#     df_sum = pd.DataFrame()


