import os, sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# Dataset link: https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset

def load_data():
    '''
    read the data from "heart.csv".
    Parameters: None.
    Return: pandas.DataFrame/ df containing the data read from "heart.csv".
    '''
    return pd.read_csv("heart.csv")

def data_cleaning(df):
    '''
    fill missing values with the average value of each column and remove duplicates.
    Parameters: pandas.DataFrame.
    Return: None.
    '''
    # loop through the items fill the missing values with avg
    for indx, val in df.isnull().sum().items():
        if val > 0:
            avg = df[indx].mean()
            df[indx].fillna(value=avg, inplace=True)
    
    # check if there is a duplicate value 
    if df.duplicated().sum() > 0:
        df.drop_duplicates(keep='first',inplace=True)

def save_data(df):
    """
    Saves the dataframe into a SQLite database.
    Parameters:
    - df:pandas.Dataframe / the dataframe of cleaned dataset.
    Return: None
    """
    engine = create_engine('sqlite:///datasets.db')
    df.to_sql('heart', engine, index=False, if_exists='replace')  

df = load_data()
print(len(df))
data_cleaning(df)
print(len(df))
save_data(df)