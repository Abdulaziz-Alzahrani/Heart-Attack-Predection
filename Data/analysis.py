import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine



def load_data():
    """
    load data from database and returns the dataframe.
    Parameters: None.
    Return: pandas.DataFrame/ data read from datasets.heart SQLite db table.
    """
    engine = create_engine('sqlite:///Model/datasets.db')
    df = pd.read_sql_table('heart', engine)

    return df

def countplot_age_sex(df):
    """
    plot age to output(chance of heart attack) and sex to output relationship.
    Parameters: pandas.DataFrame/ data read from datasets.heart SQLite db table.
    Return: None.
    """
    data = df.loc[:,["age","sex", "output"]]
    for c in ["age","sex"]:
        sns.countplot(x=c,data=data,hue="output")
        plt.title(c)
        plt.show()

def main():
    df = load_data()
    countplot_age_sex(df)

if __name__ == '__main__':
    main()