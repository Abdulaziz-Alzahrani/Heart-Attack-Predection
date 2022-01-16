import os, sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data():
    """
    load data from database and returns the dataframe.
    Parameters: None.
    Return: pandas.DataFrame/ data read from datasets.heart SQLite db table.
    """
    engine = create_engine('sqlite:///dataset.db')
    df = pd.read_sql_table('heart', engine)

    return df

