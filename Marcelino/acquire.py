import os 
import pandas as pd
from sklearn.model_selection import train_test_split


def wrangle_wine() -> pd.DataFrame:
    
    if os.path.exists('wine.csv'):
        
        df = pd.read_csv('wine.csv')
        
    else:
  
        df_red = pd.read_csv('winequality-red.csv')
        df_white = pd.read_csv('winequality-white.csv')
        df_red['type'] = 'red'
        df_white['type'] = 'white'
        df = pd.concat([df_red, df_white], ignore_index=True, axis=0)
        df.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)  
        df.to_csv('wine.csv', index=False)
        df = pd.read_csv('wine.csv')
        
    return df


def split_data(df, target=None) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    '''
    split_data will split data into train, validate, and test sets
    
    if a discrete target is in the data set, it may be specified
    with the target kwarg (Default None)
    
    return: three pandas DataFrames
    '''
    train_val, test = train_test_split(
        df, 
        train_size=0.8, 
        random_state=1349,
        stratify=target)
    train, validate = train_test_split(
        train_val,
        train_size=0.7,
        random_state=1349,
        stratify=target)
    return train, validate, test


def get_continuous_feats(df) -> list:
    '''
    find all continuous numerical features
    
    return: list of column names (strings)
    '''
    num_cols = []
    num_df = df.select_dtypes('number')
    for col in num_df:
        if num_df[col].nunique() > 20:
            num_cols.append(col)
    return num_cols