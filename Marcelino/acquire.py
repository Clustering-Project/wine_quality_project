import os 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


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
        random_state=42,
        stratify=target)
    train, validate = train_test_split(
        train_val,
        train_size=0.7,
        random_state=42,
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


def xy_split(df):
    '''
    This function splits your data for modeling. 
    
    Parameter: 
    df = data
    
    Output:
    This function returns subsets of your data. One with all columns except your target variable, and the other with only the target variable.
    '''
    
    return df.drop(columns= 'quality'), df.quality


def dummies(train, val, test):
    '''
    This function applies one hot encoding to all categorical features in your dataset.
    
    Parameters:
    train = train data
    val = val data
    test = test data
    
    Output:
    This function returns your train, val, and test subsets with dummies added.
    '''
    
    train = pd.get_dummies(train)
    
    val = pd.get_dummies(val)
    
    test = pd.get_dummies(test)
    
    return train, val, test


def scale_data(train, val, test, to_scale):
    '''
    This function scales all continuous numerical features in your train, val, and test subsets.
    
    Parameters:
    train = train data
    val = val data
    test = test data
    to_scale = features to scale
    
    Output:
    This function returns scaled features added to your data.
    '''
    #make copies for scaling
    train_scaled = train.copy()
    validate_scaled = val.copy()
    test_scaled = test.copy()

    #make the thing
    scaler = MinMaxScaler()

    #fit the thing
    scaler.fit(train[to_scale])

    #use the thing
    train_scaled[to_scale] = scaler.transform(train[to_scale])
    validate_scaled[to_scale] = scaler.transform(val[to_scale])
    test_scaled[to_scale] = scaler.transform(test[to_scale])
    
    return train_scaled, validate_scaled, test_scaled
