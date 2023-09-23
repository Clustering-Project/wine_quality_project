# Import the pandas library
import pandas as pd
from sklearn.model_selection import train_test_split

def acquire_wine():
    # Read the 'winequality-red.csv' file and store the data in a DataFrame object named df_red
    df_red = pd.read_csv('winequality-red.csv')

    # Read the 'winequality-white.csv' file and store the data in a DataFrame object named df_white
    df_white = pd.read_csv('winequality-white.csv')

    # Assign the value "red" to the 'type' column in the df_red dataframe
    df_red['type'] = "red"

    # Assign the value "white" to the 'type' column in the df_white dataframe
    df_white['type'] = "white"

    # Concatenate the df_red and df_white dataframes vertically, ignoring the original indices, and assign the result to the df dataframe
    df = pd.concat([df_red, df_white], ignore_index=True, axis=0)
    
    df.columns = [col.lower().replace(' ', '_').replace('.', '_') for col in df.columns]
    
    return df

# def prepare_wine(df):
#     df.columns = [col.lower().replace(' ', '_').replace('.', '_') for col in df.columns]
#     return df

def wine_train_val_test(df, seed = 42):
    # split 70/30
    train, val_test = train_test_split(df, train_size = 0.7,
                                       random_state = seed)
    # split the remainder 30% 50/50
    val, test = train_test_split(val_test, train_size = 0.5,
                                 random_state = seed)
    
    return train, val, test


def wrangle_wine():
    # get the initial df
    df = acquire_wine()
    # split into train val test
    train, val, test = wine_train_val_test(df, seed = 42)
    
    return train, val, test


def model_pipeline(df):
    
    train, val, test = wine_train_val_test(df)
    
    #train, val, test = scale_train_val_test(train, val, test)
    train = hot_encode(train)
    val = hot_encode(val)
    test = hot_encode(test)
    
    X_train, y_train = xy_split(train)
    X_val, y_val = xy_split(val)
    X_test, y_test = xy_split(test)
    return train, val, test, X_train, y_train, X_val, y_val, X_test, y_test