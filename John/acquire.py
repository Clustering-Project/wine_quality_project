# Import the pandas library
import pandas as pd

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
    
    return df



