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


def bin_data(df):
    '''
    This function bins specific continuous features and creates new columns for them. This will be useful when creating visuals.
    
    Parameters:
    df = data
    '''
    
    #bin data
    bin_bound = [0, 1.9, 2.9, 3.9, 4.9, 5.9, 18]

    bin_labels = [1, 2, 3, 4, 5, 18]

    df['bathrooms_bin'] = pd.cut(df['bathrooms'], bins = bin_bound, labels = bin_labels)