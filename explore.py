import pandas as pd

from scipy import stats
from scipy.stats import chi2_contingency

from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


from wrangle import wine_train_val_test, acquire_wine

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def spearmanr_test(df,col_name):
    # Perform spearmanr test
    spearman_corr, p_value = stats.spearmanr(df['quality'], df[col_name])

    # Interpret the results
    alpha = 0.05  # Set your desired significance level
    # Print results
    if p_value < alpha:
        print(f"There is a statistically significant Spearman's rank correlation (p-value = {p_value:.4f}, corr = {spearman_corr:.4f}).")
    else:
        print(f"There is no statistically significant Spearman's rank correlation (p-value = {p_value:.4f}, corr = {spearman_corr:.4f}).")

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def perform_chi2_test(df, variable1, variable2):
    """
    Perform the Chi-Squared Test of Independence and print the results.

    Parameters:
    - data: DataFrame containing the two categorical variables.
    - variable1: Name of the first categorical variable.
    - variable2: Name of the second categorical variable.

    Returns:
    - None (results are printed).
    """
    # Create a contingency table
    df = cluster_alc_dens(df)
        
    # Define custom labels for "quality"
    bins_q = [3, 5, 6, 9]
    labels_q = ['Low', 'Med', 'High']

    # Create a new column "quality_bins" to store the bin labels
    df['quality_bins'] = pd.cut(df['quality'], bins=bins_q, labels=labels_q)

    contingency_table = pd.crosstab(df[variable1], df[variable2])

    # Perform the Chi-Squared Test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Print the results
    print("Chi-Squared Test of Independence:")
    print(f"Chi-Squared Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    # print(f"Degrees of Freedom: {dof}")
    # print("Expected Frequencies:")
    # expected_table = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
    
    # print(expected_table)

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def kbest_features(df, col_name, k=2):
    """
    Selects the top k best features for regression from a DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame containing features and the target column.
        col_name (str): The name of the target column.
        k (int): The number of top features to select (default is 2).

    Returns:
        selected_df (DataFrame): A DataFrame with two columns: 'Column Name' and 'Score'.
            'Column Name' contains the column names of the selected features.

    Example:
        selected_features = select_k_best_features(your_dataframe, 'value', k=2)
    """
    # Create X and y
    X = df.drop(columns=[col_name])  # Remove the target column
    y = df[col_name]
    
    # Filter X to keep only columns that can be converted to float
    X = X.select_dtypes(include=['number', 'float'])  # Keep numeric and float columns
    
    # Initialize SelectKBest with f_regression
    skb = SelectKBest(f_regression, k=k)
    
    # Fit SelectKBest on X and y
    skb.fit(X, y)
    
    # Get the mask of selected features
    skb_mask = skb.get_support()
    
    # Get the scores for all features
    feature_scores = skb.scores_
    
    # Create a DataFrame with column names and scores
    selected_df = pd.DataFrame({
        'Kbest': X.columns[skb_mask],
        'Score': feature_scores[skb_mask]
    })
    
    # Sort the DataFrame by score in descending order
    selected_df = selected_df.sort_values(by='Score', ascending=False)
    
    selected_df.reset_index(drop=True, inplace=True)
    
    selected_df.drop('Score', axis=1, inplace=True)
    
    return selected_df

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def rfe_features(df, col_name, n_features=3):
    """
    Selects a specified number of features from a DataFrame using Linear Regression and RFE.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the features and target variable.
        col_name (str): The name of the target column.
        n_features (int): The number of features to select (default is 3).

    Returns:
        pd.Series: A Series containing the selected feature names.
    """
    # Drop the target column from the DataFrame
    X = df.drop(columns=[col_name])
    
    # Select columns with numeric or float data types
    X = X.select_dtypes(include=['number', 'float'])
    
    # Extract the target variable
    y = df[col_name]
    
    # Initialize a Linear Regression model
    lm = LinearRegression()
    
    # Initialize RFE (Recursive Feature Elimination) with the specified number of features to select
    rfe = RFE(lm, n_features_to_select=n_features)
    
    # Fit RFE on the data
    rfe.fit(X, y)
    
    # Get the mask of selected features
    rfe_mask = rfe.get_support()
    
    # Get the column names of selected features
    selected_features = X.columns[rfe_mask]
    
    # Create a DataFrame to present the selected features
    selected_df = pd.DataFrame({'RFE': selected_features})
    
    return selected_df

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def lasso_features(df, col_name, k=2):
    """
    Performs LASSO feature selection to select the top k features for regression from a DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame containing features and the target column.
        col_name (str): The name of the target column.
        k (int): The number of top features to select (default is 2).

    Returns:
        selected_df (DataFrame): A DataFrame with two columns: 'Column Name' and 'Coefficient'.
            'Column Name' contains the column names of the selected features.
            'Coefficient' contains the corresponding LASSO regression coefficients.

    Example:
        selected_features = lasso_feature_selection(your_dataframe, 'value', k=2)
    """
    # Create X and y
    X = df.drop(columns=[col_name])  # Remove the target column
    y = df[col_name]
    
    # Filter X to keep only numeric and float columns
    X = X.select_dtypes(include=['number', 'float'])
    
    # Initialize LASSO regression with alpha=1.0 (adjust as needed)
    lasso = Lasso(alpha=0.5, max_iter=100000)
    
    # Fit LASSO on X and y
    lasso.fit(X, y)
    
    # Create a DataFrame with selected column names and their coefficients
    selected_df = pd.DataFrame({
        'Lasso': X.columns,
        'Coefficient': lasso.coef_
    })
    
    # Sort the DataFrame by absolute coefficient value in descending order
    selected_df['Coefficient'] = abs(selected_df['Coefficient'])
    selected_df = selected_df.sort_values(by='Coefficient', ascending=False)
    
    # Keep the top k features
    selected_df = selected_df.head(k)
    
    selected_df.reset_index(drop=True, inplace=True)
    
    selected_df.drop('Coefficient', axis=1, inplace=True)
    
    return selected_df

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def feature_selections_results(df, col_name, k=2):
    """
    Combine the results of three feature selection functions into a final DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame containing features and the target column.
    col_name (str): The name of the target column.
    k (int, optional): The number of top features to select (default is 2).

    Returns:
    DataFrame: A DataFrame containing the selected features from three different feature selection methods.
    """
    selected_df1 = kbest_features(df, col_name, k)
    selected_df2 = rfe_features(df, col_name, k)
    selected_df3 = lasso_features(df, col_name, k)
    
    final_selected_df = pd.concat([selected_df1, selected_df2, selected_df3], axis=1)
    
    return final_selected_df

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def MinMax_Scaler(train, val, test):
    """
    Apply Min-Max scaling to selected columns of a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the columns to be scaled.

    Returns:
        pd.DataFrame: The DataFrame with specified columns scaled using Min-Max scaling.

    Note:
        - The function applies Min-Max scaling to numeric columns (float or int) in the DataFrame.
        - The 'value' column is excluded from scaling.
        - The selected columns are scaled to the range [0, 1].
    """
    mms = MinMaxScaler()

    # Select columns to scale (excluding 'value')
    to_scale = train.select_dtypes(include=['float', 'int']).columns.tolist()
    to_scale.remove('quality')

    
    # Apply Min-Max scaling to the selected columns
    train[to_scale] = mms.fit_transform(train[to_scale])
    
    val[to_scale] = mms.transform(val[to_scale])
    test [to_scale] = mms.transform(test[to_scale])
    
    return train, val, test

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def MinMax_Scaler_xy(train, val, test):
    """
    Apply Min-Max scaling to selected columns of a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the columns to be scaled.

    Returns:
        pd.DataFrame: The DataFrame with specified columns scaled using Min-Max scaling.

    Note:
        - The function applies Min-Max scaling to numeric columns (float or int) in the DataFrame.
        - The 'value' column is excluded from scaling.
        - The selected columns are scaled to the range [0, 1].
    """
    mms = MinMaxScaler()

    # Select columns to scale (excluding 'value')
    to_scale = train.select_dtypes(include=['float', 'int']).columns.tolist()

    # Apply Min-Max scaling to the selected columns
    train[to_scale] = mms.fit_transform(train[to_scale])
    
    val[to_scale] = mms.transform(val[to_scale])
    test [to_scale] = mms.transform(test[to_scale])
    
    return train, val, test

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def hot_encode(df):
    """
    Perform one-hot encoding on a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing categorical columns to be one-hot encoded.

    Returns:
        pd.DataFrame: The DataFrame with categorical columns converted to one-hot encoded columns.

    Note:
        - The function uses pd.get_dummies to perform one-hot encoding on the specified DataFrame.
        - All categorical columns are one-hot encoded without dropping the first category (drop_first=False).
    """
    # Use pd.get_dummies to one-hot encode the DataFrame
    df = pd.get_dummies(df, drop_first=True)
    
    return df

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def xy_split(df):
    """
    Split the input DataFrame into feature matrix (X) and target vector (y).
    
    Parameters:
    df (DataFrame): The input DataFrame containing features and target.
    
    Returns:
    X (DataFrame): Feature matrix (all columns except 'value').
    y (Series): Target vector (column 'value').
    """
    # Split the dataset into feature columns (X) and target column (y)
    return df.drop(columns=['quality']), df.quality

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def data_pipeline():
    df = acquire_wine()

    df = df[df.density <= 1.01]
    df = df[df.alcohol <= 14.04]
    
    train, val, test = wine_train_val_test(df)
    
    #train, val, test = scale_train_val_test(train, val, test)
    train = hot_encode(train)
    val = hot_encode(val)
    test = hot_encode(test)
    
    X_train, y_train = xy_split(train)
    X_val, y_val = xy_split(val)
    X_test, y_test = xy_split(test)
    return X_train, y_train, X_val, y_val, X_test, y_test


# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def data_pipeline_features():
    df = acquire_wine()
    df = new_feats(df)

    df = df[df.density <= 1.01]
    df = df[df.alcohol <= 14.04]
    
    train, val, test = wine_train_val_test(df)
    
    #train, val, test = scale_train_val_test(train, val, test)
    train = hot_encode(train)
    val = hot_encode(val)
    test = hot_encode(test)
    
    X_train, y_train = xy_split(train)
    X_val, y_val = xy_split(val)
    X_test, y_test = xy_split(test)
    return X_train, y_train, X_val, y_val, X_test, y_test


# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def bravo_pipeline():
    df = acquire_wine()

    df = df[df.density <= 1.01]
    df = df[df.alcohol <= 14.04]
    
    mms = MinMaxScaler()
    # Select columns to scale (excluding 'value')
    to_scale = df.select_dtypes(include=['float', 'int']).columns.tolist()
    to_scale.remove('quality')
    
    
    df[to_scale] = mms.fit_transform(df[to_scale])

    kmeans = KMeans(n_clusters=3, n_init='auto')
    features = df[['alcohol', 'density']]
    kmeans.fit(features)

    df['alc_dens_cluster'] = kmeans.labels_

    train, val, test = wine_train_val_test(df)
    
    #train, val, test = scale_train_val_test(train, val, test)
    train = hot_encode(train)
    val = hot_encode(val)
    test = hot_encode(test)
    
    X_train, y_train = xy_split(train)
    X_val, y_val = xy_split(val)
    X_test, y_test = xy_split(test)
    return X_train, y_train, X_val, y_val, X_test, y_test

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------


def outliers(df):
    df = df[df.density <= 1.01]
    df = df[df.alcohol <= 14.04]
    
    kmeans = KMeans(n_clusters=3, n_init='auto')
    features = df[['alcohol', 'density']]
    kmeans.fit(features)
    df['alc_dens_cluster'] = kmeans.labels_

    return df

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def cluster_alc_dens(df):
    df = df[df.density <= 1.01]
    df = df[df.alcohol <= 14.04]


    mms = MinMaxScaler()

    # Select columns to scale (excluding 'value')
    to_scale = df.select_dtypes(include=['float', 'int']).columns.tolist()
    to_scale.remove('quality')


    # Apply Min-Max scaling to the selected columns
    df[to_scale] = mms.fit_transform(df[to_scale])

    kmeans = KMeans(n_clusters=3, n_init='auto')
    features = df[['alcohol', 'density']]
    kmeans.fit(features)

    df['alc_dens_cluster'] = kmeans.labels_
    
    return df




def cluster_two(df):
    # Create the Total Acidity feature
    df['total_acidity'] = df['fixed_acidity'] + df['volatile_acidity'] + df['citric_acid']
    # Create the Alcohol by Density feature
    df['alcohol_by_density'] = df['alcohol'] * df['density']


    mms = MinMaxScaler()

    # Select columns to scale (excluding 'value')
    to_scale = df.select_dtypes(include=['float', 'int']).columns.tolist()
    to_scale.remove('quality')


    # Apply Min-Max scaling to the selected columns
    df[to_scale] = mms.fit_transform(df[to_scale])


    # Select the features for clustering
    to_cluster = df[['total_acidity', 'alcohol_by_density']]



    # Create the K-Means model
    kmeans = KMeans(n_clusters=3, random_state=42)

    # Fit the model to your data
    kmeans.fit(to_cluster)

    # Get cluster labels for each data point
    cluster_labels = kmeans.labels_

    # Add the cluster labels to your DataFrame
    df['cluster_labels'] = cluster_labels
    
    return df

def new_feats(df):
    # Create the Total Acidity feature
    df['total_acidity'] = df['fixed_acidity'] + df['volatile_acidity'] + df['citric_acid']

    # Create the Sulfur Dioxide Ratio feature
    df['sulfur_dioxide_ratio'] = df['free_sulfur_dioxide'] / df['total_sulfur_dioxide']

    # Create the pH to Acidity Ratio feature
    df['ph_to_acidity_ratio'] = df['ph'] / df['total_acidity']

    # Create the Alcohol by Density feature
    df['alcohol_by_density'] = df['alcohol'] * df['density']

    # Create the Chlorides by Residual Sugar feature
    df['chlorides_by_residual_sugar'] = df['chlorides'] / df['residual_sugar']

    # Create the Sulfur Dioxide Index (assuming more weight to free_sulfur_dioxide)
    df['sulfur_dioxide_index'] = (2 * df['free_sulfur_dioxide'] + df['total_sulfur_dioxide']) / 3
    
    return df