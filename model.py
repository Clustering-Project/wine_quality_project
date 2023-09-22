import pandas as pd
from scipy import stats

from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression, Lasso

from math import sqrt
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor


# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def eval_baseline(y_train):
    """
    Evaluate the baseline model's performance using the root mean squared error (RMSE).

    Parameters:
        y_train (pd.Series): The target variable from the training dataset.

    Returns:
        float: The RMSE score representing the baseline model's performance.

    Note:
        - The function creates a baseline model by predicting the mean value of the training target variable for all samples.
        - It calculates the RMSE between the actual target values and the mean predictions.
        - The RMSE score quantifies the baseline model's performance.
    """
    baselines = pd.DataFrame({'y_actual': y_train, 'y_mean': y_train.mean()})
    
    return float(f"{sqrt(mean_squared_error(baselines.y_actual, baselines.y_mean)):.4f}")

        

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def eval_model(y_actual, y_hat):
    """
    Evaluate a model's performance using the root mean squared error (RMSE).

    Parameters:
        y_actual (pd.Series): The actual target values.
        y_hat (pd.Series or np.array): The predicted target values.

    Returns:
        float: The RMSE score representing the model's performance.

    Note:
        - The function calculates the RMSE between the actual target values and the predicted values.
        - The RMSE score quantifies the model's performance, where lower values indicate better performance.
    """
    return sqrt(mean_squared_error(y_actual, y_hat))

def update_model_results(model_name, train_rmse, val_rmse, model_results=None):
    """
    Update a DataFrame with model evaluation results (RMSE) for a given model.

    Parameters:
        model_name (str): The name or identifier of the model.
        train_rmse (float): The root mean squared error (RMSE) on the training dataset.
        val_rmse (float): The root mean squared error (RMSE) on the validation dataset.
        model_results (pd.DataFrame, optional): An existing DataFrame containing model results. Default is None.

    Returns:
        pd.DataFrame: An updated DataFrame with the new model's results.

    Note:
        - The function creates a DataFrame with the model's name and RMSE results on the training and validation datasets.
        - If `model_results` is provided, it concatenates the new results with the existing DataFrame.
        - If `model_results` is not provided, it creates a new DataFrame to store the results.
    """
    # Create a DataFrame with model name and RMSE results
    results_df = pd.DataFrame({
        'Model': [model_name],
        'Train_RMSE': [train_rmse],
        'Val_RMSE': [val_rmse]
    })
    
    # Check if model_results already exists
    if model_results is not None:
        # Concatenate results with existing DataFrame
        model_results = pd.concat([model_results, results_df], ignore_index=True)
    else:
        # Create a new DataFrame if it doesn't exist
        model_results = results_df

    return model_results

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def train_model(model_name, X_train, y_train, X_val, y_val, model_results=None):
    """
    Train a machine learning model, evaluate its performance, and update the model results DataFrame.

    Parameters:
        model_name (class): The machine learning model class (e.g., LinearRegression).
        X_train (pd.DataFrame): The feature matrix of the training dataset.
        y_train (pd.Series): The target variable of the training dataset.
        X_val (pd.DataFrame): The feature matrix of the validation dataset.
        y_val (pd.Series): The target variable of the validation dataset.
        model_results (pd.DataFrame, optional): An existing DataFrame containing model results. Default is None.

    Returns:
    model: Trained machine learning model.
    model_results: Updated DataFrame containing model name and RMSE results.

    Note:
        - The function trains a machine learning model on the provided training data.
        - It evaluates the model's performance on both the training and validation sets using RMSE.
        - RMSE values are printed for both sets in a formatted manner.
        - The model name is extracted from the class and used for updating the model results DataFrame.
        - If `model_results` is provided, it is updated with the new model's results.
        - If `model_results` is not provided, a new DataFrame is created to store the results.
    """
    # Fit the model on the training data
    model = model_name()
    model.fit(X_train, y_train)
    
    # Make predictions on the training set
    train_preds = model.predict(X_train)
    
    # Calculate RMSE on the training set
    train_rmse = eval_model(y_train, train_preds)
    
    # Make predictions on the validation set
    val_preds = model.predict(X_val)
    
    # Calculate RMSE on the validation set
    val_rmse = eval_model(y_val, val_preds)
    
    # Print RMSE values for training and validation sets (formatted)
    train_rmse_formatted = "{:,.2f}".format(train_rmse)
    val_rmse_formatted = "{:,.2f}".format(val_rmse)
    print(f'The train RMSE is {train_rmse_formatted}.')
    print(f'The validate RMSE is {val_rmse_formatted}.')
    
    # Extract the name of the model class without the module path
    model_name = model.__class__.__name__

    # Update the model results DataFrame
    model_results = update_model_results(model_name, train_rmse_formatted, val_rmse_formatted, model_results)

    return model, model_results
