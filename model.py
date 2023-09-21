import pandas as pd
from scipy import stats

def spearmanr_test(df,col_name):
    # Assuming you have a DataFrame named df with columns 'quality' and 'alcohol'
    spearman_corr, p_value = stats.spearmanr(df['quality'], df[col_name])

    # Interpret the results
    alpha = 0.05  # Set your desired significance level

    if p_value < alpha:
        print(f"There is a statistically significant Spearman's rank correlation (p-value = {p_value:.4f}, corr = {spearman_corr:.4f}).")
    else:
        print(f"There is no statistically significant Spearman's rank correlation (p-value = {p_value:.4f}, corr = {spearman_corr:.4f}).")
