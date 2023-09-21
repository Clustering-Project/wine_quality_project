import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def quality_distribution(df):
    quality_counts = df['quality'].value_counts().sort_index()

    plt.figure(figsize=(8, 6))
    ax = plt.bar(quality_counts.index, quality_counts.values)

    # Add value counts on top of each bar
    for i, v in enumerate(quality_counts.values):
        plt.text(quality_counts.index[i], v + 10, str(v), ha='center', va='bottom')


    # Removing the y-axis
    plt.gca().get_yaxis().set_visible(False)

    # Removing left, right, and top lines of the current axes
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.xlabel('Quality')
    plt.title('Distribution of Wine Quality')
    plt.xticks(quality_counts.index)
    plt.show()


def alcohol_distribution(df):
    # Define the bin edges and labels
    bin_edges = [8, 9, 10, 11, 12, 14]
    bin_labels = ['8-9', '9-10', '10-11', '11-12', '12-14']

    # Bin the 'alcohol' values
    df['alcohol_bins'] = pd.cut(df['alcohol'], bins=bin_edges, labels=bin_labels, include_lowest=True)

    # Count the values in each bin
    alcohol_counts = df['alcohol_bins'].value_counts().sort_index()

    plt.figure(figsize=(4, 5))
    ax = plt.bar(alcohol_counts.index, alcohol_counts.values)

    # Adding value counts on top of each bar
    for i, count in enumerate(alcohol_counts.values):
        plt.text(i, count, str(count), ha='center', va='bottom')

    # Removing the y-axis
    plt.gca().get_yaxis().set_visible(False)

    # Removing left, right, and top lines of the current axes
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.xlabel('Alcohol')
    plt.title('Distribution of Alcohol')
    plt.xticks(alcohol_counts.index)

    # Remove the "quality_bins" column from the DataFrame
    df.drop(columns=['alcohol_bins'], inplace=True)

    plt.show()

def alcohol_vs_quanity(df):
    # Define custom labels for "quality"
    bins_q = [3, 5, 6, 9]
    labels_q = ['Low', 'Med', 'High']

    # Create a new column "quality_bins" to store the bin labels
    df['quality_bins'] = pd.cut(df['quality'], bins=bins_q, labels=labels_q)

    # Group the data by "quality" and calculate the mean alcohol content for each quality category
    quality_means = df.groupby('quality_bins')['alcohol'].mean().round(1)

    # Create a bar plot
    plt.figure(figsize=(6, 6))
    quality_means.plot(kind='bar', color='lightseagreen', width=0.7)
    plt.title('Average Alcohol Content', fontsize=20)
    plt.xlabel('Quality', fontsize=18)
    plt.xticks(rotation=0, fontsize=16)
    plt.grid(axis='y')

    # Adding average on top of each bar
    for i, count in enumerate(quality_means.values):
        plt.text(i, count,  '{:.1f}%'.format(count), ha='center', va='bottom', fontsize=16)

    # Removing the y-axis
    plt.gca().get_yaxis().set_visible(False)

    # Removing left, right, and top lines of the current axes
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
        # Remove the ticks (not labels) of the x-axis
    plt.tick_params(axis='x', which='both', bottom=False, top=False)

    plt.ylim(8, 13)

    # Remove the "quality_bins" column from the DataFrame
    df.drop(columns=['quality_bins'], inplace=True)

    # Show the plot
    plt.show()