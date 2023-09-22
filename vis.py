import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from explore import cluster_alc_dens

def quality_distribution(df):
    quality_counts = df['quality'].value_counts().sort_index()

    plt.figure(figsize=(6, 5))
    ax = plt.bar(quality_counts.index, quality_counts.values,  color='lightseagreen')

    # # Add value counts on top of each bar
    # for i, v in enumerate(quality_counts.values):
    #     plt.text(quality_counts.index[i], v + 10, str(v), ha='center', va='bottom', fontsize=16)

    # Add rounded value counts on top of each bar
    for i, v in enumerate(quality_counts.values):
        if v >= 10:
            rounded_value = round(v, -1)  # Round to the nearest 10th for values >= 10
        else:
            rounded_value = v  # Leave values less than 10 unchanged
        plt.text(quality_counts.index[i], rounded_value + 10, str(rounded_value), ha='center', va='bottom', fontsize=16)


    # Removing the y-axis
    plt.gca().get_yaxis().set_visible(False)

    # Removing left, right, and top lines of the current axes
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.xlabel('Quality', fontsize=18, labelpad=20)
    plt.title('Distribution of Wine Quality', fontsize=20)
    plt.xticks(quality_counts.index, fontsize=16)
    
    # Use plt.tight_layout() to ensure proper spacing
    plt.tight_layout()
    
    plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------------------

def alcohol_distribution(df):
    # Define the bin edges and labels
    bin_edges = [8, 9, 10, 11, 12, 14]
    bin_labels = ['8-9', '9-10', '10-11', '11-12', '12-14']

    # Bin the 'alcohol' values
    df['alcohol_bins'] = pd.cut(df['alcohol'], bins=bin_edges, labels=bin_labels, include_lowest=True)

    # Count the values in each bin
    alcohol_counts = df['alcohol_bins'].value_counts().sort_index()

    plt.figure(figsize=(4, 5))
    ax = plt.bar(alcohol_counts.index, alcohol_counts.values,  color='lightseagreen')

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

# ---------------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------------------

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
    plt.xlabel('Quality', fontsize=18, labelpad=20)
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

    # Use plt.tight_layout() to ensure proper spacing
    plt.tight_layout()

    # Show the plot
    plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------------------

def density_vs_quantity(df):
    # Define custom labels for "quality"
    bins_q = [3, 5, 6, 9]
    labels_q = ['Low', 'Med', 'High']

    # Create a new column "quality_bins" to store the bin labels
    df['quality_bins'] = pd.cut(df['quality'], bins=bins_q, labels=labels_q)

    # Group the data by "quality" and calculate the mean alcohol content for each quality category
    quality_means = df.groupby('quality_bins')['density'].mean().round(6)

    # Create a bar plot
    plt.figure(figsize=(6, 6))
    quality_means.plot(kind='bar', color='lightseagreen', width=0.7)
    plt.title('Average density', fontsize=20)
    plt.xlabel('Quality', fontsize=18, labelpad=20)
    plt.xticks(rotation=0, fontsize=16)
    plt.grid(axis='y')

    # Adding average on top of each bar
    for i, count in enumerate(quality_means.values):
        plt.text(i, count, '{:.4f}'.format(count), ha='center', va='bottom', fontsize=16)

    # Removing the y-axis
    plt.gca().get_yaxis().set_visible(False)

    # Removing left, right, and top lines of the current axes
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    
    plt.tick_params(axis='x', which='both', bottom=False, top=False)

    plt.ylim(0.98711, 1)

    # Remove the "quality_bins" column from the DataFrame
    df.drop(columns=['quality_bins'], inplace=True)

    # Use plt.tight_layout() to ensure proper spacing
    plt.tight_layout()
    # Show the plot
    plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------------------

def v_acidity_vs_quantity(df):
    # Define custom labels for "quality"
    bins_q = [3, 5, 6, 9]
    labels_q = ['Low', 'Med', 'High']

    # Create a new column "quality_bins" to store the bin labels
    df['quality_bins'] = pd.cut(df['quality'], bins=bins_q, labels=labels_q)

    # Group the data by "quality" and calculate the mean alcohol content for each quality category
    quality_means = df.groupby('quality_bins')['volatile_acidity'].mean().round(6)

    # Create a bar plot
    plt.figure(figsize=(6, 6))
    quality_means.plot(kind='bar', color='lightseagreen', width=0.7)
    plt.title('Average Volatile Acidity', fontsize=20)
    plt.xlabel('Quality', fontsize=18, labelpad=20)
    plt.xticks(rotation=0, fontsize=16)
    plt.grid(axis='y')

    # Adding average on top of each bar
    for i, count in enumerate(quality_means.values):
        plt.text(i, count, '{:.2f}'.format(count), ha='center', va='bottom', fontsize=16)

    # Removing the y-axis
    plt.gca().get_yaxis().set_visible(False)

    # Removing left, right, and top lines of the current axes
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    
    plt.gca().spines['bottom'].set_visible(False)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    
    plt.ylim(0.1, .5)

    # Remove the "quality_bins" column from the DataFrame
    df.drop(columns=['quality_bins'], inplace=True)

    # Use plt.tight_layout() to ensure proper spacing
    plt.tight_layout()
    # Show the plot
    plt.show()

def qual_cluster(df):
    
    df = cluster_alc_dens(df)
    # Define a custom legend labels dictionary
    legend_labels = {0: 'Low', 1: 'Med', 2: 'High'}

    # Create the scatterplot with Seaborn
    sns.scatterplot(data=df, x='alcohol', y='density', hue='alc_dens_cluster', palette='deep')

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    
    # Set the title
    plt.title("Alcohol vs. Density", fontsize=20)

    # Modify the legend title and labels
    plt.legend(title="Quality Cluster", labels=[legend_labels[i] for i in range(3)])
    
    # Increase the font size of x and y labels
    plt.xlabel("Alcohol", fontsize=16, labelpad=10)  # Adjust the fontsize as needed
    plt.ylabel("Density", fontsize=16)  # Adjust the fontsize as needed
    
    # Use plt.tight_layout() to ensure proper spacing
    plt.tight_layout()
    # Show the plot
    
    # Show the plot
    plt.show()