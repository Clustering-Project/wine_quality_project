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