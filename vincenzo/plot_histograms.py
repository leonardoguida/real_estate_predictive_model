import pandas as pd
import os
import numpy as np

import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv("../data/dataset.csv")

# Create plots directory if it doesn't exist
plots_dir = "../plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Iterate through each column
for column in df.columns:
    # Skip columns that can't be plotted as histograms
    if df[column].dtype == 'object':
        # Check if number of unique values is reasonable for a histogram
        if df[column].nunique() > 30:  # Arbitrary threshold
            print(f"Skipping {column}: too many unique string values")
            continue
        
    plt.figure(figsize=(10, 6))
    
    if df[column].dtype == 'object':
        # For categorical data, plot bar chart instead
        value_counts = df[column].value_counts()
        plt.bar(value_counts.index, value_counts.values)
        plt.xticks(rotation=45, ha='right')
    else:
        # For numerical data, plot histogram
        plt.hist(df[column].dropna(), bins=30, edgecolor='black')
    
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(plots_dir, f"{column}_histogram.png"))
    plt.close()

print(f"Histograms saved in {plots_dir}")