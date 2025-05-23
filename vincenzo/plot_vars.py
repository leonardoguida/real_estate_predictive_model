import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    df = pd.read_csv("../data/dataset.csv")
    print("Successfully loaded data/dataset.csv")

    # pairplot
    sns.pairplot(df, diag_kind='kde')
    plt.savefig("plots/pairplot.png")
    print("Successfully saved plots/pairplot.png")
    plt.show()

main()