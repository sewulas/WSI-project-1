import pandas as pd
from matplotlib import pyplot as plt

def visualise(df, target, top_n):
    df = pd.get_dummies(df)

    corr = df.corr()
    sale_corr = corr[target].sort_values(ascending=False)

    top_features = sale_corr.index[1:top_n + 1]
    print("Top features:")
    print(sale_corr.head(top_n + 1))

    for feature in top_features:
        visualise_feature(df, target, feature)

def visualise_feature(df, target, feature):
    print("Plotting target:", target, "against feature:", feature)

    # wykres
    plt.scatter(df[feature], df[target])

    plt.xlabel(feature)
    plt.ylabel(target)
    plt.title(target + " vs " + feature)

    plt.show()