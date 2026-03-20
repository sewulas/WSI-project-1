import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA


def plot_class_distribution(df, target):
    # ile mamy próbek każdej klasy

    plt.figure()

    df[target].value_counts().plot(kind='bar')

    plt.title("Class distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")

    plt.show()


def plot_feature_by_class(df, feature, target):
    # jak cecha rozkłada się dla różnych klas

    plt.figure()

    sns.boxplot(x=df[target], y=df[feature])

    plt.title(f"{feature} vs {target}")

    plt.show()


def plot_pca(df, target):
    # redukcja do 2D (wizualizacja separacji klas)

    X = df.drop(target, axis=1)
    y = df[target]

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure()

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=0.6)

    plt.title("PCA projection")

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.colorbar(label="Class")

    plt.show()


def plot_top_correlations(df, target, top_n=5):
    # Pokazuje które cechy są najbardziej powiązane z targetem

    df_num = df.select_dtypes(include=np.number)

    corr = df_num.corr()[target].abs().sort_values(ascending=False)

    top_features = corr.index[1:top_n+1]

    print("Top correlated features:")
    print(corr.head(top_n+1))

    return top_features


def plot_top_features(df, target, top_features):
    # Rysuje boxploty dla najważniejszych cech

    for feature in top_features:
        plt.figure()

        sns.boxplot(x=df[target], y=df[feature])

        plt.title(f"{feature} vs {target}")

        plt.show()

def plot_2d_features(df, feature_x, feature_y, target):

    plt.figure()

    plt.scatter(
        df[feature_x],
        df[feature_y],
        c=df[target],
        alpha=0.6
    )

    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f"{feature_x} vs {feature_y}")

    plt.colorbar(label="Class")

    plt.show()