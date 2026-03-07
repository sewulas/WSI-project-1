import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def visualise(df, target, top_n):
    # df = pd.get_dummies(df)

    corr = df.corr()
    sale_corr = corr[target].sort_values(ascending=False)

    top_features = sale_corr.index[1:top_n + 1]
    print("Top features:")
    print(sale_corr.head(top_n + 1))

    for feature in top_features:
        visualise_feature(df, target, feature)
    plt.show()

def visualise_feature(df, target, feature):
    print("Plotting target:", target, "against feature:", feature)

    plt.figure()

    # wykres
    plt.scatter(df[feature], df[target])

    plt.xlabel(feature)
    plt.ylabel(target)
    plt.title(target + " vs " + feature)

def plot_predictions(models_dict, X_test, y_test):

    for name, model in models_dict.items():

        predictions = model.predict(X_test)

        plt.figure()

        plt.scatter(y_test, predictions, alpha=0.5)

        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            'r'
        )

        plt.xlabel("True Price")
        plt.ylabel("Predicted Price")
        plt.title(f"Predicted vs Actual ({name})")

        plt.show()


def plot_residuals(models_dict, X_test, y_test):

    for name, model in models_dict.items():

        predictions = model.predict(X_test)
        residuals = y_test - predictions

        plt.figure()

        plt.scatter(predictions, residuals, alpha=0.5)

        plt.axhline(0)

        plt.xlabel("Predicted Price")
        plt.ylabel("Residual")
        plt.title(f"Residual Plot ({name})")

        plt.show()


def plot_feature_importance(models_dict, feature_names):

    for name, model in models_dict.items():

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
        else:
            continue

        indices = np.argsort(importances)[-15:]

        plt.figure()

        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])

        plt.title(f"Feature Importance ({name})")
        plt.show()