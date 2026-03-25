import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score

CLASS_LABELS = {
    0: "horizontal",
    1: "normal",
    2: "vertical"
}

# Wizualizacja danych
def run_classification_visualisation(df, target="growth direction"):
    # Główna funkcja do wizualizacji danych
    # Wywołuje wszystkie wykresy po kolei

    print("\n=== CLASS DISTRIBUTION ===")
    plot_class_distribution(df, target)

    print("\n=== PCA ===")
    plot_pca(df, target)

    print("\n=== TOP FEATURES ===")
    top_features = plot_top_correlations(df, target, top_n=5)

    print("\n=== FEATURE ANALYSIS (BOXPLOTS) ===")
    plot_top_features(df, target, top_features)

    print("\n=== 2D FEATURE RELATIONSHIPS ===")
    plot_selected_2d(df, target)


def plot_class_distribution(df, target):
    # Sprawdzamy ile mamy przykładów każdej klasy

    plt.figure()
    df[target].map(CLASS_LABELS).value_counts().plot(kind='bar')

    plt.title("Class distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")

    plt.show()


def plot_pca(df, target):
    # PCA redukuje dane do 2 wymiarów
    # Pozwala zobaczyć czy klasy są rozdzielone w przestrzeni

    X = df.drop(target, axis=1)
    y = df[target]

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure()

    for cls, label in CLASS_LABELS.items():
        idx = (y == cls)
        plt.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            label=label,
            alpha=0.6
        )

    plt.title("PCA projection")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.legend()

    plt.show()

def plot_top_correlations(df, target, top_n=5):
    # Sprawdzamy które cechy są najbardziej powiązane z targetem
    # Dzięki temu wiemy co może być ważne dla modelu

    df_num = df.select_dtypes(include=np.number)

    corr = df_num.corr()[target].abs().sort_values(ascending=False)

    top_features = corr.index[1:top_n+1]

    print("Top correlated features:")
    print(corr.head(top_n+1))

    return top_features


def plot_top_features(df, target, top_features):
    # Boxplot pokazuje jak dana cecha rozkłada się w klasach
    # Jeśli wykresy się różnią -> cecha dobrze rozróżnia klasy

    for feature in top_features:
        plt.figure()

        sns.boxplot(x=df[target].map(CLASS_LABELS), y=df[feature])

        plt.title(f"{feature} vs {target}")

        plt.show()

def plot_selected_2d(df, target):
    # Sprawdzamy relacje między wybranymi cechami
    # Szukamy czy klasy tworzą skupiska

    # klasyczne wartości
    plot_2d_features(df, "12_SN/MP", "12_ANB", target)

    # zmiany w czasie
    plot_2d_features(df, "delta_SN/MP", "delta_ANB", target)

    # kombinacja (najbardziej informatyczna)
    plot_2d_features(df, "12_SN/MP", "delta_SN/MP", target)

    # pokazanie redundancji cech
    plot_2d_features(df, "12_SN/MP", "9_SN/MP", target)


def plot_2d_features(df, feature_x, feature_y, target):
    # Wykres dwóch cech
    # Kolor oznacza klasę pacjenta

    plt.figure()

    for cls, label in CLASS_LABELS.items():
        subset = df[df[target] == cls]

        plt.scatter(
            subset[feature_x],
            subset[feature_y],
            label=label,
            alpha=0.6
        )

    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f"{feature_x} vs {feature_y}")

    plt.legend()

    plt.show()


# Wizualizacja wyników

def run_classification_results(models_dict, X_test, y_test, feature_names):
    # Analiza wyników modeli
    # Porównujemy modele i sprawdzamy ich działanie

    print("\n=== MODEL COMPARISON ===")
    plot_model_comparison(models_dict, X_test, y_test)

    print("\n=== CONFUSION MATRICES ===")
    plot_confusion_matrices(models_dict, X_test, y_test)

    print("\n=== FEATURE IMPORTANCE ===")
    plot_feature_importance(models_dict, feature_names)


def plot_model_comparison(models_dict, X_test, y_test):
    # Sprawdzamy accuracy każdego modelu
    # Dzięki temu wiemy który działa najlepiej

    model_names = []
    scores = []

    for name, model in models_dict.items():
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        model_names.append(name)
        scores.append(acc)

        print(f"{name}: {acc:.4f}")

    plt.figure()

    plt.bar(model_names, scores)

    plt.title("Model Comparison (Accuracy)")
    plt.ylabel("Accuracy")

    plt.show()


def plot_confusion_matrices(models_dict, X_test, y_test):
    # Confusion matrix pokazuje jak model klasyfikuje dane
    # i gdzie popełnia błędy.
    # Jak czytać wykres:
    # - wiersze (oś Y) -> prawdziwa klasa (True)
    # - kolumny (oś X) -> przewidziana klasa (Predicted)
    # Na przykład:
    # jeśli mamy klasę "vertical" (np. 2):
    # - wartość na przekątnej (2,2) -> ile przypadków poprawnie sklasyfikowanych
    # - wartości poza przekątną -> ile razy model się pomylił
    # Znaczenie dla analizy i interpretacji wyników:
    # - jeśli jedna klasa często trafia do innej to oznacza, że model ich nie rozróżnia
    # - jeśli jedna kolumna ma dużo wartości to model "faworyzuje" tę klasę nad inne
    # - jeśli jakaś klasa ma mało trafień to model sobie z nią nie radzi

    for name, model in models_dict.items():

        predictions = model.predict(X_test)
        cm = confusion_matrix(y_test, predictions)

        plt.figure()

        labels = list(CLASS_LABELS.values())

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels
        )

        plt.title(f"Confusion Matrix ({name})")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        plt.show()


def plot_feature_importance(models_dict, feature_names):
    # Pokazuje które cechy są najważniejsze dla modelu
    # Dla drzew -> feature_importances_
    # Dla modeli liniowych -> coef_

    for name, model in models_dict.items():

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).mean(axis=0)

        else:
            continue

        indices = np.argsort(importances)[-10:]

        plt.figure()

        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])

        plt.title(f"Feature Importance ({name})")

        plt.show()