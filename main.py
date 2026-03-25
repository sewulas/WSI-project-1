import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from project_utils.preprocessing_classification import preprocess_classification
from models.model import train_classification_model
from project_utils.visualisation_classification import plot_class_distribution, plot_pca, plot_top_correlations, \
    plot_top_features, run_classification_visualisation, run_classification_results


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def split_data(df: pd.DataFrame):
    X = df.drop("growth direction", axis=1)
    y = df["growth direction"]

    # Stratify=y gwarantuje, że w zbiorze treningowym i testowym 
    # proporcje pacjentów z różnymi kierunkami wzrostu będą takie same
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def evaluate_model(models_dict, X_test, y_test):
    for name, model in models_dict.items():
        print(f"\n{'='*50}")
        print(f"Wyniki dla modelu: {name}")
        print(f"{'='*50}")
        
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"Ogólna dokładność (Accuracy): {acc * 100:.2f}%\n")

        target_names = ["horizontal", "normal", "vertical"]
        print(classification_report(y_test, predictions, target_names=target_names))

def main():
    df = load_data("data/ortodoncja.csv")
    df = preprocess_classification(df)
    X_train, X_test, y_train, y_test = split_data(df)
    models = train_classification_model(X_train, y_train)
    evaluate_model(models, X_test, y_test)

    print("Visualising classification data...")
    run_classification_visualisation(df)

    feature_names = df.drop("growth direction", axis=1).columns

    run_classification_results(models, X_test, y_test, feature_names)

if __name__ == "__main__":
    main()