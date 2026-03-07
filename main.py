import pandas as pd
import numpy as np
from project_utils.preprocessing import preprocess_data
from models.model import train_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from project_utils.visualisation import visualise, plot_predictions, plot_residuals, plot_feature_importance


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def split_data(df: pd.DataFrame):
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STANDARYZACJA (Skalowanie cech)
    # Uczymy skaler tylko na danych treningowych, żeby model nie "podglądał" danych testowych
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def evaluate_model(models_dict, X_test, y_test):
    # Przechodzimy pętlą przez każdy model w słowniku
    for name, model in models_dict.items():
        print(f"\n{'='*40}")
        print(f"Wyniki dla modelu: {name}")
        print(f"{'='*40}")
        
        # Predykcja
        predictions = model.predict(X_test)

        # Metryki
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"Błąd MSE:  {mse:,.2f}")
        print(f"Błąd RMSE: {rmse:,.2f} USD")
        print(f"Błąd MAE:  {mae:,.2f} USD")
        print(f"Wynik R^2: {r2:.4f}")

def main():

    print("Loading data...")
    df = load_data("data/domy.csv")

    #print(df.columns.tolist()) ###
    print("Preprocessing data...")
    df = preprocess_data(df)

    visualise(df, "SalePrice", 5)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

    print("Visualising model results...")

    plot_predictions(model, X_test, y_test)
    plot_residuals(model, X_test, y_test)

    feature_names = df.drop("SalePrice", axis=1).columns
    plot_feature_importance(model, feature_names)

    print("Done!")


if __name__ == "__main__":
    main()