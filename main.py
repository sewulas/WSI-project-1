import pandas as pd
import numpy as np
from project_utils.preprocessing import preprocess_data
from models.model import train_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from project_utils.visualisation import plot_predictions, plot_residuals, plot_feature_importance
from project_utils.preprocessing_classification import preprocess_classification
from project_utils.visualisation_classification import (
    plot_class_distribution,
    plot_feature_by_class,
    plot_pca,
    plot_top_correlations,
    plot_top_features, plot_2d_features
)

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def split_data(df: pd.DataFrame):
    # Dzielenie danych
    # X czyli zestaw cech, to wszystkie kolumny poza targetem (SalePrice)
    # y to wektor wartości prawdziwych (SalePrice)
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    # Modelom dajemy 80% danych do nauki, natomiast 20% pozostawiamy jako test dla wytrenowanych modeli
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standaryzacja (skalowanie cech)
    # Bez tego algorytm błędnie przypisałby ważność różnych cech np.
    # Liczba pokoi = 5, metraż = 500 -> metraż byłby bardzo ważny a liczba pokoi wręcz nieistotna
    # Algorytm spłaszcza te dane, tak aby ważność cech była sensownie przypisywana
    # Skaler uczony tylko na danych treningowych (80% wszystkich danych)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def evaluate_model(models_dict, X_test, y_test):
    # Tutaj już po wytrenowaniu, modele oceniamy na danych testowych (20% wszystkich danych)
    for name, model in models_dict.items():
        print(f"\n{'='*40}")
        print(f"Wyniki dla modelu: {name}")
        print(f"{'='*40}")
        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"Błąd MSE:  {mse:,.2f}") # błąd średniokwadratowy (różnica^2)
        print(f"Błąd RMSE: {rmse:,.2f} USD") # mse.sqrt()
        print(f"Błąd MAE:  {mae:,.2f} USD") # błąd średni
        print(f"Wynik R^2: {r2:.4f}") # współczynnik determinacji (0-1) gdzie im bliżej 1 tym model dokładniejszy

def main():
    # df = load_data("data/domy.csv")
    # df = preprocess_data(df)
    # X_train, X_test, y_train, y_test = split_data(df)
    # model = train_model(X_train, y_train)
    # evaluate_model(model, X_test, y_test)
    #
    # # WIZUALIZACJA
    # plot_predictions(model, X_test, y_test)
    # plot_residuals(model, X_test, y_test)
    # feature_names = df.drop("SalePrice", axis=1).columns
    # plot_feature_importance(model, feature_names)


    # KLASYFIKACJA - ORTODONCJA
    print("\nLoading classification data...")
    df_clf = load_data("data/ortodoncja.csv")

    print("Preprocessing classification data...")
    df_clf = preprocess_classification(df_clf)

    target_clf = "growth direction"

    print("Visualising classification data...")

    # Rozkład klas
    plot_class_distribution(df_clf, target_clf)

    # PCA
    plot_pca(df_clf, target_clf)

    # Korelacje
    top_features = plot_top_correlations(df_clf, target_clf, top_n=5)

    # Boxploty najważniejszych(?) cech
    plot_top_features(df_clf, target_clf, top_features)

    plot_2d_features(df_clf, "12_SN/MP", "12_ANB", "growth direction")

    plot_2d_features(df_clf, "delta_SN/MP", "delta_ANB", target_clf)
    plot_2d_features(df_clf, "12_SN/MP", "delta_SN/MP", target_clf)
    plot_2d_features(df_clf, "12_SN/MP", "9_SN/MP", target_clf)

if __name__ == "__main__":
    main()