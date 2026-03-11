import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# visualisation.py
# Plik zawiera funkcje do wizualizacji danych oraz wyników modeli ML
# - analiza korelacji cech
# - wykresy zależności cech od targetu
# - wizualizacja predykcji modeli
# - analiza błędów (residuals)
# - wizualizacja ważności cech

def visualise(df, target, top_n):
    # Funkcja służy do wstępnej analizy danych i pre-processingu
    # Szukamy cech najbardziej skorelowanych z targetem (np. SalePrice)

    # Obliczamy macierz korelacji pomiędzy wszystkimi kolumnami numerycznymi
    corr = df.corr()

    # Sortujemy korelacje względem targetu malejąco
    # Dzięki temu pierwsze elementy listy to cechy najsilniej powiązane z ceną domu
    sale_corr = corr[target].sort_values(ascending=False)

    # Wybieramy top_n cech (pomijamy pierwszą pozycję bo to zawsze korelacja targetu z samym sobą)
    top_features = sale_corr.index[1:top_n + 1]

    # Wypisujemy najważniejsze cechy
    print("Top features:")
    print(sale_corr.head(top_n + 1))

    # Dla każdej z wybranych cech generujemy wykres zależności
    for feature in top_features:
        visualise_feature(df, target, feature)

    plt.show()

def visualise_feature(df, target, feature):
    # Funkcja rysuje wykres zależności jednej cechy od targetu, gdzie każdy punkt to jeden dom

    print("Plotting target:", target, "against feature:", feature)

    plt.figure()
    plt.scatter(df[feature], df[target])

    # Opisy osi
    plt.xlabel(feature)
    plt.ylabel(target)

    plt.title(target + " vs " + feature)


def plot_predictions(models_dict, X_test, y_test):

    # Funkcja wizualizuje jak dobrze model przewiduje ceny
    # Rysujem wykres: wartości prawdziwe vs wartości przewidziane, czyli dla danej prawdziwej ceny
    # przypisuje jej predykcję modelu na podstawie cech tego domu.

    for name, model in models_dict.items():

        predictions = model.predict(X_test)

        plt.figure()
        plt.scatter(y_test, predictions, alpha=0.5)

        # Linia idealnej predykcji (y = x)
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
    # Funkcja rysuje tzw. residual plot
    # Residual = różnica pomiędzy prawdziwą ceną a ceną przewidzianą przez model
    # Jeżeli punkt znajduje się poniżej prostej x = 0, to model zawyża cenę względem ceny oczekiwanej
    # a jeżeli punkt znajduje się nad tą prostą, to model zaniża w predykcji cenę względem prawdziwej ceny.
    # W idealnym modelu predykcje (punkty na wykresie) powinni znajdować się na prostej x = 0

    for name, model in models_dict.items():

        predictions = model.predict(X_test)

        # Obliczenie błędów modelu
        residuals = y_test - predictions

        plt.figure()
        plt.scatter(predictions, residuals, alpha=0.5)

        # Linia pozioma w punkcie 0 => przedstawia błąd równy 0, czyli predykcja == oczekiwanej cenie
        plt.axhline(0)

        plt.xlabel("Predicted Price")
        plt.ylabel("Residual")
        plt.title(f"Residual Plot ({name})")

        plt.show()


def plot_feature_importance(models_dict, feature_names):

    # Funkcja pokazuje które cechy są najważniejsze dla modelu

    for name, model in models_dict.items():

        # Modele drzewiaste (np Random Forest) mają atrybut feature_importances_
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        # Modele liniowe (Linear Regression, Ridge) mają współczynniki coef_
        # Bierzemy wartość bezwzględną bo znak oznacza tylko kierunek wpływu
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
        else:
            continue

        # Sortujemy cechy po ważności i wybieramy 15 najważniejszych
        indices = np.argsort(importances)[-15:]

        plt.figure()

        # Wykres słupkowy poziomy
        plt.barh(range(len(indices)), importances[indices])

        # Nazwy cech na osi Y przy odpowiednich słupkach poziomych
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])

        plt.title(f"Feature Importance ({name})")

        plt.show()