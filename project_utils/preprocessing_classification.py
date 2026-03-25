import pandas as pd

def preprocess_classification(df):
    # BRAKI DANYCH
    # Podobnie jak przy regresji, należy uzupełnić brakujące dane.
    # Przy brakującej liczbie wstawiamy medianę wszystkich pacjentów,
    # jest odporniejsza na zbyt duży odchył niż średnia.
    # Dla tekstu wstawiamy dominantę;
    # analog do poprzedniego tylko string.
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # DODATKOWE CECHY
    # Podane są pomiary kątów u dzieci w wieku 9 i 12 lat.
    # Model nie wie, że np. pomiar nr 1 i nr 5 dotyczą
    # tego samego pacjenta na przestrzeni 3 lat.
    # Znajdujemy więc kolumny z wiekiem 9 oraz 
    # szukamy odpowiadającecej z wiekiem 12.
    # Tworzymy w ten sposób deltę, czyli zmienną
    # charakteryzującą kierunek wzrostu szczęki.
    new_features = []

    for col in df.columns:
        if col.startswith("9_"):
            base = col[2:]
            col_12 = "12_" + base
            if col_12 in df.columns:
                new_col = "delta_" + base
                df[new_col] = df[col_12] - df[col]
                new_features.append(new_col)

    # ENCODING
    # Analogicznie do get_dummies w regresji.
    # Modele z biblioteki scikit-learn wymagają, aby
    # target był w pojedynczej kolumnie jako wartości
    # całkowite - przypisujemy 0, 1, 2.
    df["growth direction"] = df["growth direction"].map({
        "horizontal": 0,
        "normal": 1,
        "vertical": 2
    })

    return df