
def preprocess_classification(df):
    print("Preprocessing classification data...")

    # Uzupełnianie braków w danych
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # (eksperymentalne) dodanie dodatkowych cech: różnica (delta) między cechami 12_* a 9_*
    new_features = []

    for col in df.columns:
        if col.startswith("9_"):
            base = col[2:]
            col_12 = "12_" + base

            if col_12 in df.columns:
                new_col = "delta_" + base
                df[new_col] = df[col_12] - df[col]
                new_features.append(new_col)

    print(f"Created {len(new_features)} delta features")


    # Encoding targetu
    df["growth direction"] = df["growth direction"].map({
        "horizontal": 0,
        "normal": 1,
        "vertical": 2
    })

    return df