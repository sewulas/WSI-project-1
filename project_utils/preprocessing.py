import pandas as pd

def preprocess_data(df):
    # Usuwanie domów, które mają > 4000 m^2 powierzchni (4 sztuki)
    df = df[df['GrLivArea'] <= 4000]
    
    # Usunięcie kolumny Id
    # Bez tego algorytm mógłby uznać, że id jest wartością (np. dom o id 1 byłby tańszy niż o id 100)
    df = df.drop(['Id'], axis=1, errors='ignore')
    
    # Uzupełnianie braków w danych
    # Dla danych tekstowych -> "None"
    # Dla danych numerycznych -> 0
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("None")
        else:
            df[col] = df[col].fillna(0)
    
    # Kodowanie tekstu na liczby
    # Funkcja get_dummmies bierze każdą kolumnę tekstową i przypisuje wartość (podobnie jak enumy)
    # Przykład:
    # BuildingHeight(Low/Mid/High)
    # HOUSE_ID | BuildingHeight
    # 1        | Mid
    # 2        | Low
    # 3        | High
    # => Tworzymy teraz 3 kolumny (Low, Mid, High)
    # HOUSE_ID | Low | Mid | Height
    # 1        | 0   | 1   | 0
    # 2        | 1   | 0   | 0
    # 3        | 0   | 0   | 1
    # Dodatkowo drop_first wyrzuca zbędną pierwszą kolumnę (na podstawie 2 kolejnych znana jest całość informacji)
    df = pd.get_dummies(df, drop_first=True)
    
    return df