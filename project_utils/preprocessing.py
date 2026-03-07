import pandas as pd

def preprocess_data(df):
    print("Preprocessing data ...")
    
    # 1. Usuwamy domy > 4000 stóp kwadratowych [cite: 138]
    df = df[df['GrLivArea'] <= 4000]
    
    # Usuwamy kolumnę z ID
    df = df.drop(['Id'], axis=1, errors='ignore')
    
    # 2. Inteligentne uzupełnianie braków
    # Dla kolumn liczbowych wstawiamy 0, dla tekstowych tekst "Brak"
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("Brak")
        else:
            df[col] = df[col].fillna(0)
            
    # 3. KODOWANIE: Zamieniamy tekst na zera i jedynki (One-Hot Encoding) 
    # To sprawi, że model nagle dostanie dostęp do informacji o dzielnicy czy jakości!
    df = pd.get_dummies(df, drop_first=True)
    
    return df