from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_classification_model(X_train, y_train):
    trained_models = {}
    
    # 1. Regresja logistyczna (klasyfikacja)
    # W przeciwieństwie do regresji, która próbowała wyznaczyć prostą
    # przechodzącą jak najbliżej wszystkich punktów, klasyfikacja
    # oddziela od siebie różne grupy pacjentów (klasyfikuje na grupy).
    # max_iter = 1000, aby dokładniejszy podział (domyślnie 100) 
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    trained_models['Logistic Regression'] = lr
    
    # 2. Drzewo decyzyjne z Grid Search
    # Analogicznie jak w regresji
    # max_depth to liczba zapytań - testujemy 3, 5 i 10 pytań z rzędu,
    # w ostateczności None - tak długo, aż w każdej gałęzi zostanie 1 pacjent.
    # Min_samples_split - ilu minimum pacjentów musi wpaść do danej gałęzi, aby
    # opłacało się dalej dzielić tę grupę (dalsze zagłębienia)
    # Scoring_accuracy - w przeciwieństwie do regresji nie interesuje nas błąd średni
    # tylko dokładność tj. dany procent postawionych diagnoz - Grid Search testuje
    # wszystkie kombinacje drzew i wybierze to z najwiękzą liczbą celnych trafień.
    dt = DecisionTreeClassifier(random_state=42)
    dt_params = {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10]}
    dt_grid = GridSearchCV(estimator=dt, param_grid=dt_params, cv=3, scoring='accuracy')
    dt_grid.fit(X_train, y_train)
    trained_models['Decision Tree'] = dt_grid.best_estimator_
    
    # 3. Lasy losowe
    # Tworzymy 50 lub 100 małych drzew decyzyjnych (tych powyżej)
    # Końcowa przyneleżność pacjenta do grupy na zasadzie większości
    # (dany pacjent przynależy do kategorii x <=> kategoria x wystąpiła
    # najczęściej wśród wszystkich drzew decyzyjnych).
    # cv = 3 kroswalidacja
    rf = RandomForestClassifier(random_state=42)
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, 20]
    }
    rf_grid = GridSearchCV(estimator=rf, param_grid=rf_params, cv=3, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    trained_models['Random Forest'] = rf_grid.best_estimator_
    
    return trained_models