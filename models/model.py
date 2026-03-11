from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge

def train_model(X_train, y_train):
    trained_models = {}
    
    # Regresja liniowa
    # Algorytm analizuje cechy domów (X_train) oraz ceny (y_train)
    # Oblicza współczynniki dla każdej cechy, tak aby poprowadzić idealną krzywą przez te dane
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    trained_models['Linear Regression'] = lr
    
    # Regresja grzebietowa z GridSearch
    # Regresja liniowa ma wbudowany błąd tj. tylko i wyłącznie minimalizuje błąd między ceną realną a obliczaną
    # Gdy cech jest dużo, to algorytm zaczyna przypisywać duże wagi do mało znaczących cech
    # Algorytm taki będzie źle wytrenowany (mała stabilność - dla innych danych nie będzie tak dobrze estymował)
    # Ridge wprowadza kary za zbyt duże wagi (dajemy kilka opcji kar, od 0.1 mała do 100.0 bardzo duża,
    # tak aby algorytm sam dobrał najlepszy parametr)
    ridge = Ridge()
    ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    
    # cv = 3 oznacza kroswalidację
    # Dla każdej alphy (0.1, 1.0, 10.0, 100.0) algorytm dzieli dane na 3 części. Uczy się na dwóch, a ocenia na trzeciej.
    # neg_mean_squared_error dlatego, że biblioteka sklearn chce maksymalizować wynik, a my chcemy minimalny błąd stąd negacja
    ridge_grid = GridSearchCV(estimator=ridge, param_grid=ridge_params, cv=3, scoring='neg_mean_squared_error')
    ridge_grid.fit(X_train, y_train)
    trained_models['Ridge Regression'] = ridge_grid.best_estimator_
    
    # Lasy losowe z GridSearch
    # Ustawiamy seed - powtarzalność wyników
    # Analogicznie jak dla ridge nie wiemy jakie są najlepsze parametry
    # Algorytm sam wybierze (od 50 do 100 drzew, a ich głębokość od 10 do 20)
    rf = RandomForestRegressor(random_state=42)
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20]
    }
    
    # GridSearchCV analogiczny jak przy Ridge
    # n_jobs=-1 parametr optymalizacyjny - drzewa są niezależne i każde z nich może być budowane równolegle, wielowątkowo
    rf_grid = GridSearchCV(estimator=rf, param_grid=rf_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    trained_models['Random Forest'] = rf_grid.best_estimator_
    
    return trained_models