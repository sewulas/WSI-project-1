from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge

def train_model(X_train, y_train):
    # Tworzymy pusty słownik, do którego wrzucimy gotowe modele
    trained_models = {}
    
    # ---------------------------------------------------------
    # 1. Klasyczna Regresja Liniowa (Model bazowy)
    # ---------------------------------------------------------
    print("\n1. Training Linear Regression ...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    trained_models['Linear Regression'] = lr
    
    # ---------------------------------------------------------
    # 2. Regresja Grzbietowa (Ridge) z Grid Search
    # ---------------------------------------------------------
    print("2. Training Ridge Regression with Grid Search ...")
    ridge = Ridge()
    # W Ridge optymalizujemy parametr 'alpha' (siłę regularyzacji)
    ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    
    ridge_grid = GridSearchCV(estimator=ridge, param_grid=ridge_params, cv=3, scoring='neg_mean_squared_error')
    ridge_grid.fit(X_train, y_train)
    
    print(f"   Najlepsze parametry Ridge: {ridge_grid.best_params_}")
    trained_models['Ridge Regression'] = ridge_grid.best_estimator_
    
    # ---------------------------------------------------------
    # 3. Lasy Losowe (Random Forest) z Grid Search
    # ---------------------------------------------------------
    print("3. Training Random Forest with Grid Search (to może chwilę potrwać) ...")
    rf = RandomForestRegressor(random_state=42)
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20]
    }
    
    rf_grid = GridSearchCV(estimator=rf, param_grid=rf_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    
    print(f"   Najlepsze parametry RF: {rf_grid.best_params_}")
    trained_models['Random Forest'] = rf_grid.best_estimator_
    
    # Zwracamy SŁOWNIK ze wszystkimi trzema wytrenowanymi modelami!
    return trained_models