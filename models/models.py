from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

import joblib
import pandas as pd
import numpy as np


artifact = joblib.load("../Utils/preprocessing_pipeline.pkl")

pipeline = artifact["pipeline"]

df = pd.read_csv('DataVaxModelos.csv')
df = df.drop(columns=df.columns[0])

df= df.drop_duplicates()
df.dropna(inplace=True,subset="sale")

#contar 0s en "sale"
#print(f"Cantidad de 0s en 'sale': {(df['sale'] == 0).sum()}")
# eliminar filas con "sale" igual a 0
df = df[df["sale"] != 0]


Y = df["sale"]
X = df.drop(columns="sale")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

X_train_processed = pipeline.transform(X_train)
X_test_processed = pipeline.transform(X_test)



models = {
    "OLS (Linear Regression)": LinearRegression(),
    "Random Forest":           RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost":                 XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    "CatBoost":                CatBoostRegressor(iterations=100, random_state=42, verbose=0),
    #"LightGBM":                LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)

}


#--------------Training-----------------------
results = []

for name, model in models.items():
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    results.append({
        "Model": name,
        "RMSE":  rmse,
        "MAE":   mae,
        "R²":    r2,
        "MAPE":  mape
    })

    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAPE : {mape:.4f}")

# ── 5. Summary table ───────────────────────────────────────────────────────────
summary = pd.DataFrame(results).sort_values("MAE", ascending=True)
print("\n── Summary ──────────────────────────────")
print(summary.to_string(index=False))

# ── 6. Save best model ─────────────────────────────────────────────────────────
best_name  = summary.iloc[0]["Model"]
best_model = models[best_name]

joblib.dump(best_model, "../Utils/best_model.pkl")
print(f"\n✓ Best model saved: {best_name}")




# ── 7. Hyperparameter Tuning + Cross Validation ───────────────────────────────
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Diccionario con espacios de búsqueda
param_distributions = {

    "OLS (Linear Regression)": {
        # LinearRegression tiene pocos hiperparámetros útiles
        "fit_intercept": [True, False]
    },

    "Random Forest": {
        "n_estimators": randint(50, 300),
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 5),
        "max_features": ["sqrt", "log2", None]
    },

    "XGBoost": {
        "n_estimators": randint(50, 300),
        "max_depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.3),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4)
    },

    "CatBoost": {
        "iterations": randint(50, 300),
        "depth": randint(4, 10),
        "learning_rate": uniform(0.01, 0.3),
        "l2_leaf_reg": randint(1, 10)
    }

    #"LightGBM": {
    #    "n_estimators": randint(50, 300),
    #    "max_depth": randint(3, 12),
    #    "learning_rate": uniform(0.01, 0.3),
    #    "num_leaves": randint(20, 100)
    #}
}

tuned_results = []
best_tuned_model = None
best_tuned_name = None
best_score = np.inf

for name, model in models.items():

    print(f"\n{'='*50}")
    print(f"Tuning model: {name}")
    print(f"{'='*50}")

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions[name],
        n_iter=10,
        cv=5,
        scoring="neg_mean_absolute_error",
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train_processed, y_train)

    tuned_model = random_search.best_estimator_

    # Predicciones
    y_pred = tuned_model.predict(X_test_processed)

    # Métricas
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    tuned_results.append({
        "Model": name,
        "Best Params": random_search.best_params_,
        "CV Score (MAE)": -random_search.best_score_,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "MAPE": mape
    })

    print(f"\nBest Params: {random_search.best_params_}")
    print(f"CV MAE      : {-random_search.best_score_:.4f}")
    print(f"Test RMSE   : {rmse:.4f}")
    print(f"Test MAE    : {mae:.4f}")
    print(f"Test R²     : {r2:.4f}")
    print(f"Test MAPE   : {mape:.4f}")

    # Guardar el mejor modelo tuneado global
    if mae < best_score:
        best_score = mae
        best_tuned_model = tuned_model
        best_tuned_name = name

# ── 8. Summary tuning table ───────────────────────────────────────────────────
tuned_summary = pd.DataFrame(tuned_results).sort_values("MAE", ascending=True)

print("\n── Tuned Models Summary ──────────────────────────────")
print(tuned_summary.to_string(index=False))

# Opcional: guardar resultados como CSV
tuned_summary.to_csv("../Utils/tuned_models_results.csv", index=False)

# ── 9. Save best tuned model ──────────────────────────────────────────────────
joblib.dump(best_tuned_model, "../Utils/best_tuned_model.pkl")

print(f"\n✓ Best tuned model saved: {best_tuned_name}")
