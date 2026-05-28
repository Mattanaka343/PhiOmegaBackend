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


artifact = joblib.load("preprocessing_pipeline.pkl")

pipeline = artifact["pipeline"]

df = pd.read_csv('DataVaxModelos.csv')
df = df.drop(columns=df.columns[0])

df= df.drop_duplicates()
df.dropna(inplace=True,subset="sale")

#contar 0s en "sale"
print(f"Cantidad de 0s en 'sale': {(df['sale'] == 0).sum()}")
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


# entrenar modelos con randomsearch cross validation
