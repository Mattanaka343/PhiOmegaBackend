import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer


df = pd.read_csv('Data\DataVaxModelos.csv')
df = df.drop(columns=df.columns[0])

df= df.drop_duplicates()
df.dropna(inplace=True,subset="sale")

Y = df["sale"]
X= df.drop(columns="sale")

X["delta_time"] = (pd.to_datetime(X["delivery_date"]) - pd.to_datetime(X["creation_date"])).dt.days
X.drop(columns=["creation_date","delivery_date"],inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)



num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns


from transformers import log1p_transform


numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("log", FunctionTransformer(log1p_transform)),
    ("scaler", StandardScaler())
])


categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_cols),
    ("cat", categorical_pipeline, cat_cols)
])

preprocessor.set_output(transform="pandas")

pipeline = Pipeline([
    ("preprocessor", preprocessor)
])



pipeline.fit(X_train)

X_train_processed = pipeline.transform(X_train)
X_test_processed = pipeline.transform(X_test)


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


models = {
    "OLS (Linear Regression)": LinearRegression(),
    "Random Forest":           RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost":                 XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    "CatBoost":                CatBoostRegressor(iterations=100, random_state=42, verbose=0),

}


#--------------Training-----------------------
results = []

for name, model in models.items():
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "RMSE":  rmse,
        "MAE":   mae,
        "R²":    r2,
    })

    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")

# ── 5. Summary table ───────────────────────────────────────────────────────────
summary = pd.DataFrame(results).sort_values("MAE", ascending=True)
print("\n── Summary ──────────────────────────────")
print(summary.to_string(index=False))

# ── 6. Save best model ─────────────────────────────────────────────────────────
best_name  = summary.iloc[0]["Model"]
best_model = models[best_name]

joblib.dump(best_model, "Utils/best_model.pkl")
print(f"\n✓ Best model saved: {best_name}")
