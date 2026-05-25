import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import log1p_transform

class DateDifferenceTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        # convertir a datetime
        X["creation_date"] = pd.to_datetime(
            X["creation_date"],
            dayfirst=True,
            errors="coerce"
        )

        X["delivery_date"] = pd.to_datetime(
            X["delivery_date"],
            dayfirst=True,
            errors="coerce"
        )

        # diferencia en días
        X["delta_time"] = (
            X["delivery_date"]
            - X["creation_date"]
        ).dt.days

        # eliminar columnas originales
        X = X.drop(columns=[
            "creation_date",
            "delivery_date"
        ])

        return X

# ---------------------------
# 1. Cargar datos
# ---------------------------

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

#X["delta_time"] = (pd.to_datetime(X["delivery_date"]) - pd.to_datetime(X["creation_date"])).dt.days
#X.drop(columns=["creation_date","delivery_date"],inplace=True)


# ---------------------------
# 3. Split
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)

# ---------------------------
# 4. Tipos de columnas
# ---------------------------

#num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
#cat_cols = X_train.select_dtypes(include=["object", "category"]).columns


temp = DateDifferenceTransformer().transform(X_train.copy())

num_cols = temp.select_dtypes(include=["int64","float64"]).columns
cat_cols = temp.select_dtypes(include=["object","category"]).columns


# ---------------------------
# 5. Pipelines
# ---------------------------
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


# ---------------------------
# 6. Pipeline final
# ---------------------------
pipeline = Pipeline([
    ("date_features", DateDifferenceTransformer()),
    ("preprocessor", preprocessor)
])


# ---------------------------
# 7. Fit
# ---------------------------
pipeline.fit(X_train)

# ---------------------------
# 8. Transform
# ---------------------------
X_train_processed = pipeline.transform(X_train)
X_test_processed = pipeline.transform(X_test)

print(type(X_train_processed))  # <- pandas DataFrame
print(X_train_processed.head())

# ---------------------------
# 9. Guardar .pkl
# ---------------------------
artifact = {
    "pipeline": pipeline
}

joblib.dump(artifact, "../Utils/preprocessing_pipeline.pkl")

