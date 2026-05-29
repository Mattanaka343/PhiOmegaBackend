import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from transformers import log1p_transform

from pipeline_classes import DateDifferenceTransformer, TotalAccesorialCreator

df = pd.read_csv('DataVaxModelos.csv')
df = df.drop(columns=df.columns[0])

df= df.drop_duplicates()
df.dropna(inplace=True,subset="sale")

df = df[df["sale"] != 0]


Y = df["sale"]
X = df.drop(columns="sale")



X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)
temp = DateDifferenceTransformer().transform(X_train.copy())

num_cols = temp.select_dtypes(include=["int64","float64"]).columns
cat_cols = temp.select_dtypes(include=["object","category"]).columns

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

pipeline = Pipeline([('date_converter',DateDifferenceTransformer()), ('accessorial_creator',TotalAccesorialCreator), ('preprocessor', preprocessor) ])
pipeline.fit(X_train)
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

joblib.dump(artifact, "preprocessing_pipeline.pkl")

