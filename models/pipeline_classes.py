import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class DateDifferenceTransformer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        creation_col="creation_date",
        delivery_col="delivery_date",
        output_col="delta_time"
    ):
        self.creation_col = creation_col
        self.delivery_col = delivery_col
        self.output_col = output_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        X[self.creation_col] = pd.to_datetime(
            X[self.creation_col],
            dayfirst=True,
            errors="coerce"
        )

        X[self.delivery_col] = pd.to_datetime(
            X[self.delivery_col],
            dayfirst=True,
            errors="coerce"
        )

        X[self.output_col] = (
            X[self.delivery_col]
            - X[self.creation_col]
        ).dt.days

        X = X.drop(
            columns=[self.creation_col, self.delivery_col]
        )

        return X


class TotalAccesorialCreator(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        sale_col="accesorials_sale",
        cost_col="accesorials_cost",
        output_col="accesorials_total"
    ):
        self.sale_col = sale_col
        self.cost_col = cost_col
        self.output_col = output_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        X[self.output_col] = (
            X[self.sale_col]
            + X[self.cost_col]
        )

        X = X.drop(
            columns=[self.sale_col, self.cost_col]
        )

        return X