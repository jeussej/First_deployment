import typing as t

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import numpy as np


def build_estimator(hyperparams: t.Dict[str, t.Any]):
    estimator_mapping = get_estimator_mapping()
    steps = []
    for name, params in hyperparams.items():
        print(name,params)
        estimator = estimator_mapping[name](**params)
        steps.append((name, estimator))
    model = Pipeline(steps)
    return model


def get_estimator_mapping():
    return {
        "regressor": RandomForestRegressor,
        "BikeRentalFeatureExtractor": BikeRentalFeatureExtractor,
    }



class BikeRentalFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self , X , y = None):
        return self
    
    def transform(self , X):
        X_=X.copy()
        X_["last_y"]=np.append(np.array(np.mean(X_["cnt"])),(np.array(X_["cnt"][:-1])))
        X_=X_.drop(["instant","dteday","cnt"],axis=1)
        return X_
