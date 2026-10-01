# utils.py snippet
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_eng = np.asarray(X).copy()

        # DOMAIN-DRIVEN FEATURE ENGINEERING:
        X_eng = np.column_stack([
            X_eng,
            #Compressor Discharge Pressure to Ambient Pressure ratio
            X_eng[:,8]/(X_eng[:,1] + 1e-8),

            # Turbine temperature drop
            X_eng[:, 5] - X_eng[:,6],

            #Air density
            X_eng[:, 1] * 100 / ( X_eng[:,0]+ 273.15 + 1e-8),

            # Energy Per Airflow
            X_eng[:,7]/( X_eng[:,3] + 1e-8),

            #Compressor work proxy
            X_eng[:, 8] * X_eng[:, 0],

            #Core Thermal Energy
            X_eng[:,8] * X_eng[:,5],

            #Thermal_Severity_Index
            X_eng[:,5] ** 2 / (X_eng[:,8] + 1e-8),

            #Combustion Intensity Proxy
            X_eng[:,5] * X_eng[:,4] / (X_eng[:,0] + 1e-8),

            #Absolute_Humidity_Effect
            X_eng[:,2] * X_eng[:,0]
        ])
        return X_eng


class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5): self.factor = factor
    def fit(self, X, y=None):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        self.lower_bounds_ = [np.percentile(X_arr[:,i],25) - self.factor*(np.percentile(X_arr[:,i],75)-np.percentile(X_arr[:,i],25)) for i in range(X_arr.shape[1])]
        self.upper_bounds_ = [np.percentile(X_arr[:,i],75) + self.factor*(np.percentile(X_arr[:,i],75)-np.percentile(X_arr[:,i],25)) for i in range(X_arr.shape[1])]
        return self
    def transform(self, X):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X.copy()
        for i in range(X_arr.shape[1]):
            X_arr[:,i] = np.clip(X_arr[:,i], self.lower_bounds_[i], self.upper_bounds_[i])
        return X_arr