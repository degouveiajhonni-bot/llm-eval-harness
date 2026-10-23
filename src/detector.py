import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self, contamination=0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)

    def fit_predict(self, df: pd.DataFrame, feature_columns: list):
        X = df[feature_columns].values
        preds = self.model.fit_predict(X)
        df['anomaly'] = preds
        df['is_anomaly'] = df['anomaly'].apply(lambda x: True if x == -1 else False)
        return df

if __name__ == "__main__":
    np.random.seed(42)
    data = pd.DataFrame({
        'cpu_load': np.random.normal(50, 5, 100),
        'memory_usage': np.random.normal(70, 10, 100)
    })
    data.loc[10, 'cpu_load'] = 99.9
    
    detector = AnomalyDetector()
    results = detector.fit_predict(data, ['cpu_load', 'memory_usage'])
    print("Anomaly Guard ML Initialized Successfully.")
