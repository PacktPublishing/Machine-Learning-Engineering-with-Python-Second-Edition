from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# Define datasets
# Example settings
n_samples = 300
outliers_fraction = 0.15
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers

blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
data = make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5, **blobs_params)[0]

class OutlierDetector(object):
    def __init__(self, model=None):
        if model is not None:
            self.model = model

        self.pipeline = make_pipeline(StandardScaler(), self.model)

    def detect(self, data):
        return self.pipeline.fit(data).predict(data)




if __name__ == "__main__":


    model = IsolationForest(behaviour='new',
                    contamination=outliers_fraction,
                    random_state=42)

    detector = OutlierDetector(model=model)

    result = detector.detect(data)

