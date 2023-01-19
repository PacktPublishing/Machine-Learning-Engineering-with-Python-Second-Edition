from sklearn.datasets import make_blobs

def create_data():
    # Define datasets
    # Example settings
    n_samples = 300
    outliers_fraction = 0.15
    n_outliers = int(outliers_fraction * n_samples)
    n_inliers = n_samples - n_outliers

    blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
    data = make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5, **blobs_params)[0]
    return data