'''
This script executes the DBSCAN clustering algorithm on the simulated taxi ride dataset.

It 
'''

from simulate_data import simulate_ride_data
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

#==========================================
# Clustering with DBSCAN
#==========================================
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics

def cluster_and_label(data: pd.DataFrame) -> dict:
    data = StandardScaler().fit_transform(data)
    db = DBSCAN(eps=0.3, min_samples=10).fit(data)

    # Find labels from the clustering
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)

    # print('Estimated number of clusters: %d' % n_clusters_)
    # print('Estimated number of noise points: %d' % n_noise_)
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(data, labels))

    # run_metadata = {
    #     'nClusters': n_clusters_,
    #     'nNoise': n_noise_,
    #     'silhouetteCoefficient': metrics.silhouette_score(data, labels),
    #     'labels': labels,
    # }
    data['label'] = labels
    return data

# if __name__ == "__main__":
    
#     df = simulate_ride_data()
    
#     logging.info('Simulating ride data ...')
#     X = df[['ride_dist', 'ride_time']]
    
#     logging.info('Clustering and labelling')
#     results = cluster_and_label(X, create_and_show_plot=True)
#     df['label'] = results['labels']
    
#     # Output your results to json
#     logging.info('Outputting to json ...')
#     df.to_json('taxi-labels.json', orient='records')