import numpy as np
from sklearn.cluster import KMeans

def segment_audio(features, threshold=0.1):
    all_events = np.concatenate([
        features["transients"],
        features["beats"],
        features["spectral_centroid"][0][np.where(np.diff(features["spectral_centroid"][1]) > threshold)]
    ])
    all_events = np.sort(np.unique(all_events))
    segments = [(all_events[i], all_events[i+1]) for i in range(len(all_events) - 1)]
    return segments

def cluster_segments(segment_files, n_clusters):
    features = [extract_features(f) for f in segment_files]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    unique_segments = []

    for cluster in range(n_clusters):
        cluster_indices = np.where(kmeans.labels_ == cluster)[0]
        cluster_features = [features[i] for i in cluster_indices]
        centroid = kmeans.cluster_centers_[cluster]
        closest_index = cluster_indices[np.argmin([np.linalg.norm(f - centroid) for f in cluster_features])]
        unique_segments.append(segment_files[closest_index])

    return unique_segments