import librosa
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def cluster_segments(audio_file, segments, eps=0.5, min_samples=1):
    """
    Cluster segments based on similarity using spectral features.
    Returns representative segments only.
    """
    y, sr = librosa.load(audio_file)
    
    # Extract features for each segment
    features = []
    for start, end in segments:
        segment = y[int(start * sr):int(end * sr)]
        spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr).mean()
        mfcc = librosa.feature.mfcc(y=segment, sr=sr).mean(axis=1)
        features.append(np.concatenate(([spectral_centroid], mfcc)))
    
    # Standardize features
    features = StandardScaler().fit_transform(features)
    
    # Cluster using DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    cluster_labels = clustering.labels_
    
    # Select one representative per cluster
    unique_segments = []
    for cluster in np.unique(cluster_labels):
        indices = np.where(cluster_labels == cluster)[0]
        if len(indices) > 0:
            # Merge segments in the same cluster
            start_times = [segments[i][0] for i in indices]
            end_times = [segments[i][1] for i in indices]
            merged_segment = (min(start_times), max(end_times))
            unique_segments.append(merged_segment)
    
    return unique_segments

def cluster_segments_kmeans(audio_file, segments, n_clusters=10):
    """
    Cluster segments into a specific number of clusters using k-means.
    Returns representative segments and cluster labels.
    """
    y, sr = librosa.load(audio_file)
    features = []
    for start, end in segments:
        segment = y[int(start * sr):int(end * sr)]
        spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr).mean()
        mfcc = librosa.feature.mfcc(y=segment, sr=sr).mean(axis=1)
        features.append(np.concatenate(([spectral_centroid], mfcc)))
    
    features = StandardScaler().fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    
    representative_segments = []
    for cluster in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        if len(cluster_indices) > 0:
            cluster_center = kmeans.cluster_centers_[cluster]
            closest_idx = cluster_indices[
                np.argmin([np.linalg.norm(features[idx] - cluster_center) for idx in cluster_indices])
            ]
            representative_segments.append(segments[closest_idx])
    
    return representative_segments, cluster_labels