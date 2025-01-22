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

def segment_by_beats(features):
    """
    Create segments based on beat times.
    """
    beat_times = features.get("beats", [])
    return [(beat_times[i], beat_times[i + 1]) for i in range(len(beat_times) - 1)]

def segment_by_transients(features):
    """
    Create segments based on transients.
    """
    transients = features.get("transients", [])
    return [(transients[i], transients[i + 1]) for i in range(len(transients) - 1)]

def segment_by_frequency(features, min_freq=100, max_freq=5000):
    """
    Create segments where the spectral centroid falls within a frequency range.
    """
    times, spectral_centroid = features.get("spectral_centroid", ([], []))
    segments = []
    start_time = None
    for time, freq in zip(times, spectral_centroid):
        if min_freq <= freq <= max_freq:
            if start_time is None:
                start_time = time
        else:
            if start_time is not None:
                segments.append((start_time, time))
                start_time = None
    # Add the final segment if open
    if start_time is not None:
        segments.append((start_time, times[-1]))
    return segments