import librosa
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from utils import is_silent_segment

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

def cluster_segments_kmeans(audio_file, segments, n_clusters=10, similarity_threshold=0.85):
    """
    Cluster segments into a specific number of clusters using k-means and remove similar segments.
    Returns representative segments and cluster labels.
    """
    if not segments:
        print("No segments provided for clustering")
        return [], []

    # Filter out silent segments first
    non_silent_segments = []
    for start, end in segments:
        if not is_silent_segment(audio_file, start, end):
            non_silent_segments.append((start, end))
        else:
            print(f"Skipping silent segment: {start:.2f}s - {end:.2f}s")

    if not non_silent_segments:
        print("No non-silent segments found!")
        return [], []

    segments = non_silent_segments
    y, sr = librosa.load(audio_file)
    features = []
    segment_features = []

    # Extract features for each segment
    for start, end in segments:
        segment = y[int(start * sr):int(end * sr)]
        if len(segment) > 0:
            # Basic features for clustering
            spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr).mean()
            mfcc = librosa.feature.mfcc(y=segment, sr=sr).mean(axis=1)
            features.append(np.concatenate(([spectral_centroid], mfcc)))
            
            # Detailed features for similarity comparison
            chroma = librosa.feature.chroma_stft(y=segment, sr=sr).mean(axis=1)
            spectral_contrast = librosa.feature.spectral_contrast(y=segment, sr=sr).mean(axis=1)
            segment_features.append(np.concatenate([mfcc, chroma, spectral_contrast]))

    if not features:
        print("No features could be extracted from segments")
        return [], []

    # Convert to numpy array and reshape if necessary
    features = np.array(features)
    if features.ndim == 1:
        features = features.reshape(-1, 1)

    # Scale features
    features = StandardScaler().fit_transform(features)
    segment_features = StandardScaler().fit_transform(np.array(segment_features))

    # Perform clustering
    kmeans = KMeans(n_clusters=min(n_clusters, len(features)), random_state=42)
    cluster_labels = kmeans.fit_predict(features)

    # Find representative segments and remove similar ones
    representative_segments = []
    final_labels = []
    used_indices = set()

    for cluster in range(kmeans.n_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        if len(cluster_indices) > 0:
            # Find the segment closest to cluster center
            cluster_center = kmeans.cluster_centers_[cluster]
            distances = [np.linalg.norm(features[idx] - cluster_center) for idx in cluster_indices]
            closest_idx = cluster_indices[np.argmin(distances)]
            
            # Check similarity with already selected segments
            is_unique = True
            for used_idx in used_indices:
                similarity = np.dot(segment_features[closest_idx], segment_features[used_idx]) / \
                           (np.linalg.norm(segment_features[closest_idx]) * np.linalg.norm(segment_features[used_idx]))
                if similarity > similarity_threshold:
                    is_unique = False
                    print(f"Skipping similar segment {closest_idx} (similarity: {similarity:.2f})")
                    break
            
            if is_unique:
                representative_segments.append(segments[closest_idx])
                final_labels.append(cluster)
                used_indices.add(closest_idx)

    print(f"Selected {len(representative_segments)} unique segments from {len(segments)} original segments")
    return representative_segments, final_labels