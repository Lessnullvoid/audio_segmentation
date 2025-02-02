import numpy as np
from sklearn.cluster import KMeans
from utils import is_silent_segment

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

def segment_by_beats(features, min_segment_length=0.1):
    """Segment audio by detected beats"""
    segments = []
    beats = features["beats"]
    
    if len(beats) < 2:
        print("Not enough beats detected for segmentation")
        # Fall back to transients if no beats detected
        return segment_by_transients(features, min_segment_length)
    
    for i in range(len(beats) - 1):
        start = beats[i]
        end = beats[i + 1]
        
        # Check minimum segment length
        if end - start >= min_segment_length:
            if not is_silent_segment(features["audio_file"], start, end):
                segments.append((start, end))
    
    if not segments:
        print("No valid segments found using beats, falling back to transients")
        return segment_by_transients(features, min_segment_length)
        
    return segments

def segment_by_transients(features, min_segment_length=0.1):
    """Segment audio by detected transients"""
    segments = []
    transients = features["transients"]
    
    for i in range(len(transients) - 1):
        start = transients[i]
        end = transients[i + 1]
        
        # Check minimum segment length
        if end - start >= min_segment_length:
            if not is_silent_segment(features["audio_file"], start, end):
                segments.append((start, end))
    
    return segments

def segment_by_frequency(features, min_freq=20, max_freq=20000, min_segment_length=0.1):
    """Segment audio by frequency content"""
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
    
    segments = []
    for i in range(len(segments) - 1):
        start = segments[i]
        end = segments[i + 1]
        
        # Check minimum segment length
        if end - start >= min_segment_length:
            if not is_silent_segment(features["audio_file"], start, end):
                segments.append((start, end))
    
    return segments