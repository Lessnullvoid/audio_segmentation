import numpy as np
from sklearn.cluster import KMeans
from utils import is_silent_segment

def segment_audio(features, threshold=0.1):
    """Segment audio based on all features."""
    print("\nStarting audio segmentation process...")
    
    # Combine all events
    print("Combining feature events...")
    all_events = np.concatenate([
        features["transients"],
        features["beats"],
        features["spectral_centroid"][0][np.where(np.diff(features["spectral_centroid"][1]) > threshold)]
    ])
    all_events = np.sort(np.unique(all_events))
    print(f"Found {len(all_events)} potential segment boundaries")
    
    # Create segments
    print("\nGenerating segments...")
    segments = [(all_events[i], all_events[i+1]) for i in range(len(all_events) - 1)]
    print(f"Created {len(segments)} initial segments")
    
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
    """Segment audio by detected beats with adaptive segment merging"""
    print("\nStarting beat-based segmentation...")
    segments = []
    beats = features["beats"]
    
    if len(beats) < 2:
        print("Not enough beats detected for segmentation")
        print("Falling back to transient-based segmentation...")
        return segment_by_transients(features, min_segment_length)
    
    print(f"Processing {len(beats)} detected beats...")
    
    # Initialize temporary segment
    current_start = beats[0]
    current_duration = 0
    
    for i in range(1, len(beats)):
        segment_duration = beats[i] - current_start
        
        # If adding this beat makes the segment too long, save current segment and start new one
        if segment_duration >= min_segment_length:
            segments.append((current_start, beats[i]))
            current_start = beats[i]
            current_duration = 0
            
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(beats)} beats...")
    
    # Add the last segment if it meets the minimum length
    if len(beats) > 1 and beats[-1] - current_start >= min_segment_length:
        segments.append((current_start, beats[-1]))
    
    print(f"\nBeat segmentation complete:")
    print(f"- Total beats processed: {len(beats)}")
    print(f"- Segments created: {len(segments)}")
    
    if not segments:
        print("\nNo valid segments found using beats")
        print("Falling back to transient-based segmentation...")
        return segment_by_transients(features, min_segment_length)
    
    return segments

def segment_by_transients(features, min_segment_length=0.1):
    """Segment audio by detected transients with adaptive segment merging"""
    print("\nStarting transient-based segmentation...")
    segments = []
    transients = features["transients"]
    
    if len(transients) < 2:
        print("Not enough transients detected for segmentation")
        return []
    
    print(f"Processing {len(transients)} detected transients...")
    
    # Initialize temporary segment
    current_start = transients[0]
    current_duration = 0
    
    for i in range(1, len(transients)):
        segment_duration = transients[i] - current_start
        
        # If we have reached or exceeded the minimum length, create a segment
        if segment_duration >= min_segment_length:
            segments.append((current_start, transients[i]))
            current_start = transients[i]
            current_duration = 0
            
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(transients)} transients...")
    
    # Add the last segment if it meets the minimum length
    if len(transients) > 1 and transients[-1] - current_start >= min_segment_length:
        segments.append((current_start, transients[-1]))
    
    print(f"\nTransient segmentation complete:")
    print(f"- Total transients processed: {len(transients)}")
    print(f"- Segments created: {len(segments)}")
    
    return segments

def segment_by_frequency(features, min_freq=100, max_freq=2000, min_segment_length=0.1):
    """Segment audio by frequency content with adaptive segment merging"""
    print("\nStarting frequency-based segmentation...")
    times, spectral_centroid = features["spectral_centroid"]
    
    print("Analyzing frequency content...")
    print(f"Frequency range: {min_freq}Hz - {max_freq}Hz")
    
    segments = []
    start_time = None
    current_start = None
    
    for i, (time, freq) in enumerate(zip(times, spectral_centroid)):
        if min_freq <= freq <= max_freq:
            if current_start is None:
                current_start = time
        else:
            if current_start is not None:
                segment_duration = time - current_start
                if segment_duration >= min_segment_length:
                    segments.append((current_start, time))
                current_start = None
                
        if (i + 1) % (len(times) // 10) == 0:
            print(f"Processed {i + 1}/{len(times)} time points...")
    
    # Add the final segment if it meets the minimum length
    if current_start is not None and times[-1] - current_start >= min_segment_length:
        segments.append((current_start, times[-1]))
    
    print(f"\nFrequency segmentation complete:")
    print(f"- Total time points analyzed: {len(times)}")
    print(f"- Segments created: {len(segments)}")
    
    return segments

def segment_by_onsets(features, min_segment_length=0.1):
    """Segment audio by onset detection with adaptive segment merging"""
    print("\nStarting onset-based segmentation...")
    segments = []
    onsets = features["onsets"]
    
    if len(onsets) < 2:
        print("Not enough onsets detected for segmentation")
        return []
    
    print(f"Processing {len(onsets)} detected onsets...")
    
    # Initialize temporary segment
    current_start = onsets[0]
    current_duration = 0
    
    for i in range(1, len(onsets)):
        segment_duration = onsets[i] - current_start
        
        # If we have reached or exceeded the minimum length, create a segment
        if segment_duration >= min_segment_length:
            if not is_silent_segment(features["audio_file"], current_start, onsets[i]):
                segments.append((current_start, onsets[i]))
            current_start = onsets[i]
            current_duration = 0
            
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(onsets)} onsets...")
    
    # Add the last segment if it meets the minimum length
    if len(onsets) > 1 and onsets[-1] - current_start >= min_segment_length:
        if not is_silent_segment(features["audio_file"], current_start, onsets[-1]):
            segments.append((current_start, onsets[-1]))
    
    print(f"\nOnset segmentation complete:")
    print(f"- Total onsets processed: {len(onsets)}")
    print(f"- Segments created: {len(segments)}")
    
    return segments