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
    """Segment audio by detected beats"""
    print("\nStarting beat-based segmentation...")
    segments = []
    beats = features["beats"]
    
    if len(beats) < 2:
        print("Not enough beats detected for segmentation")
        print("Falling back to transient-based segmentation...")
        return segment_by_transients(features, min_segment_length)
    
    print(f"Processing {len(beats)} detected beats...")
    valid_segments = 0
    silent_segments = 0
    
    for i in range(len(beats) - 1):
        start = beats[i]
        end = beats[i + 1]
        
        # Check minimum segment length
        if end - start >= min_segment_length:
            if not is_silent_segment(features["audio_file"], start, end):
                segments.append((start, end))
                valid_segments += 1
            else:
                silent_segments += 1
                
        if (i + 1) % 10 == 0:  # Progress update every 10 beats
            print(f"Processed {i + 1}/{len(beats)} beats...")
    
    print(f"\nSegmentation complete:")
    print(f"- Total beats processed: {len(beats)}")
    print(f"- Valid segments created: {valid_segments}")
    print(f"- Silent segments skipped: {silent_segments}")
    
    if not segments:
        print("\nNo valid segments found using beats")
        print("Falling back to transient-based segmentation...")
        return segment_by_transients(features, min_segment_length)
    
    return segments

def segment_by_transients(features, min_segment_length=0.1):
    """Segment audio by detected transients"""
    print("\nStarting transient-based segmentation...")
    segments = []
    transients = features["transients"]
    
    if len(transients) < 2:
        print("Not enough transients detected for segmentation")
        return []
    
    print(f"Processing {len(transients)} detected transients...")
    valid_segments = 0
    short_segments = 0
    
    for i in range(len(transients) - 1):
        start = transients[i]
        end = transients[i + 1]
        
        if end - start >= min_segment_length:
            segments.append((start, end))
            valid_segments += 1
        else:
            short_segments += 1
            
        if (i + 1) % 20 == 0:  # Progress update every 20 transients
            print(f"Processed {i + 1}/{len(transients)} transients...")
    
    print(f"\nTransient segmentation complete:")
    print(f"- Total transients processed: {len(transients)}")
    print(f"- Valid segments created: {valid_segments}")
    print(f"- Short segments skipped: {short_segments}")
    
    return segments

def segment_by_frequency(features, min_freq=100, max_freq=2000, min_segment_length=0.1):
    """Segment audio by frequency content"""
    print("\nStarting frequency-based segmentation...")
    times, spectral_centroid = features["spectral_centroid"]
    
    print("Analyzing frequency content...")
    print(f"Frequency range: {min_freq}Hz - {max_freq}Hz")
    
    segments = []
    start_time = None
    segment_count = 0
    
    print("Generating segments based on frequency thresholds...")
    
    for time, freq in zip(times, spectral_centroid):
        if min_freq <= freq <= max_freq:
            if start_time is None:
                start_time = time
        else:
            if start_time is not None:
                if time - start_time >= min_segment_length:
                    segments.append((start_time, time))
                    segment_count += 1
                start_time = None
                
        if segment_count % 10 == 0 and segment_count > 0:
            print(f"Created {segment_count} segments so far...")
    
    # Add the final segment if open
    if start_time is not None:
        segments.append((start_time, times[-1]))
        segment_count += 1
    
    print(f"\nFrequency segmentation complete:")
    print(f"- Total time points analyzed: {len(times)}")
    print(f"- Segments created: {segment_count}")
    
    return segments

def segment_by_onsets(features, min_segment_length=0.1):
    """Segment audio by onset detection"""
    print("\nStarting onset-based segmentation...")
    segments = []
    onsets = features["onsets"]
    
    if len(onsets) < 2:
        print("Not enough onsets detected for segmentation")
        return []
    
    print(f"Processing {len(onsets)} detected onsets...")
    valid_segments = 0
    short_segments = 0
    silent_segments = 0
    
    for i in range(len(onsets) - 1):
        start = onsets[i]
        end = onsets[i + 1]
        
        if end - start >= min_segment_length:
            if not is_silent_segment(features["audio_file"], start, end):
                segments.append((start, end))
                valid_segments += 1
            else:
                silent_segments += 1
        else:
            short_segments += 1
            
        if (i + 1) % 20 == 0:  # Progress update every 20 onsets
            print(f"Processed {i + 1}/{len(onsets)} onsets...")
    
    print(f"\nOnset segmentation complete:")
    print(f"- Total onsets processed: {len(onsets)}")
    print(f"- Valid segments created: {valid_segments}")
    print(f"- Short segments skipped: {short_segments}")
    print(f"- Silent segments skipped: {silent_segments}")
    
    return segments