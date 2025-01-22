"""
This software is designed for audio feature detection and analysis using the librosa library.
It provides functionalities to detect transients, beats, and visualize audio waveforms.

Key Features:
- Transient Detection: Identifies sudden changes in the audio signal, which are often indicative of note onsets or other significant events.
- Beat Tracking: Analyzes the rhythm of the audio to determine the tempo and locate beat positions.
- Waveform Visualization: Plots the audio waveform for visual inspection and analysis.

The software is intended for use in music information retrieval, audio analysis, and other applications where understanding the structure and content of audio signals is important.
"""

from feature_detection import detect_features
from segmentation import segment_audio, cluster_segments
from visualization import plot_features
from utils import chop_audio

def main():
    audio_file = "1.Sound_Sampler.wav"
    
    # Step 1: Detect features
    features = detect_features(audio_file)
    
    # Step 2: Visualize features
    plot_features(audio_file, features)
    
    # Step 3: Segment audio
    segments = segment_audio(features)
    print(f"Detected Segments: {segments}")
    
    # Step 4: Chop audio into segments
    chop_audio(audio_file, segments)

    # List of segment files
    segment_files = [f"segment_{i+1}.wav" for i in range(len(segments))]

    # Cluster segments and get unique ones
    unique_segments = cluster_segments(segment_files, n_clusters=5)

    # unique_segments now contains the most unique segments

if __name__ == "__main__":
    main()