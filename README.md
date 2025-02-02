# Audio Segmentation Tool

This repository contains a Python-based audio segmentation tool with a graphical user interface (GUI) for feature detection, segmentation, and clustering.

## Directory Structure

audio_segmentation/
│
├── main.py               # Entry point to run the software
├── ui.py                 # Contains the UI implementation
├── feature_detection.py  # Feature detection logic
├── segmentation.py       # Segmentation logic
├── clustering.py         # Clustering similar segments
├── utils.py              # Audio chopping utility
├── visualization.py      # Visualization logic
├── audio_player.py       # Efficient audio playback handling
├── requirements.txt      # List of dependencies
└── audio_files/          # Directory for your test WAV files

## Features

- **Feature Detection**: Extract transients, beats, and spectral features using `feature_detection.py`.
- **Segmentation**: Chop audio into meaningful segments based on detected features.
- **Manual Segmentation**: Allows users to manually segment audio using a spectrogram view, providing precise control over segment boundaries.
- **Clustering**: Reduce redundant segments by clustering and selecting the most unique ones.
- **Visualization**: Display audio waveforms and feature overlays for analysis.
- **GUI**: Intuitive PyQt5 interface for user interaction, with organized layout for easy access to functionalities.
- **Batch Processing**: Process multiple audio files simultaneously for efficiency.
- **Customizable Parameters**: Fine-tune detection and segmentation parameters to suit different audio types.
- **Clustered Storage with Metadata**: Save segmented audio files into structured directories based on their cluster labels.
- **Efficient Playback**: Optimized audio playback system using pygame for better performance.

## Recent Updates

- **Improved Audio Playback**: Implemented a new audio player system using pygame for more responsive playback
- **Enhanced UI Feedback**: Added visual feedback for playback state with color-coded buttons
- **Better Memory Management**: Optimized audio loading and playback to reduce memory usage
- **Responsive Controls**: Added play/stop toggle functionality with visual indicators

## Use Recommendation

This tool is ideal for audio engineers, music producers, and researchers who need to analyze and process large audio datasets. It is particularly useful for tasks such as:

- Preparing audio samples for machine learning models.
- Analyzing musical compositions for feature extraction.
- Automating the segmentation of long audio recordings for easier editing and manipulation.
- Reducing manual effort in identifying and clustering similar audio segments.

## Prerequisites

- Python 3.7 or later
- Install dependencies:
```bash
pip install -r requirements.txt
```

## Detailed Usage

1. **Starting the Application**:
```bash
python main.py
```

2. **Basic Operations**:
   - Load Audio File: Click "Load Audio" to select a WAV file
   - Choose Segmentation Method: Select from "By Beats", "By Transients", or "By Frequency"
   - Adjust Parameters: Modify thresholds and settings as needed
   - Process Audio: Click "Segment and Visualize"

3. **Playback Controls**:
   - Select a segment from the list
   - Click "Play Selected Segment" to play
   - Button turns red during playback
   - Click again to stop playback

4. **Manual Segmentation**:
   - Enable manual mode using the toggle button
   - Click on waveform to add segment boundaries
   - Use 'C' key to clear last boundary
   - Save segments using "Save Manual Segments"

5. **Clustering**:
   - Adjust number of clusters using the slider
   - Click "Cluster Segments" to group similar segments
   - Review clusters in the list view
   - Play representative segments from each cluster

## Clustered Storage with Metadata

The software allows you to save segmented audio files into directories organized by cluster labels. Each segment is saved with a filename that includes its spectral centroid frequency and the corresponding musical note. This feature is particularly useful for organizing and retrieving audio segments based on their acoustic characteristics, facilitating easier analysis and processing in subsequent tasks.

## Performance Tips

- For best playback performance, use WAV files with standard sample rates (44.1kHz or 48kHz)
- Keep segment lengths reasonable (typically 0.1-10 seconds) for optimal clustering
- Use the manual segmentation mode for precise control over important boundaries
- Adjust clustering parameters based on your specific audio material
- Save frequently when working with large files or many segments
