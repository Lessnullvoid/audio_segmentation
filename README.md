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

├── requirements.txt      # List of dependencies

└── audio_files/          # Directory for your test WAV files

## Features

- **Feature Detection**: Extract transients, beats, and spectral features using `feature_detection.py`.
- **Segmentation**: Chop audio into meaningful segments based on detected features.
- **Clustering**: Reduce redundant segments by clustering and selecting the most unique ones.
- **Visualization**: Display audio waveforms and feature overlays for analysis.
- **GUI**: Intuitive PyQt5 interface for user interaction.
- **Batch Processing**: Process multiple audio files simultaneously for efficiency.
- **Customizable Parameters**: Fine-tune detection and segmentation parameters to suit different audio types.
- **Clustered Storage with Metadata**: Save segmented audio files into structured directories based on their cluster labels, with filenames containing frequency and musical note metadata.

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

## Usage
- Load Audio File: Select a WAV file for analysis.
- Adjust Segmentation Threshold: Modify the sensitivity of feature detection.
- Visualize Features: View waveform with feature overlays.
- Cluster Segments: Reduce redundant segments automatically.
- Save Segments: Export processed segments as individual WAV files.

## Clustered Storage with Metadata

The software allows you to save segmented audio files into directories organized by cluster labels. Each segment is saved with a filename that includes its spectral centroid frequency and the corresponding musical note. This feature is particularly useful for organizing and retrieving audio segments based on their acoustic characteristics, facilitating easier analysis and processing in subsequent tasks.
