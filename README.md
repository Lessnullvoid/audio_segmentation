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

## Prerequisites

- Python 3.7 or later
- Install dependencies:
  ```bash
  pip install -r requirements.txt

  ## Usage
	•	Load Audio File: Select a WAV file for analysis.
	•	Adjust Segmentation Threshold: Modify the sensitivity of feature detection.
	•	Visualize Features: View waveform with feature overlays.
	•	Cluster Segments: Reduce redundant segments automatically.
	•	Save Segments: Export processed segments as individual WAV files.
