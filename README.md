# Audio Segmentation Tool

<p align="center">
  <img src="https://github.com/Lessnullvoid/audio_segmentation/blob/main/audio_seg_tool.png" alt="Audio Segmentation Tool Interface" width="800"/>
</p>

This repository contains a Python-based audio segmentation tool with a graphical user interface (GUI) for feature detection, segmentation, and clustering.

## Directory Structure
```
audio_segmentation/
├── main.py               # Entry point to run the software
├── ui.py                # Contains the UI implementation
├── feature_detection.py  # Feature detection logic
├── segmentation.py      # Segmentation logic
├── clustering.py        # Clustering similar segments
├── utils.py            # Audio chopping utility
├── visualization.py     # Visualization logic
├── audio_player.py      # Efficient audio playback handling
├── requirements.txt     # List of dependencies
└── audio_files/        # Directory for your test WAV files
```

## Features

- **Feature Detection**: Extract transients, beats, onsets, and spectral features
- **Multiple Segmentation Methods**:
  - Beat-based segmentation
  - Transient-based segmentation
  - Frequency range-based segmentation
  - Onset-based segmentation (New!)
- **Manual Segmentation**: Allows users to manually segment audio using a spectrogram view
- **Clustering**: Reduce redundant segments by clustering and selecting the most unique ones
- **Smart Segment Selection**: Request specific number of segments and get the most representative ones
- **Visualization**: Display audio waveforms and feature overlays for analysis
- **GUI**: Intuitive PyQt5 interface for user interaction
- **Batch Processing**: Process multiple audio files simultaneously for efficiency
- **Customizable Parameters**: Fine-tune detection and segmentation parameters
- **Clustered Storage**: Save segmented audio files into structured directories
- **Efficient Playback**: Optimized audio playback system using pygame

## Recent Updates

- **Added Onset Detection**: New segmentation method using onset detection
- **Improved Segment Selection**: Smart selection of representative segments when specific count is requested
- **Enhanced Terminal Feedback**: Detailed progress and status messages during processing
- **Improved Audio Playback**: New audio player system using pygame for more responsive playback
- **Enhanced UI Feedback**: Visual feedback for playback state with color-coded buttons
- **Better Memory Management**: Optimized audio loading and playback
- **Responsive Controls**: Play/stop toggle functionality with visual indicators

## Use Recommendation

This tool is ideal for audio engineers, music producers, and researchers who need to analyze and process large audio datasets. It is particularly useful for:

- Preparing audio samples for machine learning models
- Analyzing musical compositions for feature extraction
- Automating the segmentation of long audio recordings
- Reducing manual effort in identifying and clustering similar audio segments

## Prerequisites

- Python 3.7 or later
- Install dependencies:
```bash
pip install -r requirements.txt
```

## Detailed Usage

### 1. Starting the Application
```bash
python main.py
```

### 2. Basic Operations
- Load Audio File: Click "Load Audio" to select a WAV file
- Choose Segmentation Method: Select from "By Beats", "By Transients", or "By Frequency"
- Adjust Parameters: Modify thresholds and settings as needed
- Process Audio: Click "Segment and Visualize"

### 3. Playback Controls
- Select a segment from the list
- Click "Play Selected Segment" to play
- Button turns red during playback
- Click again to stop playback

### 4. Manual Segmentation
- Enable manual mode using the toggle button
- Click on waveform to add segment boundaries
- Use 'C' key to clear last boundary
- Save segments using "Save Manual Segments"

### 5. Clustering
- Adjust number of clusters using the slider
- Click "Cluster Segments" to group similar segments
- Review clusters in the list view
- Play representative segments from each cluster

## Clustered Storage with Metadata

The software organizes segmented audio files into directories by cluster labels. Each segment's filename includes its spectral centroid frequency and musical note, making it easy to analyze and process segments based on their acoustic characteristics.

## Performance Tips

- Use WAV files with standard sample rates (44.1kHz or 48kHz)
- Keep segment lengths reasonable (0.1-10 seconds) for optimal clustering
- Use manual segmentation mode for precise control
- Adjust clustering parameters based on your audio material
- Save frequently when working with large files
- For percussive audio, try the new onset-based segmentation
- When requesting specific segment count, ensure it's reasonable for your audio length

## TODO

### 1. Session Management System
- [ ] Implement JSON-based session storage
  - [ ] Save segment positions and metadata
  - [ ] Save clustering information
  - [ ] Store audio file path and parameters
  - [ ] Add auto-save functionality
  - [ ] Create session recovery on program start
  - [ ] Add manual session save/load options

### 2. Onset-based Segmentation
- [x] Add onset detection mode
- [x] Implement onset detection algorithm
- [x] Add onset sensitivity controls
- [x] Create onset visualization overlay
- [x] Add onset threshold adjustment
- [x] Implement onset-based segment generation
- [x] Add onset type selection (percussive/harmonic)

### Future Improvements
- [ ] Add batch processing for onset detection
- [ ] Implement segment preview in list view
- [ ] Add waveform zoom synchronization
- [ ] Improve error handling and user feedback
- [ ] Add export options for session data
- [ ] Create preset system for different audio types
