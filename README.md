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

- **Audio Loading and Visualization**: Load and visualize WAV files with waveform display
- **Multiple Segmentation Methods**:
  - Beat-based segmentation
  - Transient-based segmentation
  - Frequency range-based segmentation
  - Onset-based segmentation
- **Segmentation Controls**:
  - Minimum and maximum time constraints
  - Similarity threshold filtering
  - Automatic merging of short segments
- **Clustering Capabilities**:
  - Optional clustering of segments by similarity
  - Preservation of all segments during clustering
  - Similarity scores to cluster centers
  - Organized storage in cluster-based folders
- **Manual Segmentation**: Create segments manually using the waveform view
- **Segment Management**:
  - Play/stop individual segments
  - Clear all segments
  - View segment duration and time range
- **Export Features**:
  - Save segments with or without clustering
  - Organized folder structure based on clustering state
  - Metadata inclusion in filenames

## Recent Updates

- **Enhanced Time Controls**: Added minimum and maximum duration constraints for segments
- **Improved Similarity Processing**: Segments are filtered based on similarity while preserving uniqueness
- **Flexible Clustering**: Optional clustering that preserves all segments while organizing by similarity
- **Streamlined Saving**: Unified save functionality that adapts to clustering state
- **Better Segment Organization**: Clear separation between general segments and clustered segments
- **Enhanced Feedback**: Display of segment information including duration and similarity scores

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
- Choose Segmentation Method: Select from available methods
- Adjust Parameters:
  - Set minimum and maximum segment duration
  - Adjust similarity threshold
  - Configure method-specific parameters
- Generate Segments: Click "Segment Audio" to process
- Optional Clustering:
  - Click "Cluster Segments" to organize by similarity
  - View similarity scores and cluster assignments
- Save Segments:
  - Click "Save Segments" to export
  - Segments save to cluster folders if clustered
  - Segments save to single folder if not clustered

### 3. Playback Controls
- Select segments from the list
- Use Play/Stop button to control playback
- View segment duration and time range
- Clear segments using the clear button

### 4. Manual Segmentation
- Click on waveform to add segment boundaries
- Review segments in the list
- Save along with other segments using "Save Segments"

### 5. Clustering
- Optional step after segmentation
- Click "Cluster Segments" to organize similar segments
- View similarity scores to cluster centers
- All segments are preserved during clustering
- Save to organized cluster folders

## Clustered Storage with Metadata

The software organizes segmented audio files into directories by cluster labels. Each segment's filename includes its spectral centroid frequency and musical note, making it easy to analyze and process segments based on their acoustic characteristics.

## Performance Tips

- Use WAV files with standard sample rates (44.1kHz or 48kHz)
- Set appropriate minimum and maximum segment durations
- Adjust similarity threshold based on your needs
- Consider total audio duration when setting time constraints
- Use clustering when organization by similarity is needed
- Save segments before clustering for general organization

## TODO

### 1. Session Management System
- [ ] Implement JSON-based session storage
  - [ ] Save segment positions and metadata
  - [ ] Save clustering information
  - [ ] Store audio file path and parameters
  - [ ] Add auto-save functionality
  - [ ] Create session recovery on program start
  - [ ] Add manual session save/load options

### Future Improvements
- [ ] Add batch processing for onset detection
- [ ] Implement segment preview in list view
- [ ] Add waveform zoom synchronization
- [ ] Improve error handling and user feedback
- [ ] Add export options for session data
- [ ] Create preset system for different audio types
