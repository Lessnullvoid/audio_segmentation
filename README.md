# Audio Segmentation Tool

A Python-based GUI application for segmenting audio files using various methods and clustering similar segments together.

## Features

- **Multiple Segmentation Methods**:
  - Beat-based segmentation
  - Transient-based segmentation
  - Frequency range-based segmentation

- **Interactive Visualization**:
  - Waveform display
  - Zoom controls
  - Manual segmentation through clicking
  - Real-time segment visualization

- **Clustering Capabilities**:
  - K-means clustering of similar segments
  - Adjustable number of clusters
  - Similarity threshold control

- **Playback Features**:
  - Play individual segments
  - Stop/resume playback
  - Real-time playback status indication

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/audio_segmentation.git
cd audio_segmentation
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Launch the application:
```bash
python main.py
```

2. **Basic Workflow**:
   - Click "Load Audio" to select a WAV file
   - Choose a segmentation method from the dropdown
   - Adjust parameters as needed
   - Click "Segment and Visualize" to process the audio
   - Use the cluster controls to group similar segments
   - Play segments by selecting them and clicking "Play"

3. **Manual Segmentation**:
   - Click "Manual Segmentation" to enable
   - Click on the waveform to add segment boundaries
   - Press 'C' to clear the last boundary
   - Save manual segments using "Save Manual Segments"

## Controls

### Segmentation Parameters

- **Segmentation Threshold**: Controls sensitivity of segmentation
- **Clustering Epsilon**: Affects cluster boundary decisions
- **Min Samples**: Minimum samples for cluster formation
- **Number of Segments**: Target number of clusters
- **Frequency Range**: Min/Max frequencies for frequency-based segmentation
- **Similarity Threshold**: Controls segment similarity matching

### Visualization Controls

- **Zoom In (+)**: Increase zoom level
- **Zoom Out (-)**: Decrease zoom level
- **Reset View**: Return to full waveform view

### Playback Controls

- **Play Selected Segment**: Plays the currently selected segment
- **Stop Playback**: Stops current playback (same button)

## File Management

- **Save Segments**: Exports segments with metadata
- **Save Manual Segments**: Exports manually created segments
- **Clear Segments**: Removes all current segments

## Key Components

1. **Feature Detection** (`feature_detection.py`):
   - Extracts audio features (beats, transients, spectral features)
   - Provides basis for segmentation

2. **Segmentation** (`segmentation.py`):
   - Implements different segmentation algorithms
   - Handles segment boundary detection

3. **Clustering** (`clustering.py`):
   - Groups similar segments
   - Reduces redundant segments

4. **Visualization** (`visualization.py`):
   - Handles waveform display
   - Manages interactive visualization elements

5. **Audio Player** (`audio_player.py`):
   - Manages audio playback
   - Handles segment-specific playback

## Tips

- For best results with beat-based segmentation, use rhythmic audio
- Transient-based segmentation works well for percussive sounds
- Adjust the similarity threshold to control cluster tightness
- Use manual segmentation for precise control
- Combine automatic segmentation with manual adjustments for optimal results

## Requirements

- Python 3.7+
- PyQt5
- librosa
- numpy
- matplotlib
- pygame
- soundfile
- scikit-learn

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]
