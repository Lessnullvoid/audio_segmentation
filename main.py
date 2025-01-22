"""
This software is a GUI-based audio segmentation tool designed for feature detection, segmentation, and clustering of audio files.

Key Features:
- Intuitive PyQt5 interface for loading and processing audio files.
- Feature Detection: Identifies transients, beats, and spectral features.
- Segmentation: Divides audio into meaningful segments based on detected features.
- Clustering: Groups similar segments to reduce redundancy.
- Visualization: Displays audio waveforms and feature overlays for analysis.
- Clustered Storage with Metadata: Saves segments with frequency and note metadata in organized directories.

The application is ideal for audio engineers, music producers, and researchers who need to analyze and process large audio datasets efficiently.
"""

from PyQt5.QtWidgets import QApplication
import sys
from ui import AudioSegmentationApp  # Import the UI class

def main():
    app = QApplication(sys.argv)
    window = AudioSegmentationApp()  # Create an instance of the UI class
    window.show()  # Show the main window
    sys.exit(app.exec_())  # Start the application event loop

if __name__ == "__main__":
    main()