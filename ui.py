from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QSlider, QLabel, QFileDialog, QWidget, QListWidget
)
import sys
from feature_detection import detect_features
from segmentation import segment_audio
from visualization import plot_features
from utils import chop_audio
from clustering import cluster_segments


class AudioSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Segmentation Tool")
        self.setGeometry(100, 100, 800, 600)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Load Audio Button
        self.load_button = QPushButton("Load Audio File")
        self.load_button.clicked.connect(self.load_audio)
        layout.addWidget(self.load_button)

        # Adjust Threshold Slider
        self.threshold_label = QLabel("Segmentation Threshold: 0.1")
        self.threshold_slider = QSlider()
        self.threshold_slider.setOrientation(1)  # Horizontal
        self.threshold_slider.setMinimum(1)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(10)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        layout.addWidget(self.threshold_label)
        layout.addWidget(self.threshold_slider)

        # Slider for Clustering Epsilon
        self.eps_slider = QSlider()
        self.eps_slider.setOrientation(1)  # Horizontal
        self.eps_slider.setMinimum(1)
        self.eps_slider.setMaximum(100)
        self.eps_slider.setValue(50)
        self.eps_slider.valueChanged.connect(self.update_eps)
        layout.addWidget(QLabel("Clustering Epsilon:"))
        layout.addWidget(self.eps_slider)

        # Slider for Clustering Min Samples
        self.min_samples_slider = QSlider()
        self.min_samples_slider.setOrientation(1)  # Horizontal
        self.min_samples_slider.setMinimum(1)
        self.min_samples_slider.setMaximum(10)
        self.min_samples_slider.setValue(1)
        self.min_samples_slider.valueChanged.connect(self.update_min_samples)
        layout.addWidget(QLabel("Clustering Min Samples:"))
        layout.addWidget(self.min_samples_slider)

        # Visualize and Segment Button
        self.segment_button = QPushButton("Segment and Visualize")
        self.segment_button.clicked.connect(self.segment_audio)
        layout.addWidget(self.segment_button)

        # Cluster Segments Button
        self.cluster_button = QPushButton("Cluster Segments")
        self.cluster_button.clicked.connect(self.cluster_segments)
        layout.addWidget(self.cluster_button)

        # Display Clustered Segments
        self.cluster_list = QListWidget()
        layout.addWidget(self.cluster_list)

        # Save Segments Button
        self.save_button = QPushButton("Save Segments")
        self.save_button.clicked.connect(self.save_segments)
        layout.addWidget(self.save_button)

        # Main Widget
        main_widget = QWidget()
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

    def load_audio(self):
        self.audio_file, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "Audio Files (*.wav)")
        if self.audio_file:
            print(f"Loaded audio file: {self.audio_file}")

    def update_threshold(self):
        value = self.threshold_slider.value() / 100
        self.threshold_label.setText(f"Segmentation Threshold: {value}")

    def update_eps(self):
        self.eps = self.eps_slider.value() / 100.0
        print(f"Updated Epsilon: {self.eps}")

    def update_min_samples(self):
        self.min_samples = self.min_samples_slider.value()
        print(f"Updated Min Samples: {self.min_samples}")

    def segment_audio(self):
        if not hasattr(self, "audio_file"):
            print("No audio file loaded!")
            return
        self.features = detect_features(self.audio_file)
        plot_features(self.audio_file, self.features)
        threshold = self.threshold_slider.value() / 100
        self.segments = segment_audio(self.features, threshold)
        print(f"Detected Segments: {self.segments}")

    def cluster_segments(self):
        if not hasattr(self, "segments"):
            print("No segments to cluster!")
            return
        
        # Use dynamic parameters from sliders
        eps = self.eps_slider.value() / 100.0
        min_samples = self.min_samples_slider.value()
        
        # Automatically reduce similar segments
        self.unique_segments = cluster_segments(self.audio_file, self.segments, eps=eps, min_samples=min_samples)
        print(f"Unique Segments: {self.unique_segments}")
        
        # Update the cluster list
        self.cluster_list.clear()
        for i, segment in enumerate(self.unique_segments):
            self.cluster_list.addItem(f"Unique Segment {i + 1}: {segment}")

    def save_segments(self):
        # Save only the unique segments if they exist
        if hasattr(self, "unique_segments"):
            chop_audio(self.audio_file, self.unique_segments)
            print("Unique segments saved!")
        elif hasattr(self, "segments"):
            chop_audio(self.audio_file, self.segments)
            print("All segments saved!")
        else:
            print("No segments to save!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioSegmentationApp()
    window.show()
    sys.exit(app.exec_())