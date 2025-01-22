from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QSlider, QLabel, QFileDialog, QWidget, QListWidget, QComboBox, QLineEdit
)
import sys
from feature_detection import detect_features
from segmentation import segment_audio, segment_by_beats, segment_by_transients, segment_by_frequency
from visualization import plot_features, simplified_waveform_with_segments
from utils import chop_audio_with_metadata
from clustering import cluster_segments, cluster_segments_kmeans


class AudioSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Segmentation Tool")
        self.setGeometry(100, 100, 800, 600)
        self.segments = []  # Initialize the segments attribute
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

        # Slider for Number of Clusters
        self.cluster_label = QLabel("Number of Segments (Clusters):")
        layout.addWidget(self.cluster_label)

        self.cluster_slider = QSlider()
        self.cluster_slider.setOrientation(1)  # Horizontal
        self.cluster_slider.setMinimum(1)
        self.cluster_slider.setMaximum(100)
        self.cluster_slider.setValue(10)
        self.cluster_slider.valueChanged.connect(self.update_clusters)
        layout.addWidget(self.cluster_slider)

        # Segmentation Method Selection
        self.method_label = QLabel("Select Segmentation Method:")
        layout.addWidget(self.method_label)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["By Beats", "By Transients", "By Frequency Range"])
        layout.addWidget(self.method_combo)

        # Frequency Range Sliders
        self.min_freq_slider = QSlider()
        self.min_freq_slider.setOrientation(1)  # Horizontal
        self.min_freq_slider.setMinimum(20)
        self.min_freq_slider.setMaximum(20000)
        self.min_freq_slider.setValue(100)
        self.min_freq_slider.valueChanged.connect(self.update_min_freq)
        layout.addWidget(QLabel("Minimum Frequency:"))
        layout.addWidget(self.min_freq_slider)

        self.max_freq_slider = QSlider()
        self.max_freq_slider.setOrientation(1)  # Horizontal
        self.max_freq_slider.setMinimum(20)
        self.max_freq_slider.setMaximum(20000)
        self.max_freq_slider.setValue(5000)
        self.max_freq_slider.valueChanged.connect(self.update_max_freq)
        layout.addWidget(QLabel("Maximum Frequency:"))
        layout.addWidget(self.max_freq_slider)

        # Input for Desired Number of Segments
        self.manual_segments_label = QLabel("Desired Number of Segments:")
        layout.addWidget(self.manual_segments_label)

        self.manual_segments_input = QLineEdit()
        self.manual_segments_input.setPlaceholderText("Enter number of segments (e.g., 10)")
        layout.addWidget(self.manual_segments_input)

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
            self.segments = []  # Reset segments when a new audio file is loaded

    def update_threshold(self):
        value = self.threshold_slider.value() / 100
        self.threshold_label.setText(f"Segmentation Threshold: {value}")

    def update_eps(self):
        self.eps = self.eps_slider.value() / 100.0
        print(f"Updated Epsilon: {self.eps}")

    def update_min_samples(self):
        self.min_samples = self.min_samples_slider.value()
        print(f"Updated Min Samples: {self.min_samples}")

    def update_min_freq(self):
        self.min_freq = self.min_freq_slider.value()
        print(f"Updated Minimum Frequency: {self.min_freq}")

    def update_max_freq(self):
        self.max_freq = self.max_freq_slider.value()
        print(f"Updated Maximum Frequency: {self.max_freq}")

    def update_clusters(self):
        self.n_clusters = self.cluster_slider.value()
        self.cluster_label.setText(f"Number of Segments (Clusters): {self.n_clusters}")
        print(f"Updated Number of Clusters: {self.n_clusters}")

    def segment_audio(self):
        if not hasattr(self, "audio_file"):
            print("No audio file loaded!")
            return

        self.features = detect_features(self.audio_file)
        print(f"Extracted features: {self.features}")

        manual_segment_count = self.manual_segments_input.text()
        if manual_segment_count.isdigit():
            n_segments = int(manual_segment_count)
            self.segments, self.cluster_labels = cluster_segments_kmeans(self.audio_file, self.segments, n_clusters=n_segments)
        else:
            selected_method = self.method_combo.currentText()
            if selected_method == "By Beats":
                self.segments = segment_by_beats(self.features)
            elif selected_method == "By Transients":
                self.segments = segment_by_transients(self.features)
            elif selected_method == "By Frequency Range":
                min_freq = self.min_freq_slider.value()
                max_freq = self.max_freq_slider.value()
                self.segments = segment_by_frequency(self.features, min_freq=min_freq, max_freq=max_freq)
            else:
                self.segments = []
            self.cluster_labels = None

        print(f"Detected Segments: {self.segments}")

        # Update the cluster list with detected segments
        self.cluster_list.clear()
        for i, segment in enumerate(self.segments):
            self.cluster_list.addItem(f"Segment {i + 1}: {segment}")

        # Call the visualization function
        simplified_waveform_with_segments(self.audio_file, self.segments)

    def cluster_segments(self):
        if not hasattr(self, "segments"):
            print("No segments to cluster!")
            return
        
        # Use the user-specified number of clusters
        n_clusters = getattr(self, "n_clusters", 10)  # Default to 10 clusters
        
        self.representative_segments = cluster_segments_kmeans(self.audio_file, self.segments, n_clusters=n_clusters)
        print(f"Clustered into {n_clusters} segments: {self.representative_segments}")
        
        self.cluster_list.clear()
        for i, segment in enumerate(self.representative_segments):
            self.cluster_list.addItem(f"Cluster {i + 1}: {segment}")

    def save_segments(self):
        # Save segments with metadata and folder structure
        if hasattr(self, "segments") and self.segments:
            chop_audio_with_metadata(self.audio_file, self.segments, clusters=self.cluster_labels)
            print("Segments saved with metadata!")
        else:
            print("No segments to save!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioSegmentationApp()
    window.show()
    sys.exit(app.exec_())