from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QSlider, QLabel, QFileDialog, QWidget, QListWidget, QComboBox, QLineEdit
)
from PyQt5.QtCore import Qt
import sys
from feature_detection import detect_features
from segmentation import segment_audio, segment_by_beats, segment_by_transients, segment_by_frequency
from visualization import plot_features, simplified_waveform_with_segments
from utils import chop_audio_with_metadata
from clustering import cluster_segments, cluster_segments_kmeans
import matplotlib.pyplot as plt
import numpy as np
import librosa
from pydub import AudioSegment
from pydub.playback import play


class AudioSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Segmentation Tool")
        self.setGeometry(100, 100, 800, 600)
        self.segments = []  # Initialize the segments attribute
        self.manual_segments = []  # Store manually selected segments
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Title
        title_label = QLabel("Load and manual")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title_label)

        # Button Row 1
        button_layout1 = QVBoxLayout()
        self.load_button = QPushButton("Load audio")
        self.manual_button = QPushButton("Manual segmentation")
        button_layout1.addWidget(self.load_button)
        button_layout1.addWidget(self.manual_button)
        layout.addLayout(button_layout1)

        # Button Row 2
        button_layout2 = QVBoxLayout()
        self.play_button = QPushButton("Play segment")
        self.save_manual_button = QPushButton("Save manual segments")
        button_layout2.addWidget(self.play_button)
        button_layout2.addWidget(self.save_manual_button)
        layout.addLayout(button_layout2)

        # Subtitle
        subtitle_label = QLabel("Algorithmic Segmentation")
        subtitle_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(subtitle_label)

        # Segmentation Method
        self.method_label = QLabel("Select segmentation method")
        layout.addWidget(self.method_label)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["By beats", "By transients", "By frequency range"])
        layout.addWidget(self.method_combo)

        # Sliders and Inputs
        self.threshold_label = QLabel("Segmentation Threshold: 0.1")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(1)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(10)
        layout.addWidget(self.threshold_label)
        layout.addWidget(self.threshold_slider)

        self.eps_slider = QSlider(Qt.Horizontal)
        self.eps_slider.setMinimum(1)
        self.eps_slider.setMaximum(100)
        self.eps_slider.setValue(50)
        layout.addWidget(QLabel("Clustering Epsilon:"))
        layout.addWidget(self.eps_slider)

        self.min_samples_slider = QSlider(Qt.Horizontal)
        self.min_samples_slider.setMinimum(1)
        self.min_samples_slider.setMaximum(10)
        self.min_samples_slider.setValue(1)
        layout.addWidget(QLabel("Clustering Min Samples:"))
        layout.addWidget(self.min_samples_slider)

        self.cluster_label = QLabel("Number of Segments (Clusters):")
        layout.addWidget(self.cluster_label)
        self.cluster_slider = QSlider(Qt.Horizontal)
        self.cluster_slider.setMinimum(1)
        self.cluster_slider.setMaximum(100)
        self.cluster_slider.setValue(10)
        layout.addWidget(self.cluster_slider)

        self.min_freq_slider = QSlider(Qt.Horizontal)
        self.min_freq_slider.setMinimum(20)
        self.min_freq_slider.setMaximum(20000)
        self.min_freq_slider.setValue(100)
        layout.addWidget(QLabel("Minimum Frequency:"))
        layout.addWidget(self.min_freq_slider)

        self.max_freq_slider = QSlider(Qt.Horizontal)
        self.max_freq_slider.setMinimum(20)
        self.max_freq_slider.setMaximum(20000)
        self.max_freq_slider.setValue(5000)
        layout.addWidget(QLabel("Maximum Frequency:"))
        layout.addWidget(self.max_freq_slider)

        self.manual_segments_label = QLabel("Desired Number of Segments:")
        layout.addWidget(self.manual_segments_label)
        self.manual_segments_input = QLineEdit()
        self.manual_segments_input.setPlaceholderText("Enter number of segments (e.g., 10)")
        layout.addWidget(self.manual_segments_input)

        # Action Buttons
        self.segment_button = QPushButton("Segment and Visualize")
        self.cluster_button = QPushButton("Cluster Segments")
        self.save_button = QPushButton("Save Segments")
        layout.addWidget(self.segment_button)
        layout.addWidget(self.cluster_button)
        layout.addWidget(self.save_button)

        # Segment List
        self.cluster_list = QListWidget()
        layout.addWidget(self.cluster_list)

        # Main Widget
        main_widget = QWidget()
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        # Connect buttons to functions
        self.load_button.clicked.connect(self.load_audio)
        self.manual_button.clicked.connect(self.manual_segmentation)
        self.play_button.clicked.connect(self.play_segment)
        self.save_manual_button.clicked.connect(self.save_manual_segments)
        self.segment_button.clicked.connect(self.segment_audio)
        self.cluster_button.clicked.connect(self.cluster_segments)
        self.save_button.clicked.connect(self.save_segments)

    def load_audio(self):
        self.audio_file, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "Audio Files (*.wav)")
        if self.audio_file:
            print(f"Loaded audio file: {self.audio_file}")
            self.segments = []  # Reset segments when a new audio file is loaded
            # Remove the visualization call
            # simplified_waveform_with_segments(self.audio_file, self.segments)

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

    def manual_segmentation(self):
        if not hasattr(self, "audio_file"):
            print("No audio file loaded!")
            return

        y, sr = librosa.load(self.audio_file)
        # Compute the spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots()
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax, fmax=8000)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title('Manual Segmentation: Click to select segment boundaries')

        self.manual_segments = []

        def onclick(event):
            if event.inaxes == ax:
                time_clicked = event.xdata
                if event.key == 'c':
                    # Clear the nearest boundary
                    if self.manual_segments:
                        nearest_index = min(range(len(self.manual_segments)), key=lambda i: abs(self.manual_segments[i] - time_clicked))
                        ax.lines.pop(nearest_index)
                        self.manual_segments.pop(nearest_index)
                        fig.canvas.draw()
                else:
                    if len(self.manual_segments) % 2 == 0:
                        ax.axvline(x=time_clicked, color='r', linestyle='--')  # Red for start
                    else:
                        ax.axvline(x=time_clicked, color='b', linestyle='--')  # Blue for end
                        # Add the segment to the list
                        start = self.manual_segments[-1]
                        end = time_clicked
                        self.segments.append((start, end))
                        self.cluster_list.addItem(f"Manual Segment {len(self.segments)}: ({start:.2f}, {end:.2f})")
                    self.manual_segments.append(time_clicked)
                    fig.canvas.draw()

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

    def play_segment(self):
        if not hasattr(self, "audio_file") or not self.segments:
            print("No segments available to play!")
            return

        selected_items = self.cluster_list.selectedItems()
        if not selected_items:
            print("No segment selected!")
            return

        selected_index = self.cluster_list.row(selected_items[0])
        start, end = self.segments[selected_index]

        self.play_segment_range(start, end)

    def play_segment_range(self, start, end):
        # Load the audio segment
        audio = AudioSegment.from_wav(self.audio_file)
        segment = audio[start * 1000:end * 1000]  # Convert to milliseconds

        # Play the segment
        play(segment)

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

    def save_manual_segments(self):
        # Save only manually selected segments
        if hasattr(self, "segments") and self.segments:
            chop_audio_with_metadata(self.audio_file, self.segments)
            print("Manual segments saved!")
        else:
            print("No manual segments to save!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioSegmentationApp()
    window.show()
    sys.exit(app.exec_())