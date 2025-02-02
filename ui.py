from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QSlider, QLabel, QFileDialog, QWidget, QListWidget, QComboBox, QLineEdit, QHBoxLayout
)
from PyQt5.QtCore import Qt, QTimer
import sys
from feature_detection import detect_features
from segmentation import segment_audio, segment_by_beats, segment_by_transients, segment_by_frequency, segment_by_onsets
from visualization import plot_features, simplified_waveform_with_segments, WaveformVisualizer
from utils import chop_audio_with_metadata
from clustering import cluster_segments, cluster_segments_kmeans
import matplotlib.pyplot as plt
import numpy as np
import librosa
from pydub import AudioSegment
from pydub.playback import play
from audio_player import AudioPlayer


class AudioSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Segmentation Tool")
        self.setGeometry(100, 100, 1000, 800)  # Made window larger to accommodate visualization
        
        # Define button styles
        self.play_button_style = """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px;
                border: none;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """
        
        self.stop_button_style = """
            QPushButton {
                background-color: #ff4444;
                color: white;
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
        """
        
        # Initialize other attributes
        self.auto_segments = []  # For automatic segmentation
        self.manual_segments = []  # For manual segmentation
        self.segments = []  # Current active segments
        self.visualizer = WaveformVisualizer()  # Create visualizer instance
        self.audio_player = AudioPlayer()
        
        self.initUI()

    def initUI(self):
        # Create main widget and layout
        main_widget = QWidget()
        layout = QVBoxLayout()
        
        # Create a horizontal layout for controls and visualization
        h_layout = QHBoxLayout()
        
        # Create a widget for controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout()

        # Style for black buttons (all buttons except play)
        black_button_style = """
            QPushButton {
                background-color: #1a1a1a;
                color: white;
                padding: 8px;
                border: none;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #333333;
            }
            QPushButton:pressed {
                background-color: #0d0d0d;
            }
            QPushButton:checked {
                background-color: #FF4444;
            }
            QPushButton:disabled {
                background-color: #666666;
            }
        """

        # 1. Load and Manual Controls
        button_layout1 = QVBoxLayout()
        self.load_button = QPushButton("Load audio")
        self.manual_button = QPushButton("Manual segmentation")
        self.manual_button.setCheckable(True)
        self.save_manual_button = QPushButton("Save manual segments")
        
        # Apply black style
        self.load_button.setStyleSheet(black_button_style)
        self.manual_button.setStyleSheet(black_button_style)
        self.save_manual_button.setStyleSheet(black_button_style)
        
        button_layout1.addWidget(self.load_button)
        button_layout1.addWidget(self.manual_button)
        button_layout1.addWidget(self.save_manual_button)
        controls_layout.addLayout(button_layout1)

        # 2. Segmentation Method
        self.method_label = QLabel("Select segmentation method")
        controls_layout.addWidget(self.method_label)
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "By Beats", 
            "By Transients", 
            "By Frequency Range",
            "By Onsets"
        ])
        controls_layout.addWidget(self.method_combo)

        # 3. Segmentation Parameters
        # Threshold
        self.threshold_label = QLabel("Segmentation Threshold: 0.1")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(1)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(10)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        controls_layout.addWidget(self.threshold_label)
        controls_layout.addWidget(self.threshold_slider)

        # Clustering Parameters
        self.eps_label = QLabel("Clustering Epsilon: 0.5")
        self.eps_slider = QSlider(Qt.Horizontal)
        self.eps_slider.setMinimum(1)
        self.eps_slider.setMaximum(100)
        self.eps_slider.setValue(50)
        self.eps_slider.valueChanged.connect(self.update_eps)
        controls_layout.addWidget(self.eps_label)
        controls_layout.addWidget(self.eps_slider)

        self.min_samples_label = QLabel("Clustering Min Samples: 1")
        self.min_samples_slider = QSlider(Qt.Horizontal)
        self.min_samples_slider.setMinimum(1)
        self.min_samples_slider.setMaximum(10)
        self.min_samples_slider.setValue(1)
        self.min_samples_slider.valueChanged.connect(self.update_min_samples)
        controls_layout.addWidget(self.min_samples_label)
        controls_layout.addWidget(self.min_samples_slider)

        # Number of Segments
        self.cluster_label = QLabel("Number of Segments (Clusters): 10")
        self.cluster_slider = QSlider(Qt.Horizontal)
        self.cluster_slider.setMinimum(1)
        self.cluster_slider.setMaximum(100)
        self.cluster_slider.setValue(10)
        self.cluster_slider.valueChanged.connect(self.update_clusters)
        controls_layout.addWidget(self.cluster_label)
        controls_layout.addWidget(self.cluster_slider)

        # Frequency Range
        self.min_freq_label = QLabel("Minimum Frequency: 100 Hz")
        self.min_freq_slider = QSlider(Qt.Horizontal)
        self.min_freq_slider.setMinimum(20)
        self.min_freq_slider.setMaximum(20000)
        self.min_freq_slider.setValue(100)
        self.min_freq_slider.valueChanged.connect(self.update_min_freq)
        controls_layout.addWidget(self.min_freq_label)
        controls_layout.addWidget(self.min_freq_slider)

        self.max_freq_label = QLabel("Maximum Frequency: 5000 Hz")
        self.max_freq_slider = QSlider(Qt.Horizontal)
        self.max_freq_slider.setMinimum(20)
        self.max_freq_slider.setMaximum(20000)
        self.max_freq_slider.setValue(5000)
        self.max_freq_slider.valueChanged.connect(self.update_max_freq)
        controls_layout.addWidget(self.max_freq_label)
        controls_layout.addWidget(self.max_freq_slider)

        # Desired Number of Segments
        self.manual_segments_label = QLabel("Desired Number of Segments:")
        controls_layout.addWidget(self.manual_segments_label)
        self.manual_segments_input = QLineEdit()
        self.manual_segments_input.setPlaceholderText("Enter number of segments (e.g., 10)")
        controls_layout.addWidget(self.manual_segments_input)

        # Similarity Threshold
        self.similarity_label = QLabel("Similarity Threshold: 0.85")
        self.similarity_slider = QSlider(Qt.Horizontal)
        self.similarity_slider.setMinimum(50)
        self.similarity_slider.setMaximum(100)
        self.similarity_slider.setValue(85)
        self.similarity_slider.valueChanged.connect(self.update_similarity)
        controls_layout.addWidget(self.similarity_label)
        controls_layout.addWidget(self.similarity_slider)

        # 4. Action Buttons
        self.segment_button = QPushButton("Segment and Visualize")
        self.cluster_button = QPushButton("Cluster Segments")
        self.save_button = QPushButton("Save Segments")
        self.clear_button = QPushButton("Clear Segments")
        
        # Apply black style
        self.segment_button.setStyleSheet(black_button_style)
        self.cluster_button.setStyleSheet(black_button_style)
        self.save_button.setStyleSheet(black_button_style)
        self.clear_button.setStyleSheet(black_button_style)
        
        controls_layout.addWidget(self.segment_button)
        controls_layout.addWidget(self.cluster_button)
        controls_layout.addWidget(self.save_button)
        controls_layout.addWidget(self.clear_button)

        # 5. Segment List
        self.cluster_list = QListWidget()
        controls_layout.addWidget(self.cluster_list)

        # 6. Play Button (at the bottom)
        self.play_button = QPushButton("Play Selected Segment")
        self.play_button.setStyleSheet(self.play_button_style)
        controls_layout.addWidget(self.play_button)

        controls_widget.setLayout(controls_layout)
        
        # Create a container for visualization and its controls
        viz_container = QWidget()
        viz_layout = QVBoxLayout()
        
        # Zoom controls at the top
        viz_controls_layout = QHBoxLayout()
        
        # Zoom buttons
        zoom_in_button = QPushButton("+")
        zoom_out_button = QPushButton("-")
        reset_zoom_button = QPushButton("Reset View")
        
        # Style zoom buttons
        zoom_button_style = """
            QPushButton {
                background-color: #1a1a1a;
                color: white;
                padding: 4px;
                border: none;
                border-radius: 4px;
                min-width: 30px;
            }
            QPushButton:hover {
                background-color: #333333;
            }
        """
        
        zoom_in_button.setStyleSheet(zoom_button_style)
        zoom_out_button.setStyleSheet(zoom_button_style)
        reset_zoom_button.setStyleSheet(black_button_style)
        
        viz_controls_layout.addWidget(zoom_in_button)
        viz_controls_layout.addWidget(zoom_out_button)
        viz_controls_layout.addWidget(reset_zoom_button)
        viz_controls_layout.addStretch()
        
        # Add visualization toolbar
        viz_controls_layout.addWidget(self.visualizer.toolbar)
        
        # Add controls and canvas to viz container
        viz_layout.addLayout(viz_controls_layout)
        viz_layout.addWidget(self.visualizer.canvas)
        
        viz_container.setLayout(viz_layout)
        
        # Add controls and visualization container to main layout
        h_layout.addWidget(controls_widget)
        h_layout.addWidget(viz_container)
        
        # Add the horizontal layout to the main layout
        layout.addLayout(h_layout)
        
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        # Connect buttons to functions
        self.load_button.clicked.connect(self.load_audio)
        self.manual_button.clicked.connect(self.toggle_manual_mode)
        self.play_button.clicked.connect(self.play_segment)
        self.save_manual_button.clicked.connect(self.save_manual_segments)
        self.segment_button.clicked.connect(self.segment_audio)
        self.cluster_button.clicked.connect(self.cluster_segments)
        self.save_button.clicked.connect(self.save_segments)
        self.clear_button.clicked.connect(self.clear_segments)
        
        # Connect zoom buttons
        zoom_in_button.clicked.connect(self.zoom_in)
        zoom_out_button.clicked.connect(self.zoom_out)
        reset_zoom_button.clicked.connect(self.reset_zoom)

    def load_audio(self):
        self.audio_file, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "Audio Files (*.wav)")
        if self.audio_file:
            print(f"Loaded audio file: {self.audio_file}")
            self.segments = []
            self.visualizer.plot_waveform(self.audio_file)  # Initial visualization

    def update_threshold(self):
        value = self.threshold_slider.value() / 100
        self.threshold_label.setText(f"Segmentation Threshold: {value:.2f}")

    def update_eps(self):
        value = self.eps_slider.value() / 100.0
        self.eps_label.setText(f"Clustering Epsilon: {value:.2f}")
        self.eps = value

    def update_min_samples(self):
        value = self.min_samples_slider.value()
        self.min_samples_label.setText(f"Clustering Min Samples: {value}")
        self.min_samples = value

    def update_min_freq(self):
        value = self.min_freq_slider.value()
        self.min_freq_label.setText(f"Minimum Frequency: {value} Hz")
        self.min_freq = value

    def update_max_freq(self):
        value = self.max_freq_slider.value()
        self.max_freq_label.setText(f"Maximum Frequency: {value} Hz")
        self.max_freq = value

    def update_clusters(self):
        value = self.cluster_slider.value()
        self.cluster_label.setText(f"Number of Segments (Clusters): {value}")
        self.n_clusters = value

    def update_similarity(self):
        value = self.similarity_slider.value() / 100
        self.similarity_label.setText(f"Similarity Threshold: {value:.2f}")

    def segment_audio(self):
        if not hasattr(self, "audio_file"):
            print("\n[ERROR] No audio file loaded!")
            return

        print("\n" + "="*50)
        print("STARTING AUDIO SEGMENTATION PROCESS")
        print("="*50)

        print("\n[1/4] Detecting audio features...")
        self.features = detect_features(self.audio_file)
        print(f"✓ Features extracted successfully")

        manual_segment_count = self.manual_segments_input.text()
        selected_method = self.method_combo.currentText()
        print(f"\n[2/4] Using segmentation method: {selected_method}")

        # Special handling for when user requests specific number of segments
        if manual_segment_count.isdigit():
            n_segments = int(manual_segment_count)
            print("\n[3/4] Processing with requested segment count:")
            print(f"└── Target number of segments: {n_segments}")
            print("└── Generating initial segments...")

            # Generate all possible segments based on method
            if selected_method == "By Beats":
                all_segments = segment_by_beats(self.features)
            elif selected_method == "By Transients":
                all_segments = segment_by_transients(self.features)
            elif selected_method == "By Frequency Range":
                min_freq = self.min_freq_slider.value()
                max_freq = self.max_freq_slider.value()
                print(f"└── Frequency range: {min_freq}Hz - {max_freq}Hz")
                all_segments = segment_by_frequency(self.features, min_freq=min_freq, max_freq=max_freq)
            elif selected_method == "By Onsets":
                print("└── Using onset detection...")
                all_segments = segment_by_onsets(self.features)
            else:
                print("\n[ERROR] Unknown segmentation method")
                return

            if all_segments:
                print(f"└── Found {len(all_segments)} potential segments")
                print("\n[4/4] Clustering process:")
                print(f"└── Selecting {n_segments} most representative segments...")
                
                if n_segments > len(all_segments):
                    print(f"└── WARNING: Found fewer segments than requested")
                    print(f"    ├── Requested: {n_segments}")
                    print(f"    └── Available: {len(all_segments)}")
                    n_segments = len(all_segments)
                
                similarity_threshold = self.similarity_slider.value() / 100
                print(f"└── Using similarity threshold: {similarity_threshold:.2f}")
                
                self.auto_segments, self.cluster_labels = cluster_segments_kmeans(
                    self.audio_file,
                    all_segments,
                    n_clusters=n_segments,
                    similarity_threshold=similarity_threshold
                )
                print(f"✓ Successfully selected {len(self.auto_segments)} segments")
            else:
                print("\n[ERROR] No segments could be generated!")
                return

        # Original logic for when no specific segment count is requested
        else:
            print("\n[3/4] Processing with automatic segmentation:")
            if selected_method == "By Beats":
                print("└── Detecting and segmenting by beats...")
                self.auto_segments = segment_by_beats(self.features)
            elif selected_method == "By Transients":
                print("└── Detecting and segmenting by transients...")
                self.auto_segments = segment_by_transients(self.features)
            elif selected_method == "By Frequency Range":
                min_freq = self.min_freq_slider.value()
                max_freq = self.max_freq_slider.value()
                print(f"└── Segmenting by frequency range: {min_freq}Hz - {max_freq}Hz")
                self.auto_segments = segment_by_frequency(self.features, min_freq=min_freq, max_freq=max_freq)
            elif selected_method == "By Onsets":
                print("└── Detecting and segmenting by onsets...")
                self.auto_segments = segment_by_onsets(self.features)
            else:
                print("\n[ERROR] Unknown segmentation method")
                return

        # Update segments and visualization
        print("\n[4/4] Finalizing:")
        self.segments = self.auto_segments
        
        if self.segments:
            print(f"└── Generated {len(self.segments)} final segments")
            self.visualizer.plot_waveform(self.audio_file, self.segments)
            self.cluster_list.clear()
            for i, segment in enumerate(self.segments):
                self.cluster_list.addItem(
                    f"Segment {i + 1}: {segment[0]:.2f}s - {segment[1]:.2f}s"
                )
            print("\n✓ Segmentation process completed successfully!")
            print("="*50)
        else:
            print("\n[ERROR] No segments were generated!")
            print("="*50)

    def toggle_manual_mode(self):
        """Toggle manual segmentation mode"""
        if not hasattr(self, "audio_file"):
            print("No audio file loaded!")
            self.manual_button.setChecked(False)
            return
        
        if self.manual_button.isChecked():
            # Enable manual mode
            self.manual_button.setStyleSheet("background-color: #FF4444; color: white;")
            self.visualizer.enable_manual_mode(self.manual_segment_click)
            print("Manual segmentation mode enabled. Click to add segment boundaries.")
        else:
            # Disable manual mode
            self.manual_button.setStyleSheet("")
            self.visualizer.disable_manual_mode()
            print("Manual segmentation mode disabled.")

    def manual_segment_click(self, event):
        """Handle clicks during manual segmentation"""
        if not self.manual_button.isChecked():
            return
        
        if event.inaxes in [self.visualizer.ax_wave, self.visualizer.ax_spec]:
            time_clicked = event.xdata
            
            if event.key == 'c':  # Clear last boundary
                if self.visualizer.temp_boundaries:
                    self.visualizer.ax_wave.lines.pop()
                    self.visualizer.ax_spec.lines.pop()
                    self.visualizer.temp_boundaries.pop()
                    self.visualizer.canvas.draw()
            else:
                # Add new boundary
                segment_complete = self.visualizer.add_boundary(time_clicked)
                
                if segment_complete:
                    # Create new segment
                    start = self.visualizer.temp_boundaries[-2]
                    end = self.visualizer.temp_boundaries[-1]
                    self.manual_segments.append((start, end))
                    self.segments = self.manual_segments
                    
                    # Update segment list
                    self.cluster_list.addItem(
                        f"Manual Segment {len(self.manual_segments)}: ({start:.2f}s - {end:.2f}s)"
                    )

    def play_segment(self):
        """Play selected segment"""
        if not hasattr(self, "audio_file") or not self.segments:
            print("No segments available to play!")
            return

        selected_items = self.cluster_list.selectedItems()
        if not selected_items:
            print("No segment selected!")
            return

        # Get selected segment
        selected_index = self.cluster_list.row(selected_items[0])
        start, end = self.segments[selected_index]

        # Update button text/style while playing
        self.play_button.setText("Stop Playback")
        self.play_button.setStyleSheet(self.stop_button_style)
        
        # Play the segment
        self.audio_player.play_segment(self.audio_file, start, end)
        
        # Start a timer to check when playback is finished
        QTimer.singleShot(100, self.check_playback_status)

    def check_playback_status(self):
        """Check if playback has finished and reset button"""
        if not self.audio_player.is_playing():
            self.play_button.setText("Play Selected Segment")
            self.play_button.setStyleSheet(self.play_button_style)
        else:
            # Check again in 100ms
            QTimer.singleShot(100, self.check_playback_status)

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

    def clear_segments(self):
        """Clear all segments and reset the visualization"""
        self.auto_segments = []
        self.manual_segments = []
        self.segments = []
        self.cluster_labels = None
        self.cluster_list.clear()
        
        # Disable manual mode if active
        if self.manual_button.isChecked():
            self.manual_button.setChecked(False)
            self.manual_button.setStyleSheet("")
            self.visualizer.disable_manual_mode()
        
        if hasattr(self, "audio_file"):
            self.visualizer.plot_waveform(self.audio_file)
            print("All segments cleared!")
        else:
            print("No audio file loaded!")

    def zoom_in(self):
        """Zoom in on both visualizations"""
        self.visualizer.zoom(0.8)  # Zoom in by 20%

    def zoom_out(self):
        """Zoom out on both visualizations"""
        self.visualizer.zoom(1.25)  # Zoom out by 25%

    def reset_zoom(self):
        """Reset zoom to show full waveform"""
        if hasattr(self.visualizer, 'time_range'):
            self.visualizer.zoom(float('inf'))  # This will force it to maximum range


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioSegmentationApp()
    window.show()
    sys.exit(app.exec_())