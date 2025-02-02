import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import librosa
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

def plot_features(audio_file, features):
    y, sr = librosa.load(audio_file)
    plt.figure(figsize=(15, 6))
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    plt.title("Waveform with Features")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    for transient in features["transients"]:
        plt.axvline(transient, color='r', linestyle='--', label='Transient' if transient == features["transients"][0] else "")
    for beat in features["beats"]:
        plt.axvline(beat, color='g', linestyle=':', label='Beat' if beat == features["beats"][0] else "")

    times, spectral_centroid = features["spectral_centroid"]
    plt.plot(times, spectral_centroid / max(spectral_centroid) * max(y), color='b', label="Spectral Centroid")

    plt.legend(loc="upper right")
    plt.show()

def simplified_waveform_with_segments(audio_file, segments):
    y, sr = librosa.load(audio_file)
    times = np.linspace(0, len(y) / sr, num=len(y))
    
    plt.figure(figsize=(12, 2))
    plt.plot(times, y, color="orange", linewidth=0.8)
    plt.title("Simplified Waveform with Segments")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.ylim([-1, 1])

    print(f"Plotting {len(segments)} segments")  # Debug print
    
    # Add segment markers
    for i, (start, end) in enumerate(segments):
        plt.axvline(x=start, color="cyan", linestyle="--", linewidth=0.7)
        plt.text(start, 0.8, f"{i+1}", color="blue", fontsize=8, ha="center")
        # Optionally mark segment ends
        plt.axvline(x=end, color="red", linestyle=":", linewidth=0.5)

    plt.tight_layout()
    plt.show()

class WaveformVisualizer:
    def __init__(self):
        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.fig)
        
        self.ax_wave = self.fig.add_subplot(211)
        self.ax_spec = self.fig.add_subplot(212)
        
        self.current_audio = None
        self.current_segments = []
        self.manual_mode = False
        self.temp_boundaries = []
        
        # Add navigation toolbar
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        self.toolbar = NavigationToolbar2QT(self.canvas, None)
        
        # Store the full time range for zooming
        self.time_range = [0, 1]  # Will be updated when audio is loaded
        
        # Connect mouse wheel event
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Store zoom level and position
        self.zoom_level = 1.0
        self.zoom_position = 0.0  # Center position
        
        # Store colorbar reference
        self.colorbar = None
        
    def on_scroll(self, event):
        """Handle mouse wheel scrolling for zooming"""
        if event.inaxes:
            # Get current x axis limits
            x_min, x_max = event.inaxes.get_xlim()
            x_range = x_max - x_min
            
            # Calculate zoom factor
            if event.button == 'up':
                scale_factor = 0.8  # Zoom in
            else:
                scale_factor = 1.25  # Zoom out
                
            # Calculate new range centered on mouse position
            center = event.xdata
            new_range = x_range * scale_factor
            new_min = max(center - new_range/2, self.time_range[0])
            new_max = min(center + new_range/2, self.time_range[1])
            
            # Update both axes
            self.ax_wave.set_xlim(new_min, new_max)
            self.ax_spec.set_xlim(new_min, new_max)
            self.canvas.draw()

    def enable_manual_mode(self, callback):
        """Enable manual segmentation mode"""
        self.manual_mode = True
        self.temp_boundaries = []
        self.canvas.mpl_connect('button_press_event', callback)
        
    def disable_manual_mode(self):
        """Disable manual segmentation mode"""
        self.manual_mode = False
        self.temp_boundaries = []
        
    def add_boundary(self, time_clicked):
        """Add a boundary line in manual mode"""
        if len(self.temp_boundaries) % 2 == 0:
            # Start boundary (red)
            self.ax_wave.axvline(x=time_clicked, color="red", linestyle="--", linewidth=0.7)
        else:
            # End boundary (blue)
            self.ax_wave.axvline(x=time_clicked, color="blue", linestyle="-", linewidth=0.7)
            # Add segment number
            segment_num = len(self.temp_boundaries) // 2 + 1
            segment_center = (self.temp_boundaries[-1] + time_clicked) / 2
            self.ax_wave.text(segment_center, 0.8, f"{segment_num}", 
                             color="black", fontsize=8, ha="center")
        
        self.temp_boundaries.append(time_clicked)
        self.canvas.draw()
        
        return len(self.temp_boundaries) % 2 == 0  # Return True if segment is complete

    def plot_waveform(self, audio_file, segments=None):
        """Plot or update both waveform and spectrogram"""
        # Remove existing colorbar if it exists
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None
        
        # Clear both axes
        self.ax_wave.clear()
        self.ax_spec.clear()
        
        # Load audio
        y, sr = librosa.load(audio_file)
        times = np.linspace(0, len(y) / sr, num=len(y))
        self.time_range = [0, len(y) / sr]
        
        # Plot waveform
        self.ax_wave.plot(times, y, color="orange", linewidth=0.8)
        self.ax_wave.set_title("Audio Waveform with Segments")
        self.ax_wave.set_xlabel("")
        self.ax_wave.set_ylabel("Amplitude")
        self.ax_wave.set_ylim([-1, 1])
        
        # Create and plot spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_db, 
                                     y_axis='mel', 
                                     x_axis='time',
                                     sr=sr,
                                     fmax=8000,
                                     ax=self.ax_spec)
        
        # Create new colorbar
        self.colorbar = self.fig.colorbar(img, ax=self.ax_spec, format='%+2.0f dB')
        
        self.ax_spec.set_title("Mel Spectrogram")
        self.ax_spec.set_xlabel("Time (s)")
        self.ax_spec.set_ylabel("Frequency (Hz)")
        
        # Plot segments only on waveform if provided
        if segments:
            for i, (start, end) in enumerate(segments):
                # Add lines and labels to waveform only
                self.ax_wave.axvline(x=start, color="red", linestyle="--", linewidth=0.7)
                self.ax_wave.axvline(x=end, color="blue", linestyle="-", linewidth=0.7)
                # Add segment number in the middle of the segment
                segment_center = (start + end) / 2
                self.ax_wave.text(segment_center, 0.8, f"{i+1}", 
                                color="black", fontsize=8, ha="center")
        
        self.fig.tight_layout()
        self.canvas.draw()
        
    def clear(self):
        """Clear both visualizations"""
        self.ax_wave.clear()
        self.ax_spec.clear()
        
        self.ax_wave.set_title("Audio Waveform")
        self.ax_wave.set_xlabel("")
        self.ax_wave.set_ylabel("Amplitude")
        
        self.ax_spec.set_title("Mel Spectrogram")
        self.ax_spec.set_xlabel("Time (s)")
        self.ax_spec.set_ylabel("Frequency (Hz)")
        
        self.canvas.draw()

    def zoom(self, factor, center=None):
        """
        Simplified zoom function that works on both visualizations
        factor: > 1 for zoom out, < 1 for zoom in
        center: position to zoom around (if None, use current center)
        """
        if not hasattr(self, 'time_range'):
            return
            
        # Get current view limits
        x_min, x_max = self.ax_wave.get_xlim()
        current_range = x_max - x_min
        
        # Use provided center or current view center
        if center is None:
            center = (x_min + x_max) / 2
        
        # Calculate new range
        new_range = current_range * factor
        
        # Limit the zoom range
        min_range = (self.time_range[1] - self.time_range[0]) * 0.01  # Minimum 1% of total
        max_range = self.time_range[1] - self.time_range[0]  # Maximum full range
        new_range = np.clip(new_range, min_range, max_range)
        
        # Calculate new limits
        new_min = center - new_range / 2
        new_max = center + new_range / 2
        
        # Ensure we don't go out of bounds
        if new_min < self.time_range[0]:
            new_min = self.time_range[0]
            new_max = new_min + new_range
        if new_max > self.time_range[1]:
            new_max = self.time_range[1]
            new_min = new_max - new_range
            
        # Apply zoom to both visualizations
        self.ax_wave.set_xlim(new_min, new_max)
        self.ax_spec.set_xlim(new_min, new_max)
        self.canvas.draw()