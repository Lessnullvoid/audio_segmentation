import pygame
import numpy as np
import librosa
from threading import Thread
import io
import soundfile as sf

class AudioPlayer:
    def __init__(self):
        pygame.mixer.init(frequency=44100)
        self.currently_playing = None
        
    def play_segment(self, audio_file, start_time, end_time):
        """
        Play audio segment efficiently using pygame
        """
        if self.currently_playing:
            self.stop()
            
        try:
            # Load just the segment we need
            y, sr = librosa.load(audio_file, offset=start_time, duration=(end_time - start_time))
            
            # Convert to 16-bit PCM WAV
            buffer = io.BytesIO()
            sf.write(buffer, y, sr, format='WAV')
            buffer.seek(0)
            
            # Play using pygame
            pygame.mixer.music.load(buffer)
            pygame.mixer.music.play()
            self.currently_playing = (start_time, end_time)
            
        except Exception as e:
            print(f"Error playing segment: {e}")
    
    def stop(self):
        """Stop current playback"""
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        self.currently_playing = None
    
    def is_playing(self):
        """Check if audio is currently playing"""
        return pygame.mixer.music.get_busy() 