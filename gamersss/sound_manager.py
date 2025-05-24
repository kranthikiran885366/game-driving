import pygame
import os
from typing import Dict, Optional

class SoundManager:
    def __init__(self):
        pygame.mixer.init()
        self.sounds: Dict[str, pygame.mixer.Sound] = {}
        self.music: Optional[str] = None
        self.volume = 0.7
        self.music_volume = 0.5
        
        # Load all sound effects
        self._load_sounds()
    
    def _load_sounds(self):
        """Load all sound effects from the sounds directory"""
        sound_files = {
            'engine': 'engine.wav',
            'crash': 'crash.wav',
            'horn': 'horn.wav',
            'gear_shift': 'gear_shift.wav',
            'indicator': 'indicator.wav',
            'nitro': 'nitro.wav',
            'achievement': 'achievement.wav',
            'button_click': 'button_click.wav',
            'menu_select': 'menu_select.wav',
            'game_start': 'game_start.wav',
            'game_over': 'game_over.wav',
            'countdown': 'countdown.wav',
            'checkpoint': 'checkpoint.wav',
            'coin_collect': 'coin_collect.wav',
            'power_up': 'power_up.wav'
        }
        
        # Create sounds directory if it doesn't exist
        if not os.path.exists('sounds'):
            os.makedirs('sounds')
        
        # Load each sound
        for sound_name, filename in sound_files.items():
            try:
                sound_path = os.path.join('sounds', filename)
                if os.path.exists(sound_path):
                    self.sounds[sound_name] = pygame.mixer.Sound(sound_path)
            except Exception as e:
                print(f"Error loading sound {filename}: {e}")
    
    def play_sound(self, sound_name: str, loops: int = 0):
        """Play a sound effect"""
        if sound_name in self.sounds:
            self.sounds[sound_name].set_volume(self.volume)
            self.sounds[sound_name].play(loops)
    
    def stop_sound(self, sound_name: str):
        """Stop a specific sound effect"""
        if sound_name in self.sounds:
            self.sounds[sound_name].stop()
    
    def play_music(self, music_file: str, loops: int = -1):
        """Play background music"""
        try:
            music_path = os.path.join('sounds', music_file)
            if os.path.exists(music_path):
                pygame.mixer.music.load(music_path)
                pygame.mixer.music.set_volume(self.music_volume)
                pygame.mixer.music.play(loops)
                self.music = music_file
        except Exception as e:
            print(f"Error playing music {music_file}: {e}")
    
    def stop_music(self):
        """Stop background music"""
        pygame.mixer.music.stop()
        self.music = None
    
    def set_volume(self, volume: float):
        """Set the volume for sound effects"""
        self.volume = max(0.0, min(1.0, volume))
        for sound in self.sounds.values():
            sound.set_volume(self.volume)
    
    def set_music_volume(self, volume: float):
        """Set the volume for background music"""
        self.music_volume = max(0.0, min(1.0, volume))
        pygame.mixer.music.set_volume(self.music_volume)
    
    def pause_music(self):
        """Pause background music"""
        pygame.mixer.music.pause()
    
    def unpause_music(self):
        """Unpause background music"""
        pygame.mixer.music.unpause()
    
    def fade_out_music(self, time_ms: int = 1000):
        """Fade out background music"""
        pygame.mixer.music.fadeout(time_ms)
    
    def is_music_playing(self) -> bool:
        """Check if music is currently playing"""
        return pygame.mixer.music.get_busy()
    
    def cleanup(self):
        """Clean up sound resources"""
        pygame.mixer.quit() 