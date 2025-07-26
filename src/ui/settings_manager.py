import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class HandType(Enum):
    LEFT = "left"
    RIGHT = "right"

class ControlMode(Enum):
    GESTURE = "gesture"
    TOUCH = "touch"
    HYBRID = "hybrid"

class GraphicsQuality(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class AudioSettings:
    master_volume: float = 1.0
    music_volume: float = 0.7
    sfx_volume: float = 0.8
    voice_volume: float = 0.9
    vibration_enabled: bool = True

@dataclass
class ControlSettings:
    gesture_sensitivity: float = 1.0
    hand_type: HandType = HandType.RIGHT
    control_mode: ControlMode = ControlMode.GESTURE
    custom_gestures: Dict[str, str] = None
    vibration_feedback: bool = True

@dataclass
class DisplaySettings:
    graphics_quality: GraphicsQuality = GraphicsQuality.MEDIUM
    show_hud: bool = True
    hud_elements: List[str] = None
    language: str = "en"
    adaptive_ui: bool = True

class SettingsManager:
    def __init__(self):
        self.audio = AudioSettings()
        self.controls = ControlSettings()
        self.display = DisplaySettings()
        self._load_settings()
    
    def _load_settings(self):
        try:
            with open('settings.json', 'r') as f:
                data = json.load(f)
                
                # Load audio settings
                audio_data = data.get('audio', {})
                self.audio.master_volume = audio_data.get('master_volume', 1.0)
                self.audio.music_volume = audio_data.get('music_volume', 0.7)
                self.audio.sfx_volume = audio_data.get('sfx_volume', 0.8)
                self.audio.voice_volume = audio_data.get('voice_volume', 0.9)
                self.audio.vibration_enabled = audio_data.get('vibration_enabled', True)
                
                # Load control settings
                control_data = data.get('controls', {})
                self.controls.gesture_sensitivity = control_data.get('gesture_sensitivity', 1.0)
                self.controls.hand_type = HandType(control_data.get('hand_type', 'right'))
                self.controls.control_mode = ControlMode(control_data.get('control_mode', 'gesture'))
                self.controls.custom_gestures = control_data.get('custom_gestures', {})
                self.controls.vibration_feedback = control_data.get('vibration_feedback', True)
                
                # Load display settings
                display_data = data.get('display', {})
                self.display.graphics_quality = GraphicsQuality(display_data.get('graphics_quality', 'medium'))
                self.display.show_hud = display_data.get('show_hud', True)
                self.display.hud_elements = display_data.get('hud_elements', [
                    'speedometer', 'gear', 'fuel', 'minimap', 'timer',
                    'mission_goals', 'gesture_status', 'directional_arrows',
                    'indicators', 'power_bars', 'nitro', 'collision_warning',
                    'distance_tracker'
                ])
                self.display.language = display_data.get('language', 'en')
                self.display.adaptive_ui = display_data.get('adaptive_ui', True)
        except FileNotFoundError:
            self._save_settings()
    
    def _save_settings(self):
        data = {
            'audio': {
                'master_volume': self.audio.master_volume,
                'music_volume': self.audio.music_volume,
                'sfx_volume': self.audio.sfx_volume,
                'voice_volume': self.audio.voice_volume,
                'vibration_enabled': self.audio.vibration_enabled
            },
            'controls': {
                'gesture_sensitivity': self.controls.gesture_sensitivity,
                'hand_type': self.controls.hand_type.value,
                'control_mode': self.controls.control_mode.value,
                'custom_gestures': self.controls.custom_gestures,
                'vibration_feedback': self.controls.vibration_feedback
            },
            'display': {
                'graphics_quality': self.display.graphics_quality.value,
                'show_hud': self.display.show_hud,
                'hud_elements': self.display.hud_elements,
                'language': self.display.language,
                'adaptive_ui': self.display.adaptive_ui
            }
        }
        
        with open('settings.json', 'w') as f:
            json.dump(data, f, indent=4)
    
    def update_audio_settings(self, settings: Dict):
        """Update audio settings"""
        for key, value in settings.items():
            if hasattr(self.audio, key):
                setattr(self.audio, key, value)
        self._save_settings()
    
    def update_control_settings(self, settings: Dict):
        """Update control settings"""
        for key, value in settings.items():
            if hasattr(self.controls, key):
                if key == 'hand_type':
                    value = HandType(value)
                elif key == 'control_mode':
                    value = ControlMode(value)
                setattr(self.controls, key, value)
        self._save_settings()
    
    def update_display_settings(self, settings: Dict):
        """Update display settings"""
        for key, value in settings.items():
            if hasattr(self.display, key):
                if key == 'graphics_quality':
                    value = GraphicsQuality(value)
                setattr(self.display, key, value)
        self._save_settings()
    
    def get_volume(self, volume_type: str) -> float:
        """Get effective volume for a specific type"""
        base_volume = getattr(self.audio, f"{volume_type}_volume", 1.0)
        return base_volume * self.audio.master_volume
    
    def is_hud_element_enabled(self, element: str) -> bool:
        """Check if a specific HUD element is enabled"""
        return self.display.show_hud and element in self.display.hud_elements
    
    def get_graphics_settings(self) -> Dict:
        """Get graphics settings based on quality level"""
        settings = {
            GraphicsQuality.LOW: {
                'shadow_quality': 'low',
                'texture_quality': 'low',
                'particle_effects': False,
                'post_processing': False,
                'anti_aliasing': False
            },
            GraphicsQuality.MEDIUM: {
                'shadow_quality': 'medium',
                'texture_quality': 'medium',
                'particle_effects': True,
                'post_processing': True,
                'anti_aliasing': True
            },
            GraphicsQuality.HIGH: {
                'shadow_quality': 'high',
                'texture_quality': 'high',
                'particle_effects': True,
                'post_processing': True,
                'anti_aliasing': True
            }
        }
        return settings[self.display.graphics_quality]
    
    def get_language_file(self) -> str:
        """Get the path to the current language file"""
        return f"languages/{self.display.language}.json"
    
    def remap_gesture(self, gesture: str, action: str):
        """Remap a gesture to a different action"""
        if not self.controls.custom_gestures:
            self.controls.custom_gestures = {}
        self.controls.custom_gestures[gesture] = action
        self._save_settings()
    
    def reset_to_defaults(self):
        """Reset all settings to default values"""
        self.audio = AudioSettings()
        self.controls = ControlSettings()
        self.display = DisplaySettings()
        self._save_settings() 