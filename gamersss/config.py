import json
import os
from typing import Dict, Any

class GameConfig:
    def __init__(self):
        self.config_file = 'config.json'
        self.default_config = {
            'display': {
                'width': 800,
                'height': 600,
                'fullscreen': False,
                'vsync': True,
                'fps_limit': 60
            },
            'audio': {
                'master_volume': 0.7,
                'music_volume': 0.5,
                'sfx_volume': 0.8,
                'mute': False
            },
            'controls': {
                'gesture_sensitivity': 0.8,
                'steering_sensitivity': 1.0,
                'acceleration_sensitivity': 1.0,
                'brake_sensitivity': 1.0,
                'auto_calibration': True
            },
            'gameplay': {
                'difficulty': 'normal',
                'tutorial_enabled': True,
                'auto_save': True,
                'save_interval': 300,  # seconds
                'max_vehicles': 5,
                'starting_coins': 1000
            },
            'graphics': {
                'quality': 'medium',
                'shadows': True,
                'particles': True,
                'anti_aliasing': True,
                'bloom': True
            },
            'debug': {
                'show_fps': True,
                'show_debug_info': False,
                'log_level': 'INFO',
                'save_logs': True
            }
        }
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
                return self.default_config
        return self.default_config
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        try:
            return self.config[section][key]
        except KeyError:
            return default
    
    def set(self, section: str, key: str, value: Any):
        """Set a configuration value"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save_config()
    
    def reset_to_default(self):
        """Reset configuration to default values"""
        self.config = self.default_config
        self.save_config()

# Create global config instance
config = GameConfig() 