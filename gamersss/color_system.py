from typing import Dict, Tuple
from dataclasses import dataclass
from enum import Enum

class Theme(Enum):
    DEFAULT = "default"
    NEON_TECH = "neon_tech"
    ASPHALT_NIGHT = "asphalt_night"
    METRO_GLOW = "metro_glow"

@dataclass
class ColorPalette:
    # Primary Colors
    ELECTRIC_BLUE: str = "#007BFF"
    CHARCOAL_BLACK: str = "#1B1B1B"
    WHITE_SMOKE: str = "#F5F5F5"
    
    # Secondary Colors
    AMBER_YELLOW: str = "#FFC107"
    SLATE_PURPLE: str = "#5E60CE"
    DARK_SLATE: str = "#2C2C2E"
    
    # Accent Colors
    NEON_GREEN: str = "#00FFAB"
    CRIMSON_RED: str = "#FF3B30"
    HOT_PINK: str = "#FF2E9A"
    SKY_BLUE: str = "#40C4FF"
    
    # Status Colors
    EMERALD_GREEN: str = "#4CAF50"
    RED_ALERT: str = "#D32F2F"
    COOL_GRAY: str = "#B0BEC5"
    INDIGO_BLUE: str = "#3F51B5"
    
    # In-Game Colors
    ROAD_LINES: str = "#FFEB3B"
    BRAKE_LIGHTS: str = "#FF0000"
    TURN_INDICATORS: str = "#FFA500"
    SPEEDOMETER_DIAL: str = "#2196F3"
    MAP_ROUTE: str = "#00B0FF"
    GESTURE_HUD: str = "#00E676"
    
    # Background Colors (Light Mode)
    LIGHT_BG: str = "#FFFFFF"
    LIGHT_PANEL: str = "#F0F0F0"
    LIGHT_HUD_BG: str = "rgba(0,0,0,0.05)"
    
    # Background Colors (Night Mode)
    DARK_BG: str = "#121212"
    DARK_PANEL: str = "#1F1F1F"
    DARK_HUD_BG: str = "rgba(255,255,255,0.05)"

class ThemeManager:
    def __init__(self):
        self.current_theme = Theme.DEFAULT
        self.is_night_mode = False
        self.palette = ColorPalette()
        self.theme_colors = {
            Theme.NEON_TECH: {
                'primary': "#00FFFF",
                'secondary': "#FF00FF"
            },
            Theme.ASPHALT_NIGHT: {
                'primary': "#2E2E2E",
                'secondary': "#FF6F00"
            },
            Theme.METRO_GLOW: {
                'primary': "#00BCD4",
                'secondary': "#FFC400"
            }
        }
    
    def set_theme(self, theme: Theme):
        """Set the current theme"""
        self.current_theme = theme
    
    def toggle_night_mode(self):
        """Toggle between light and dark mode"""
        self.is_night_mode = not self.is_night_mode
    
    def get_background_color(self) -> str:
        """Get the current background color based on theme and mode"""
        if self.is_night_mode:
            return self.palette.DARK_BG
        return self.palette.LIGHT_BG
    
    def get_panel_color(self) -> str:
        """Get the current panel color based on theme and mode"""
        if self.is_night_mode:
            return self.palette.DARK_PANEL
        return self.palette.LIGHT_PANEL
    
    def get_hud_background(self) -> str:
        """Get the current HUD background color based on theme and mode"""
        if self.is_night_mode:
            return self.palette.DARK_HUD_BG
        return self.palette.LIGHT_HUD_BG
    
    def get_theme_colors(self) -> Dict[str, str]:
        """Get the current theme's primary and secondary colors"""
        return self.theme_colors.get(self.current_theme, {
            'primary': self.palette.ELECTRIC_BLUE,
            'secondary': self.palette.AMBER_YELLOW
        })
    
    def get_status_color(self, status: str) -> str:
        """Get the appropriate color for a given status"""
        status_colors = {
            'success': self.palette.EMERALD_GREEN,
            'error': self.palette.RED_ALERT,
            'warning': self.palette.AMBER_YELLOW,
            'info': self.palette.SKY_BLUE,
            'neutral': self.palette.COOL_GRAY
        }
        return status_colors.get(status.lower(), self.palette.COOL_GRAY)
    
    def get_gesture_color(self, gesture_type: str) -> str:
        """Get the appropriate color for a gesture feedback"""
        gesture_colors = {
            'steering': self.palette.SKY_BLUE,
            'acceleration': self.palette.NEON_GREEN,
            'brake': self.palette.CRIMSON_RED,
            'gear': self.palette.AMBER_YELLOW,
            'indicator': self.palette.TURN_INDICATORS,
            'lights': self.palette.INDIGO_BLUE,
            'horn': self.palette.HOT_PINK,
            'parking': self.palette.EMERALD_GREEN
        }
        return gesture_colors.get(gesture_type.lower(), self.palette.SKY_BLUE)
    
    def get_vehicle_color(self, vehicle_type: str) -> str:
        """Get the appropriate color for a vehicle type"""
        vehicle_colors = {
            'sports': self.palette.HOT_PINK,
            'suv': self.palette.ELECTRIC_BLUE,
            'sedan': self.palette.AMBER_YELLOW,
            'truck': self.palette.SLATE_PURPLE
        }
        return vehicle_colors.get(vehicle_type.lower(), self.palette.ELECTRIC_BLUE)
    
    def get_ui_color(self, element_type: str) -> str:
        """Get the appropriate color for a UI element"""
        ui_colors = {
            'button': self.palette.ELECTRIC_BLUE,
            'text': self.palette.CHARCOAL_BLACK if not self.is_night_mode else self.palette.WHITE_SMOKE,
            'border': self.palette.COOL_GRAY,
            'highlight': self.palette.NEON_GREEN,
            'progress': self.palette.EMERALD_GREEN,
            'error': self.palette.RED_ALERT
        }
        return ui_colors.get(element_type.lower(), self.palette.ELECTRIC_BLUE) 