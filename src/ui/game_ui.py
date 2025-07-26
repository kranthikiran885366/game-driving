import pygame
import cv2
import json
import os
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from src.core.vehicle_customization import VehicleManager, VehicleType, PaintType, TireType
from src.core.game_modes import GameMode, GameModeManager
from src.ai.gesture_controller import GestureController

class Screen(Enum):
    HOME = "home"
    DRIVE_MODE = "drive_mode"
    GARAGE = "garage"
    STORE = "store"
    PROFILE = "profile"
    SETTINGS = "settings"
    CALIBRATION = "calibration"
    TUTORIAL = "tutorial"
    REPLAY = "replay"

@dataclass
class UserProfile:
    username: str
    level: int
    xp: int
    coins: int
    achievements: List[str]
    stats: Dict[str, int]
    unlocked_vehicles: List[VehicleType]
    unlocked_skins: List[str]
    saved_replays: List[str]

class Button:
    def __init__(self, x: int, y: int, width: int, height: int, text: str, color: Tuple[int, int, int]):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = tuple(min(c + 30, 255) for c in color)
        self.is_hovered = False
        self.scale = 1.0
        self.target_scale = 1.0
        self.anim_speed = 0.1
    
    def draw(self, screen: pygame.Surface):
        # Animate scale
        if self.is_hovered:
            self.target_scale = 1.1
        else:
            self.target_scale = 1.0
        self.scale += (self.target_scale - self.scale) * self.anim_speed
        scaled_rect = self.rect.inflate(self.rect.width * (self.scale - 1), self.rect.height * (self.scale - 1))
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, scaled_rect)
        pygame.draw.rect(screen, (255, 255, 255), scaled_rect, 2)
        font = pygame.font.Font(None, 36)
        text_surface = font.render(self.text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=scaled_rect.center)
        screen.blit(text_surface, text_rect)
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_hovered:
                self.target_scale = 1.2
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.target_scale = 1.0
        return False

class GameUI:
    def __init__(self, screen_width: int, screen_height: int):
        self.width = screen_width
        self.height = screen_height
        self.current_screen = Screen.HOME
        self.buttons: Dict[Screen, List[Button]] = {}
        self.vehicle_manager = VehicleManager()
        self.game_mode_manager = GameModeManager()
        self.gesture_controller = GestureController()
        self.profile = self._load_profile()
        self.settings = self._load_settings()
        self.replays = self._load_replays()
        self.menu_car_angle = 0  # For 3D car animation
        self.selected_vehicle_type = VehicleType.SEDAN
        
        self._setup_buttons()
        
    def _load_profile(self) -> UserProfile:
        try:
            with open('profile.json', 'r') as f:
                data = json.load(f)
                return UserProfile(**data)
        except FileNotFoundError:
            return UserProfile(
                username="Player",
                level=1,
                xp=0,
                coins=1000,
                achievements=[],
                stats={},
                unlocked_vehicles=[VehicleType.SEDAN],
                unlocked_skins=[],
                saved_replays=[]
            )
    
    def _load_settings(self) -> Dict:
        try:
            with open('settings.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'controls': {
                    'steering_sensitivity': 1.0,
                    'acceleration_sensitivity': 1.0,
                    'brake_sensitivity': 1.0
                },
                'gestures': {
                    'steering_threshold': 0.5,
                    'acceleration_threshold': 0.5,
                    'brake_threshold': 0.5
                },
                'audio': {
                    'master_volume': 1.0,
                    'music_volume': 0.7,
                    'sfx_volume': 0.8
                },
                'graphics': {
                    'quality': 'high',
                    'resolution': '1920x1080'
                }
            }
    
    def _load_replays(self) -> List[Dict]:
        try:
            with open('replays.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def _setup_buttons(self):
        # Home screen buttons
        self.buttons[Screen.HOME] = [
            Button(300, 200, 200, 50, "Play", (0, 100, 0)),
            Button(300, 300, 200, 50, "Garage", (0, 0, 100)),
            Button(300, 400, 200, 50, "Store", (100, 0, 0)),
            Button(300, 500, 200, 50, "Settings", (100, 100, 0)),
            Button(300, 600, 200, 50, "Profile", (100, 0, 100))
        ]
        
        # Drive mode buttons
        self.buttons[Screen.DRIVE_MODE] = [
            Button(100, 200, 200, 50, "Free Drive", (0, 100, 0)),
            Button(100, 300, 200, 50, "Parking Challenge", (0, 0, 100)),
            Button(100, 400, 200, 50, "Highway Drive", (100, 0, 0)),
            Button(100, 500, 200, 50, "Fuel Saver", (100, 100, 0)),
            Button(100, 600, 200, 50, "Time Trial", (100, 0, 100))
        ]
        
        # Garage buttons
        self.buttons[Screen.GARAGE] = [
            Button(100, 200, 200, 50, "Sports Car", (0, 100, 0)),
            Button(100, 300, 200, 50, "SUV", (0, 0, 100)),
            Button(100, 400, 200, 50, "Sedan", (100, 0, 0)),
            Button(100, 500, 200, 50, "Truck", (100, 100, 0)),
            Button(100, 600, 200, 50, "Customize", (100, 0, 100))
        ]
        
        # Store buttons
        self.buttons[Screen.STORE] = [
            Button(100, 200, 200, 50, "Coins", (0, 100, 0)),
            Button(100, 300, 200, 50, "Vehicles", (0, 0, 100)),
            Button(100, 400, 200, 50, "Skins", (100, 0, 0)),
            Button(100, 500, 200, 50, "Upgrades", (100, 100, 0))
        ]
        
        # Settings buttons
        self.buttons[Screen.SETTINGS] = [
            Button(100, 200, 200, 50, "Controls", (0, 100, 0)),
            Button(100, 300, 200, 50, "Gestures", (0, 0, 100)),
            Button(100, 400, 200, 50, "Audio", (100, 0, 0)),
            Button(100, 500, 200, 50, "Graphics", (100, 100, 0)),
            Button(100, 600, 200, 50, "Calibration", (100, 0, 100))
        ]
    
    def draw_home_screen(self, screen: pygame.Surface):
        # Draw background
        screen.fill((0, 0, 0))
        
        # Draw title
        font = pygame.font.Font(None, 74)
        title = font.render("Hand-Controlled Driving", True, (255, 255, 255))
        screen.blit(title, (self.width//2 - title.get_width()//2, 50))
        
        # Draw 3D car model in the center (OpenGL context assumed to be set up)
        try:
            from driving_hand import draw_3d_car_model
            self.menu_car_angle = (self.menu_car_angle + 1) % 360
            draw_3d_car_model(rotation_angle=self.menu_car_angle, vehicle_type=self.selected_vehicle_type)
        except ImportError:
            pass
        
        # Draw buttons
        for button in self.buttons[Screen.HOME]:
            button.draw(screen)
    
    def draw_drive_mode_screen(self, screen: pygame.Surface):
        screen.fill((0, 0, 0))
        
        # Draw title
        font = pygame.font.Font(None, 74)
        title = font.render("Select Drive Mode", True, (255, 255, 255))
        screen.blit(title, (self.width//2 - title.get_width()//2, 50))
        
        # Draw mode previews
        for i, button in enumerate(self.buttons[Screen.DRIVE_MODE]):
            button.draw(screen)
            # Draw mode description
            desc_font = pygame.font.Font(None, 24)
            desc = self._get_mode_description(button.text)
            desc_surface = desc_font.render(desc, True, (200, 200, 200))
            screen.blit(desc_surface, (350, 200 + i * 100))
    
    def draw_garage_screen(self, screen: pygame.Surface):
        screen.fill((0, 0, 0))
        
        # Draw title
        font = pygame.font.Font(None, 74)
        title = font.render("Garage", True, (255, 255, 255))
        screen.blit(title, (self.width//2 - title.get_width()//2, 50))
        
        # Draw 3D car model for selected vehicle
        try:
            from driving_hand import draw_3d_car_model
            self.menu_car_angle = (self.menu_car_angle + 1) % 360
            draw_3d_car_model(rotation_angle=self.menu_car_angle, vehicle_type=self.selected_vehicle_type, highlight=True)
        except ImportError:
            pass
        # Show real-time car stats below the car
        stats = self.vehicle_manager.get_selected_vehicle_stats_string()
        font = pygame.font.Font(None, 36)
        stats_surface = font.render(stats, True, (255, 255, 255))
        screen.blit(stats_surface, (self.width//2 - stats_surface.get_width()//2, 350))
        # Animate upgrade feedback (simple flash)
        if hasattr(self, 'upgrade_flash') and self.upgrade_flash > 0:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 255, 0, int(120 * self.upgrade_flash)))
            screen.blit(overlay, (0, 0))
            self.upgrade_flash *= 0.9
        # Draw vehicle selection
        for button in self.buttons[Screen.GARAGE]:
            button.draw(screen)
            if button.text != "Customize":
                vehicle_type = self._get_vehicle_type(button.text)
                if vehicle_type:
                    status = self.vehicle_manager.get_unlock_status(vehicle_type)
                    if not status['unlocked']:
                        lock_font = pygame.font.Font(None, 24)
                        lock_text = f"Level {status['level_required']} • {status['coins_required']} coins"
                        lock_surface = lock_font.render(lock_text, True, (200, 200, 200))
                        screen.blit(lock_surface, (350, button.rect.y + 10))
    
    def draw_store_screen(self, screen: pygame.Surface):
        screen.fill((0, 0, 0))
        # Draw title
        font = pygame.font.Font(None, 74)
        title = font.render("Store", True, (255, 255, 255))
        screen.blit(title, (self.width//2 - title.get_width()//2, 50))
        # Show current coins
        coins = self.profile.coins if hasattr(self.profile, 'coins') else 0
        coins_font = pygame.font.Font(None, 36)
        coins_surface = coins_font.render(f"Coins: {coins}", True, (255, 215, 0))
        screen.blit(coins_surface, (self.width - 250, 60))
        # Draw store items
        for button in self.buttons[Screen.STORE]:
            price = self._get_item_price(button.text)
            affordable = coins >= price
            # Animate button if affordable
            button.is_hovered = affordable
            button.draw(screen)
            # Draw item prices
            price_font = pygame.font.Font(None, 24)
            price_color = (0, 255, 0) if affordable else (200, 200, 200)
            price_surface = price_font.render(f"{price} coins", True, price_color)
            screen.blit(price_surface, (350, button.rect.y + 10))
        # Animate purchase feedback (simple flash)
        if hasattr(self, 'purchase_flash') and self.purchase_flash > 0:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((255, 255, 0, int(120 * self.purchase_flash)))
            screen.blit(overlay, (0, 0))
            self.purchase_flash *= 0.9
    
    def draw_settings_screen(self, screen: pygame.Surface):
        screen.fill((0, 0, 0))
        
        # Draw title
        font = pygame.font.Font(None, 74)
        title = font.render("Settings", True, (255, 255, 255))
        screen.blit(title, (self.width//2 - title.get_width()//2, 50))
        
        # Draw settings options
        for button in self.buttons[Screen.SETTINGS]:
            button.draw(screen)
            # Draw current setting value
            value_font = pygame.font.Font(None, 24)
            value = self._get_setting_value(button.text)
            value_surface = value_font.render(str(value), True, (200, 200, 200))
            screen.blit(value_surface, (350, button.rect.y + 10))
    
    def draw_calibration_screen(self, screen: pygame.Surface):
        screen.fill((0, 0, 0))
        
        # Draw title
        font = pygame.font.Font(None, 74)
        title = font.render("Gesture Calibration", True, (255, 255, 255))
        screen.blit(title, (self.width//2 - title.get_width()//2, 50))
        
        # Draw calibration instructions
        inst_font = pygame.font.Font(None, 36)
        instructions = [
            "1. Position yourself in front of the camera",
            "2. Follow the gesture prompts",
            "3. Hold each gesture for 3 seconds",
            "4. Complete all gestures to finish calibration"
        ]
        
        for i, instruction in enumerate(instructions):
            inst_surface = inst_font.render(instruction, True, (200, 200, 200))
            screen.blit(inst_surface, (100, 200 + i * 50))
    
    def draw_tutorial_screen(self, screen: pygame.Surface):
        screen.fill((0, 0, 0))
        
        # Draw title
        font = pygame.font.Font(None, 74)
        title = font.render("Tutorial", True, (255, 255, 255))
        screen.blit(title, (self.width//2 - title.get_width()//2, 50))
        
        # Draw tutorial content
        tut_font = pygame.font.Font(None, 36)
        tutorials = [
            "Steering: Move hand left/right",
            "Acceleration: Push hand forward",
            "Brake: Pull hand back",
            "Gear Shift: Up/down hand movement",
            "Indicators: V-shape with fingers",
            "Headlights: Raise all fingers",
            "Horn: Make a fist",
            "Parking: Pinch fingers"
        ]
        
        for i, tutorial in enumerate(tutorials):
            tut_surface = tut_font.render(tutorial, True, (200, 200, 200))
            screen.blit(tut_surface, (100, 200 + i * 50))
    
    def draw_replay_screen(self, screen: pygame.Surface):
        screen.fill((0, 0, 0))
        
        # Draw title
        font = pygame.font.Font(None, 74)
        title = font.render("Replays", True, (255, 255, 255))
        screen.blit(title, (self.width//2 - title.get_width()//2, 50))
        
        # Draw replay list
        replay_font = pygame.font.Font(None, 36)
        for i, replay in enumerate(self.replays):
            replay_text = f"{replay['date']} - {replay['mode']} - Score: {replay['score']}"
            replay_surface = replay_font.render(replay_text, True, (200, 200, 200))
            screen.blit(replay_surface, (100, 200 + i * 50))
    
    def _get_mode_description(self, mode: str) -> str:
        descriptions = {
            "Free Drive": "Drive freely without objectives",
            "Parking Challenge": "Test your parking skills",
            "Highway Drive": "Maintain high speed on the highway",
            "Fuel Saver": "Complete objectives with limited fuel",
            "Time Trial": "Race against the clock"
        }
        return descriptions.get(mode, "")
    
    def _get_vehicle_type(self, name: str) -> Optional[VehicleType]:
        types = {
            "Sports Car": VehicleType.SPORTS,
            "SUV": VehicleType.SUV,
            "Sedan": VehicleType.SEDAN,
            "Truck": VehicleType.TRUCK
        }
        return types.get(name)
    
    def _get_item_price(self, item: str) -> int:
        prices = {
            "Coins": 1000,
            "Vehicles": 50000,
            "Skins": 2000,
            "Upgrades": 5000
        }
        return prices.get(item, 0)
    
    def _get_setting_value(self, setting: str) -> str:
        if setting in self.settings:
            return str(self.settings[setting])
        return "Not set"
    
    def handle_event(self, event: pygame.event.Event) -> Optional[Screen]:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if self.current_screen != Screen.HOME:
                    return Screen.HOME
        
        if self.current_screen in self.buttons:
            for button in self.buttons[self.current_screen]:
                if button.handle_event(event):
                    return self._handle_button_click(button.text)
        
        return None
    
    def _handle_button_click(self, button_text: str) -> Optional[Screen]:
        if button_text == "Play":
            return Screen.DRIVE_MODE
        elif button_text == "Garage":
            return Screen.GARAGE
        elif button_text == "Store":
            return Screen.STORE
        elif button_text == "Settings":
            return Screen.SETTINGS
        elif button_text == "Profile":
            return Screen.PROFILE
        elif button_text == "Calibration":
            return Screen.CALIBRATION
        elif self.current_screen == Screen.STORE:
            price = self._get_item_price(button_text)
            coins = self.profile.coins if hasattr(self.profile, 'coins') else 0
            if coins >= price:
                self.profile.coins -= price
                self.save_profile()
                self.purchase_flash = 1.0  # Trigger purchase animation
        return None
    
    def update(self):
        # Update any animations or dynamic UI elements
        pass
    
    def draw(self, screen: pygame.Surface):
        if self.current_screen == Screen.HOME:
            self.draw_home_screen(screen)
        elif self.current_screen == Screen.DRIVE_MODE:
            self.draw_drive_mode_screen(screen)
        elif self.current_screen == Screen.GARAGE:
            self.draw_garage_screen(screen)
        elif self.current_screen == Screen.STORE:
            self.draw_store_screen(screen)
        elif self.current_screen == Screen.SETTINGS:
            self.draw_settings_screen(screen)
        elif self.current_screen == Screen.CALIBRATION:
            self.draw_calibration_screen(screen)
        elif self.current_screen == Screen.TUTORIAL:
            self.draw_tutorial_screen(screen)
        elif self.current_screen == Screen.REPLAY:
            self.draw_replay_screen(screen)
    
    def save_profile(self):
        with open('profile.json', 'w') as f:
            json.dump(self.profile.__dict__, f)
    
    def save_settings(self):
        with open('settings.json', 'w') as f:
            json.dump(self.settings, f)
    
    def save_replay(self, replay_data: Dict):
        self.replays.append(replay_data)
        with open('replays.json', 'w') as f:
            json.dump(self.replays, f) 