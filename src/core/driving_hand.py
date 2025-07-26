import cv2
import sys
import pygame
import random
import math
import numpy as np
import psutil
import os
from OpenGL.GL import *
from OpenGL.GLU import *

# Import custom modules with fallbacks
try:
    from src.ai.gesture_controller import GestureController
    from src.core.vehicle_customization import VehicleManager, VehicleType, PaintType, TireType
    from src.core.camera import Camera
    from src.core.game_modes import GameMode, GameModeManager
    from src.ui.game_ui import GameUI, Screen
    from src.ui.game_hud import GameHUD, Gear
    from src.utils.rewards_manager import RewardsManager
    from src.ai.analytics_manager import AnalyticsManager
    from src.utils.sound_manager import SoundManager
    from src.utils.config import config
    from src.utils.logger import logger
except ImportError as e:
    print(f"Warning: Could not import module: {e}")
    # Create working dummy classes
    class DummyClass:
        def __init__(self, *args, **kwargs): 
            pass
        def __getattr__(self, name): 
            return lambda *args, **kwargs: None
        def get_selected_vehicle(self):
            return None
        def get_control_state(self):
            return {}
        def set_screen(self, screen):
            pass
        def handle_click(self, pos):
            return False
        def draw(self, surface):
            pass
        def update(self, *args):
            pass
    
    # Define dummy Screen enum
    class Screen:
        HOME = "HOME"
        DRIVE_MODE = "DRIVE_MODE"
        GARAGE = "GARAGE"
        STORE = "STORE"
        SETTINGS = "SETTINGS"
        PROFILE = "PROFILE"
    
    GestureController = VehicleManager = Camera = GameModeManager = DummyClass
    GameUI = GameHUD = RewardsManager = AnalyticsManager = SoundManager = DummyClass
    config = type('Config', (), {'get': lambda self, *args: args[-1] if args else None})()
    logger = type('Logger', (), {
        'log_error': lambda self, *args: None, 
        'log_game_event': lambda self, *args: None
    })()

# Global constants
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 768
DEFAULT_FPS = 60

class UIButton:
    def __init__(self, x, y, width, height, text, color=(70, 130, 180), hover_color=(100, 149, 237), text_color=(255, 255, 255)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.hovered = False
        self.clicked = False
        self.font = None
        
    def set_font(self, font):
        self.font = font
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.clicked = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.clicked = False
        return False
    
    def draw(self, surface):
        color = self.hover_color if self.hovered else self.color
        if self.clicked:
            color = tuple(max(0, c - 30) for c in color)
        
        # Draw button with rounded corners effect
        pygame.draw.rect(surface, color, self.rect, border_radius=10)
        pygame.draw.rect(surface, (255, 255, 255, 100), self.rect, 2, border_radius=10)
        
        # Draw text
        if self.font:
            text_surface = self.font.render(self.text, True, self.text_color)
            text_rect = text_surface.get_rect(center=self.rect.center)
            surface.blit(text_surface, text_rect)

class GameEngine:
    def __init__(self):
        self.running = False
        self.screen = None
        self.clock = None
        self.cap = None
        self.width = DEFAULT_WIDTH
        self.height = DEFAULT_HEIGHT
        self.fps_limit = DEFAULT_FPS
        
        # UI State
        self.current_screen = "HOME"
        self.selected_car = 0
        self.car_models = ["Sports Car", "SUV", "Truck", "Supercar", "Classic"]
        self.car_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        self.settings = {
            'graphics_quality': 2,  # 0=Low, 1=Medium, 2=High
            'sound_volume': 80,
            'music_volume': 60,
            'difficulty': 1  # 0=Easy, 1=Normal, 2=Hard
        }
        
        # Game components
        self.vehicle_manager = None
        self.camera = None
        self.gesture_controller = None
        self.game_mode_manager = None
        self.game_ui = None
        self.game_hud = None
        self.rewards_manager = None
        self.analytics_manager = None
        self.sound_manager = None
        
        # Game state
        self.in_game = False
        self.last_time = 0
        self.last_save_time = 0
        self.save_interval = 300000  # 5 minutes in milliseconds
        self.opengl_mode = False
        self.fonts = {}
        
        # UI Buttons
        self.buttons = {}
        self.create_ui_buttons()

    def create_ui_buttons(self):
        """Create all UI buttons for different screens"""
        # Home screen buttons
        self.buttons['HOME'] = [
            UIButton(self.width//2 - 150, 300, 300, 60, "DRIVE", (255, 69, 0), (255, 99, 30)),
            UIButton(self.width//2 - 150, 380, 300, 60, "GARAGE", (34, 139, 34), (50, 205, 50)),
            UIButton(self.width//2 - 150, 460, 300, 60, "SETTINGS", (70, 130, 180), (100, 149, 237)),
            UIButton(self.width//2 - 150, 540, 300, 60, "PROFILE", (138, 43, 226), (147, 112, 219)),
            UIButton(self.width//2 - 150, 620, 300, 60, "EXIT", (220, 20, 60), (255, 20, 60))
        ]
        
        # Garage screen buttons
        self.buttons['GARAGE'] = [
            UIButton(50, 500, 120, 50, "< PREV CAR", (70, 130, 180), (100, 149, 237)),
            UIButton(200, 500, 120, 50, "NEXT CAR >", (70, 130, 180), (100, 149, 237)),
            UIButton(400, 500, 120, 50, "CUSTOMIZE", (255, 140, 0), (255, 165, 0)),
            UIButton(550, 500, 120, 50, "TEST DRIVE", (34, 139, 34), (50, 205, 50)),
            UIButton(50, 650, 120, 50, "BACK", (220, 20, 60), (255, 20, 60))
        ]
        
        # Settings screen buttons
        self.buttons['SETTINGS'] = [
            UIButton(300, 250, 150, 40, "LOW", (70, 130, 180), (100, 149, 237)),
            UIButton(470, 250, 150, 40, "MEDIUM", (70, 130, 180), (100, 149, 237)),
            UIButton(640, 250, 150, 40, "HIGH", (70, 130, 180), (100, 149, 237)),
            UIButton(300, 350, 150, 40, "EASY", (34, 139, 34), (50, 205, 50)),
            UIButton(470, 350, 150, 40, "NORMAL", (255, 140, 0), (255, 165, 0)),
            UIButton(640, 350, 150, 40, "HARD", (220, 20, 60), (255, 20, 60)),
            UIButton(50, 650, 120, 50, "BACK", (220, 20, 60), (255, 20, 60)),
            UIButton(self.width - 170, 650, 120, 50, "SAVE", (34, 139, 34), (50, 205, 50))
        ]
        
        # Profile screen buttons
        self.buttons['PROFILE'] = [
            UIButton(300, 400, 200, 50, "ACHIEVEMENTS", (138, 43, 226), (147, 112, 219)),
            UIButton(520, 400, 200, 50, "STATISTICS", (255, 140, 0), (255, 165, 0)),
            UIButton(300, 480, 200, 50, "LEADERBOARD", (34, 139, 34), (50, 205, 50)),
            UIButton(520, 480, 200, 50, "RESET PROGRESS", (220, 20, 60), (255, 20, 60)),
            UIButton(50, 650, 120, 50, "BACK", (220, 20, 60), (255, 20, 60))
        ]
        
        # Game screen buttons
        self.buttons['GAME'] = [
            UIButton(self.width - 150, 20, 120, 40, "PAUSE", (70, 130, 180), (100, 149, 237)),
            UIButton(20, 20, 120, 40, "MAP", (34, 139, 34), (50, 205, 50))
        ]

    def initialize_pygame(self):
        """Initialize pygame with error handling"""
        try:
            pygame.mixer.pre_init(44100, -16, 2, 2048)
            pygame.mixer.init()
            pygame.font.init()
            pygame.init()
            
            # Initialize fonts
            try:
                self.fonts['large'] = pygame.font.Font(None, 48)
                self.fonts['medium'] = pygame.font.Font(None, 36)
                self.fonts['small'] = pygame.font.Font(None, 24)
                self.fonts['title'] = pygame.font.Font(None, 72)
            except:
                self.fonts['large'] = pygame.font.SysFont('Arial', 48, bold=True)
                self.fonts['medium'] = pygame.font.SysFont('Arial', 36)
                self.fonts['small'] = pygame.font.SysFont('Arial', 24)
                self.fonts['title'] = pygame.font.SysFont('Arial', 72, bold=True)
            
            # Set font for all buttons
            for screen_buttons in self.buttons.values():
                for button in screen_buttons:
                    button.set_font(self.fonts['medium'])
            
            print("[OK] Pygame initialized successfully")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to initialize pygame: {e}")
            return False

    def initialize_display(self):
        """Initialize display with fallback options"""
        try:
            # Get display settings
            self.width = getattr(config, 'get', lambda *args: args[-1])('display', 'width', DEFAULT_WIDTH)
            self.height = getattr(config, 'get', lambda *args: args[-1])('display', 'height', DEFAULT_HEIGHT)
            self.fps_limit = getattr(config, 'get', lambda *args: args[-1])('display', 'fps_limit', DEFAULT_FPS)
            
            # Recreate buttons with correct dimensions
            self.create_ui_buttons()
            for screen_buttons in self.buttons.values():
                for button in screen_buttons:
                    button.set_font(self.fonts['medium'])
            
            # Try different display modes (prioritize non-OpenGL for stability)
            display_modes = [
                (pygame.DOUBLEBUF | pygame.HWSURFACE, "Hardware accelerated 2D"),
                (pygame.DOUBLEBUF, "Software 2D"),
                (pygame.OPENGL | pygame.DOUBLEBUF, "OpenGL"),
                (0, "Basic mode")
            ]
            
            for flags, mode_name in display_modes:
                try:
                    if flags & pygame.OPENGL:
                        # Set OpenGL attributes before creating OpenGL surface
                        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
                        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
                    
                    self.screen = pygame.display.set_mode((self.width, self.height), flags)
                    pygame.display.set_caption("3D Hand-Controlled Driving")
                    
                    if flags & pygame.OPENGL:
                        self.opengl_mode = True
                        self.initialize_opengl()
                    else:
                        self.opengl_mode = False
                    
                    print(f"[OK] Display initialized with {mode_name}")
                    return True
                    
                except pygame.error as e:
                    print(f"[WARN] {mode_name} failed: {e}")
                    continue
            
            raise Exception("All display modes failed")
            
        except Exception as e:
            print(f"[ERROR] Display initialization failed: {e}")
            return False

    def initialize_opengl(self):
        """Initialize OpenGL settings"""
        try:
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            # Set up viewport and perspective
            glViewport(0, 0, self.width, self.height)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45, (self.width / self.height), 0.1, 50.0)
            glMatrixMode(GL_MODELVIEW)
            
            # Clear color
            glClearColor(0.1, 0.1, 0.2, 1.0)  # Dark blue instead of black
            
            print("[OK] OpenGL initialized successfully")
        except Exception as e:
            print(f"[WARN] OpenGL initialization failed: {e}")
            self.opengl_mode = False

    def initialize_camera(self):
        """Initialize camera with error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                backends = [cv2.CAP_DSHOW, cv2.CAP_ANY, 0]
                for backend in backends:
                    if backend != 0:
                        self.cap = cv2.VideoCapture(0, backend)
                    else:
                        self.cap = cv2.VideoCapture(0)
                    
                    if self.cap.isOpened():
                        # Set camera properties
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        print("[OK] Camera initialized successfully")
                        return True
                    
                print(f"[WARN] Camera initialization attempt {attempt + 1} failed")
                if attempt < max_retries - 1:
                    pygame.time.wait(1000)
                    
            except Exception as e:
                print(f"[WARN] Camera error (attempt {attempt + 1}): {e}")
        
        print("[WARN] Proceeding without camera")
        return False

    def initialize_game_components(self):
        """Initialize all game components"""
        try:
            self.vehicle_manager = VehicleManager()
            self.camera = Camera()
            self.gesture_controller = GestureController()
            self.game_mode_manager = GameModeManager()
            self.game_ui = GameUI(self.width, self.height)
            self.game_hud = GameHUD(self.width, self.height)
            self.rewards_manager = RewardsManager()
            self.analytics_manager = AnalyticsManager()
            self.sound_manager = SoundManager()
            
            # Configure sound
            if hasattr(self.sound_manager, 'set_volume'):
                sfx_volume = getattr(config, 'get', lambda *args: args[-1])('audio', 'sfx_volume', 0.8)
                music_volume = getattr(config, 'get', lambda *args: args[-1])('audio', 'music_volume', 0.5)
                self.sound_manager.set_volume(sfx_volume)
                self.sound_manager.set_music_volume(music_volume)
            
            # Start analytics
            if hasattr(self.analytics_manager, 'start_session'):
                self.analytics_manager.start_session()
            
            print("[OK] Game components initialized")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize game components: {e}")
            return False

    def draw_gradient_background(self, color1, color2):
        """Draw a gradient background"""
        for y in range(self.height):
            progress = y / self.height
            r = int(color1[0] * (1 - progress) + color2[0] * progress)
            g = int(color1[1] * (1 - progress) + color2[1] * progress)
            b = int(color1[2] * (1 - progress) + color2[2] * progress)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.width, y))

    def draw_home_screen(self):
        """Draw the home screen"""
        # Gradient background
        self.draw_gradient_background((20, 30, 60), (60, 80, 120))
        
        # Animated title
        time = pygame.time.get_ticks() / 1000.0
        title_y = 100 + math.sin(time) * 5
        
        # Shadow effect for title
        shadow_text = self.fonts['title'].render("DRIVE MASTER", True, (0, 0, 0))
        shadow_rect = shadow_text.get_rect(center=(self.width // 2 + 3, title_y + 3))
        self.screen.blit(shadow_text, shadow_rect)
        
        # Main title
        title_text = self.fonts['title'].render("DRIVE MASTER", True, (255, 215, 0))
        title_rect = title_text.get_rect(center=(self.width // 2, title_y))
        self.screen.blit(title_text, title_rect)
        
        # Subtitle
        subtitle_text = self.fonts['medium'].render("Hand-Controlled Racing Experience", True, (200, 200, 200))
        subtitle_rect = subtitle_text.get_rect(center=(self.width // 2, title_y + 80))
        self.screen.blit(subtitle_text, subtitle_rect)
        
        # Draw buttons
        for button in self.buttons['HOME']:
            button.draw(self.screen)
        
        # Status indicators
        status_y = self.height - 100
        status_items = [
            f"Camera: {'✓' if self.cap and self.cap.isOpened() else '✗'}",
            f"OpenGL: {'✓' if self.opengl_mode else '✗'}",
            f"Resolution: {self.width}x{self.height}",
            f"FPS: {int(self.clock.get_fps()) if self.clock else 0}"
        ]
        
        for i, item in enumerate(status_items):
            color = (0, 255, 0) if '✓' in item else (255, 255, 255)
            text = self.fonts['small'].render(item, True, color)
            self.screen.blit(text, (20 + i * 200, status_y))

    def draw_garage_screen(self):
        """Draw the garage screen"""
        # Dark background
        self.draw_gradient_background((40, 40, 60), (20, 20, 40))
        
        # Title
        title_text = self.fonts['large'].render("GARAGE", True, (255, 215, 0))
        title_rect = title_text.get_rect(center=(self.width // 2, 50))
        self.screen.blit(title_text, title_rect)
        
        # Car display area
        car_area = pygame.Rect(self.width // 2 - 200, 120, 400, 300)
        pygame.draw.rect(self.screen, (60, 60, 80), car_area, border_radius=15)
        pygame.draw.rect(self.screen, (100, 100, 120), car_area, 3, border_radius=15)
        
        # Draw car representation
        car_color = self.car_colors[self.selected_car % len(self.car_colors)]
        car_rect = pygame.Rect(car_area.centerx - 80, car_area.centery - 30, 160, 60)
        pygame.draw.ellipse(self.screen, car_color, car_rect)
        
        # Car wheels
        wheel_color = (50, 50, 50)
        pygame.draw.circle(self.screen, wheel_color, (car_rect.left + 30, car_rect.bottom - 10), 15)
        pygame.draw.circle(self.screen, wheel_color, (car_rect.right - 30, car_rect.bottom - 10), 15)
        
        # Car info
        car_name = self.car_models[self.selected_car % len(self.car_models)]
        name_text = self.fonts['medium'].render(car_name, True, (255, 255, 255))
        name_rect = name_text.get_rect(center=(self.width // 2, car_area.bottom + 30))
        self.screen.blit(name_text, name_rect)
        
        # Car stats
        stats = ["Speed: ★★★★☆", "Handling: ★★★☆☆", "Acceleration: ★★★★★"]
        for i, stat in enumerate(stats):
            stat_text = self.fonts['small'].render(stat, True, (200, 200, 200))
            self.screen.blit(stat_text, (self.width // 2 - 100, name_rect.bottom + 20 + i * 25))
        
        # Draw buttons
        for button in self.buttons['GARAGE']:
            button.draw(self.screen)

    def draw_settings_screen(self):
        """Draw the settings screen"""
        # Dark background
        self.draw_gradient_background((30, 30, 50), (50, 30, 70))
        
        # Title
        title_text = self.fonts['large'].render("SETTINGS", True, (255, 215, 0))
        title_rect = title_text.get_rect(center=(self.width // 2, 50))
        self.screen.blit(title_text, title_rect)
        
        # Graphics Quality section
        graphics_text = self.fonts['medium'].render("Graphics Quality:", True, (255, 255, 255))
        self.screen.blit(graphics_text, (50, 200))
        
        # Difficulty section
        difficulty_text = self.fonts['medium'].render("Difficulty:", True, (255, 255, 255))
        self.screen.blit(difficulty_text, (50, 300))
        
        # Volume sliders simulation
        volume_text = self.fonts['medium'].render("Sound Volume:", True, (255, 255, 255))
        self.screen.blit(volume_text, (50, 400))
        
        # Volume bar
        volume_bar = pygame.Rect(300, 410, 300, 20)
        pygame.draw.rect(self.screen, (100, 100, 100), volume_bar)
        volume_fill = pygame.Rect(300, 410, int(300 * self.settings['sound_volume'] / 100), 20)
        pygame.draw.rect(self.screen, (0, 255, 0), volume_fill)
        
        volume_value = self.fonts['small'].render(f"{self.settings['sound_volume']}%", True, (255, 255, 255))
        self.screen.blit(volume_value, (620, 410))
        
        # Music volume
        music_text = self.fonts['medium'].render("Music Volume:", True, (255, 255, 255))
        self.screen.blit(music_text, (50, 450))
        
        music_bar = pygame.Rect(300, 460, 300, 20)
        pygame.draw.rect(self.screen, (100, 100, 100), music_bar)
        music_fill = pygame.Rect(300, 460, int(300 * self.settings['music_volume'] / 100), 20)
        pygame.draw.rect(self.screen, (0, 100, 255), music_fill)
        
        music_value = self.fonts['small'].render(f"{self.settings['music_volume']}%", True, (255, 255, 255))
        self.screen.blit(music_value, (620, 460))
        
        # Highlight selected options
        quality_options = ["LOW", "MEDIUM", "HIGH"]
        for i, button in enumerate(self.buttons['SETTINGS'][:3]):
            if i == self.settings['graphics_quality']:
                pygame.draw.rect(self.screen, (255, 215, 0), button.rect, 3, border_radius=10)
        
        difficulty_options = ["EASY", "NORMAL", "HARD"]
        for i, button in enumerate(self.buttons['SETTINGS'][3:6]):
            if i == self.settings['difficulty']:
                pygame.draw.rect(self.screen, (255, 215, 0), button.rect, 3, border_radius=10)
        
        # Draw buttons
        for button in self.buttons['SETTINGS']:
            button.draw(self.screen)

    def draw_profile_screen(self):
        """Draw the profile screen"""
        # Gradient background
        self.draw_gradient_background((40, 20, 60), (20, 40, 80))
        
        # Title
        title_text = self.fonts['large'].render("PLAYER PROFILE", True, (255, 215, 0))
        title_rect = title_text.get_rect(center=(self.width // 2, 50))
        self.screen.blit(title_text, title_rect)
        
        # Player avatar area
        avatar_rect = pygame.Rect(50, 120, 150, 150)
        pygame.draw.rect(self.screen, (100, 100, 150), avatar_rect, border_radius=75)
        pygame.draw.rect(self.screen, (200, 200, 255), avatar_rect, 3, border_radius=75)
        
        # Player icon
        pygame.draw.circle(self.screen, (255, 255, 255), avatar_rect.center, 40)
        pygame.draw.circle(self.screen, (100, 100, 150), (avatar_rect.centerx, avatar_rect.centery - 10), 15)
        pygame.draw.arc(self.screen, (100, 100, 150), 
                       (avatar_rect.centerx - 25, avatar_rect.centery + 10, 50, 30), 0, math.pi, 5)
        
        # Player stats
        stats_x = 250
        stats = [
            "Player: Racing Pro",
            "Level: 15",
            "Total Races: 127",
            "Wins: 89",
            "Best Lap: 1:23.45",
            "Total Distance: 2,847 km"
        ]
        
        for i, stat in enumerate(stats):
            stat_text = self.fonts['medium'].render(stat, True, (255, 255, 255))
            self.screen.blit(stat_text, (stats_x, 140 + i * 35))
        
        # Achievements preview
        achievements_text = self.fonts['medium'].render("Recent Achievements:", True, (255, 215, 0))
        self.screen.blit(achievements_text, (50, 320))
        
        achievements = ["🏆 Speed Demon", "🥇 Perfect Driver", "⚡ Lightning Fast"]
        for i, achievement in enumerate(achievements):
            achievement_text = self.fonts['small'].render(achievement, True, (200, 255, 200))
            self.screen.blit(achievement_text, (50, 350 + i * 25))
        
        # Draw buttons
        for button in self.buttons['PROFILE']:
            button.draw(self.screen)

    def draw_game_scene(self):
        """Draw the game scene when in game mode"""
        if self.opengl_mode:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            
            # Simple 3D scene
            glTranslatef(0.0, -1.0, -5.0)
            
            # Draw ground
            glColor3f(0.2, 0.5, 0.2)  # Green ground
            glBegin(GL_QUADS)
            glVertex3f(-10.0, 0.0, -10.0)
            glVertex3f(10.0, 0.0, -10.0)
            glVertex3f(10.0, 0.0, 10.0)
            glVertex3f(-10.0, 0.0, 10.0)
            glEnd()
            
            # Draw simple car
            glTranslatef(0.0, 0.5, 0.0)
            glColor3f(1.0, 0.0, 0.0)  # Red car
            
            # Car body
            glPushMatrix()
            glScalef(2.0, 0.5, 1.0)
            self.draw_cube()
            glPopMatrix()
            
            # Car wheels
            glColor3f(0.1, 0.1, 0.1)  # Black wheels
            positions = [(-0.7, -0.3, 0.4), (0.7, -0.3, 0.4), (-0.7, -0.3, -0.4), (0.7, -0.3, -0.4)]
            for pos in positions:
                glPushMatrix()
                glTranslatef(*pos)
                glScalef(0.3, 0.3, 0.1)
                self.draw_cube()
                glPopMatrix()
        else:
            # 2D game scene with enhanced graphics
            self.draw_gradient_background((50, 100, 50), (100, 150, 100))
            
            # Draw enhanced road with perspective
            road_points = [
                (self.width//4, 0), (3*self.width//4, 0),
                (self.width//3, self.height), (2*self.width//3, self.height)
            ]
            pygame.draw.polygon(self.screen, (60, 60, 60), road_points)
            
            # Draw road markings with animation
            time = pygame.time.get_ticks() / 100.0
            for i in range(20):
                y = (i * 50 - int(time) % 50) % self.height
                if y > 0:
                    # Calculate road width at this y position
                    progress = y / self.height
                    left_x = self.width//4 + progress * (self.width//12)
                    right_x = 3*self.width//4 - progress * (self.width//12)
                    center_x = (left_x + right_x) // 2
                    
                    # Draw center line
                    line_width = max(2, int(10 * (1 - progress * 0.8)))
                    pygame.draw.rect(self.screen, (255, 255, 255), 
                                   (center_x - line_width//2, y, line_width, 20))
            
            # Draw enhanced car with 3D effect
            car_color = self.car_colors[self.selected_car % len(self.car_colors)]
            car_x = self.width // 2 - 30
            car_y = self.height - 120
            
            # Car shadow
            shadow_points = [(car_x + 5, car_y + 85), (car_x + 65, car_y + 85),
                           (car_x + 55, car_y + 15), (car_x + 15, car_y + 15)]
            pygame.draw.polygon(self.screen, (0, 0, 0, 100), shadow_points)
            
            # Car body
            car_points = [(car_x, car_y + 80), (car_x + 60, car_y + 80),
                         (car_x + 50, car_y + 10), (car_x + 10, car_y + 10)]
            pygame.draw.polygon(self.screen, car_color, car_points)
            
            # Car details
            # Windows
            window_color = (100, 150, 255)
            window_points = [(car_x + 15, car_y + 25), (car_x + 45, car_y + 25),
                           (car_x + 40, car_y + 15), (car_x + 20, car_y + 15)]
            pygame.draw.polygon(self.screen, window_color, window_points)
            
            # Headlights
            pygame.draw.circle(self.screen, (255, 255, 200), (car_x + 15, car_y + 15), 5)
            pygame.draw.circle(self.screen, (255, 255, 200), (car_x + 45, car_y + 15), 5)
            
            # Wheels
            wheel_color = (30, 30, 30)
            pygame.draw.circle(self.screen, wheel_color, (car_x + 15, car_y + 70), 10)
            pygame.draw.circle(self.screen, wheel_color, (car_x + 45, car_y + 70), 10)
            pygame.draw.circle(self.screen, wheel_color, (car_x + 15, car_y + 25), 8)
            pygame.draw.circle(self.screen, wheel_color, (car_x + 45, car_y + 25), 8)
            
            # Wheel rims
            rim_color = (150, 150, 150)
            pygame.draw.circle(self.screen, rim_color, (car_x + 15, car_y + 70), 6)
            pygame.draw.circle(self.screen, rim_color, (car_x + 45, car_y + 70), 6)
            pygame.draw.circle(self.screen, rim_color, (car_x + 15, car_y + 25), 5)
            pygame.draw.circle(self.screen, rim_color, (car_x + 45, car_y + 25), 5)
        
        # Draw game HUD
        self.draw_game_hud()
        
        # Draw game buttons
        for button in self.buttons['GAME']:
            button.draw(self.screen)

    def draw_game_hud(self):
        """Draw the game HUD with speedometer, minimap, etc."""
        if self.opengl_mode:
            return  # Skip HUD for OpenGL mode for now
        
        # Semi-transparent HUD background
        hud_surface = pygame.Surface((self.width, 150), pygame.SRCALPHA)
        hud_surface.fill((0, 0, 0, 128))
        self.screen.blit(hud_surface, (0, self.height - 150))
        
        # Speedometer
        speed = random.randint(80, 120)  # Simulated speed
        speed_text = self.fonts['large'].render(f"{speed}", True, (255, 255, 255))
        speed_label = self.fonts['small'].render("KM/H", True, (200, 200, 200))
        
        self.screen.blit(speed_text, (50, self.height - 120))
        self.screen.blit(speed_label, (50, self.height - 80))
        
        # Draw speedometer circle
        center = (150, self.height - 75)
        pygame.draw.circle(self.screen, (100, 100, 100), center, 40, 3)
        
        # Speed needle
        angle = math.radians(-90 + (speed / 200) * 180)
        end_x = center[0] + 35 * math.cos(angle)
        end_y = center[1] + 35 * math.sin(angle)
        pygame.draw.line(self.screen, (255, 0, 0), center, (end_x, end_y), 3)
        
        # Gear indicator
        gear = random.choice(['1', '2', '3', '4', '5', 'D'])
        gear_text = self.fonts['large'].render(f"GEAR: {gear}", True, (255, 255, 255))
        self.screen.blit(gear_text, (250, self.height - 120))
        
        # Mini map
        minimap_rect = pygame.Rect(self.width - 200, self.height - 140, 180, 120)
        pygame.draw.rect(self.screen, (50, 50, 50), minimap_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), minimap_rect, 2)
        
        # Draw road on minimap
        road_rect = pygame.Rect(minimap_rect.centerx - 10, minimap_rect.y + 10, 20, 100)
        pygame.draw.rect(self.screen, (100, 100, 100), road_rect)
        
        # Player position on minimap
        player_pos = (minimap_rect.centerx, minimap_rect.bottom - 20)
        pygame.draw.circle(self.screen, (255, 0, 0), player_pos, 3)
        
        # Lap time
        lap_time = "1:23.45"
        time_text = self.fonts['medium'].render(f"LAP: {lap_time}", True, (255, 255, 0))
        self.screen.blit(time_text, (self.width - 400, self.height - 40))

    def draw_cube(self):
        """Draw a simple cube using OpenGL"""
        glBegin(GL_QUADS)
        
        # Front face
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        
        # Back face
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(-0.5, 0.5, -0.5)
        glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(0.5, -0.5, -0.5)
        
        # Top face
        glVertex3f(-0.5, 0.5, -0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(0.5, 0.5, -0.5)
        
        # Bottom face
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        
        # Right face
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(0.5, -0.5, 0.5)
        
        # Left face
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, -0.5)
        
        glEnd()

    def handle_button_clicks(self, event):
        """Handle button clicks for different screens"""
        if self.current_screen == "HOME":
            for i, button in enumerate(self.buttons['HOME']):
                if button.handle_event(event):
                    if i == 0:  # DRIVE
                        self.in_game = True
                        self.current_screen = "GAME"
                        print("Started driving mode")
                    elif i == 1:  # GARAGE
                        self.current_screen = "GARAGE"
                        print("Entered garage")
                    elif i == 2:  # SETTINGS
                        self.current_screen = "SETTINGS"
                        print("Opened settings")
                    elif i == 3:  # PROFILE
                        self.current_screen = "PROFILE"
                        print("Opened profile")
                    elif i == 4:  # EXIT
                        self.running = False
                        print("Exiting game")
                    return True
        
        elif self.current_screen == "GARAGE":
            for i, button in enumerate(self.buttons['GARAGE']):
                if button.handle_event(event):
                    if i == 0:  # PREV CAR
                        self.selected_car = (self.selected_car - 1) % len(self.car_models)
                        print(f"Selected car: {self.car_models[self.selected_car]}")
                    elif i == 1:  # NEXT CAR
                        self.selected_car = (self.selected_car + 1) % len(self.car_models)
                        print(f"Selected car: {self.car_models[self.selected_car]}")
                    elif i == 2:  # CUSTOMIZE
                        print("Car customization opened")
                    elif i == 3:  # TEST DRIVE
                        self.in_game = True
                        self.current_screen = "GAME"
                        print("Started test drive")
                    elif i == 4:  # BACK
                        self.current_screen = "HOME"
                        print("Returned to home")
                    return True
        
        elif self.current_screen == "SETTINGS":
            for i, button in enumerate(self.buttons['SETTINGS']):
                if button.handle_event(event):
                    if i < 3:  # Graphics quality buttons
                        self.settings['graphics_quality'] = i
                        print(f"Graphics quality set to: {['Low', 'Medium', 'High'][i]}")
                    elif i < 6:  # Difficulty buttons
                        self.settings['difficulty'] = i - 3
                        print(f"Difficulty set to: {['Easy', 'Normal', 'Hard'][i-3]}")
                    elif i == 6:  # BACK
                        self.current_screen = "HOME"
                        print("Returned to home")
                    elif i == 7:  # SAVE
                        print("Settings saved")
                    return True
        
        elif self.current_screen == "PROFILE":
            for i, button in enumerate(self.buttons['PROFILE']):
                if button.handle_event(event):
                    if i == 0:  # ACHIEVEMENTS
                        print("Achievements screen opened")
                    elif i == 1:  # STATISTICS
                        print("Statistics screen opened")
                    elif i == 2:  # LEADERBOARD
                        print("Leaderboard opened")
                    elif i == 3:  # RESET PROGRESS
                        print("Progress reset confirmation")
                    elif i == 4:  # BACK
                        self.current_screen = "HOME"
                        print("Returned to home")
                    return True
        
        elif self.current_screen == "GAME":
            for i, button in enumerate(self.buttons['GAME']):
                if button.handle_event(event):
                    if i == 0:  # PAUSE
                        print("Game paused")
                    elif i == 1:  # MAP
                        print("Full map opened")
                    return True
        
        return False

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            
            # Handle button clicks
            if self.handle_button_clicks(event):
                if self.sound_manager and hasattr(self.sound_manager, 'play_sound'):
                    self.sound_manager.play_sound('button_click')
            
            # Handle mouse events for sliders in settings
            if event.type == pygame.MOUSEBUTTONDOWN and self.current_screen == "SETTINGS":
                mouse_x, mouse_y = event.pos
                
                # Sound volume slider
                if 300 <= mouse_x <= 600 and 410 <= mouse_y <= 430:
                    self.settings['sound_volume'] = int((mouse_x - 300) / 3)
                    print(f"Sound volume: {self.settings['sound_volume']}%")
                
                # Music volume slider
                elif 300 <= mouse_x <= 600 and 460 <= mouse_y <= 480:
                    self.settings['music_volume'] = int((mouse_x - 300) / 3)
                    print(f"Music volume: {self.settings['music_volume']}%")
            
            # Handle all button mouse events
            for screen_buttons in self.buttons.values():
                for button in screen_buttons:
                    button.handle_event(event)
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.in_game:
                        self.in_game = False
                        self.current_screen = "HOME"
                        print("Returned to menu")
                    elif self.current_screen != "HOME":
                        self.current_screen = "HOME"
                        print("Returned to home")
                    else:
                        self.running = False
                        return False
                
                elif event.key == pygame.K_RETURN and self.current_screen == "HOME":
                    self.in_game = True
                    self.current_screen = "GAME"
                    print("Started game mode")
                    if self.sound_manager and hasattr(self.sound_manager, 'play_sound'):
                        self.sound_manager.play_sound('game_start')
        
        return True

    def update_game_logic(self, dt):
        """Update game logic"""
        try:
            # Process camera input if available
            if self.cap and self.gesture_controller and self.in_game:
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.flip(frame, 1)
                    
                    if hasattr(self.gesture_controller, 'process_frame'):
                        frame = self.gesture_controller.process_frame(frame)
                        control_state = self.gesture_controller.get_control_state()
                    
                    # Display camera feed in separate window
                    cv2.imshow('Hand Tracking', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
            
            # Update other components if they exist
            if self.game_mode_manager and hasattr(self.game_mode_manager, 'update'):
                self.game_mode_manager.update(None, dt)
                
        except Exception as e:
            print(f"Error in game logic: {e}")

    def render(self):
        """Render the game"""
        try:
            if self.in_game and self.current_screen == "GAME":
                self.draw_game_scene()
            elif self.current_screen == "HOME":
                self.draw_home_screen()
            elif self.current_screen == "GARAGE":
                self.draw_garage_screen()
            elif self.current_screen == "SETTINGS":
                self.draw_settings_screen()
            elif self.current_screen == "PROFILE":
                self.draw_profile_screen()
            
            pygame.display.flip()
            
        except Exception as e:
            print(f"Error in rendering: {e}")
            # Fallback rendering
            if not self.opengl_mode:
                self.screen.fill((50, 0, 0))  # Red background to indicate error
                if self.fonts.get('medium'):
                    error_text = self.fonts['medium'].render("Rendering Error - Check Console", True, (255, 255, 255))
                    self.screen.blit(error_text, (10, 10))
                pygame.display.flip()

    def cleanup(self):
        """Clean up resources"""
        try:
            if self.cap:
                self.cap.release()
            
            if self.analytics_manager and hasattr(self.analytics_manager, 'end_session'):
                self.analytics_manager.end_session()
            
            if self.sound_manager and hasattr(self.sound_manager, 'cleanup'):
                self.sound_manager.cleanup()
            
            cv2.destroyAllWindows()
            pygame.quit()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def run(self):
        """Main game loop"""
        if not self.initialize_pygame():
            return
        
        if not self.initialize_display():
            pygame.quit()
            return
        
        self.initialize_camera()
        self.initialize_game_components()
        
        self.clock = pygame.time.Clock()
        self.running = True
        self.last_time = pygame.time.get_ticks()
        self.last_save_time = self.last_time
        
        print("Starting main game loop...")
        print("Controls: Mouse = Navigate UI, ESC = Back/Exit, ENTER = Quick Start")
        print("Features: Garage, Settings, Profile, Full Game Mode")
        
        while self.running:
            current_time = pygame.time.get_ticks()
            dt = (current_time - self.last_time) / 1000.0
            self.last_time = current_time
            
            # Handle events
            if not self.handle_events():
                break
            
            # Update game logic
            self.update_game_logic(dt)
            
            # Render
            self.render()
            
            # Control frame rate
            self.clock.tick(self.fps_limit)
        
        self.cleanup()


def main():
    """Main entry point"""
    print("=== Starting Enhanced 3D Hand-Controlled Driving Game ===")
    print("Features:")
    print("- Professional UI with multiple screens")
    print("- Garage with car selection")
    print("- Settings with graphics and difficulty options")
    print("- Player profile with stats and achievements")
    print("- Enhanced game mode with HUD")
    print("- Hand gesture controls (when camera available)")
    
    # Create and run game engine
    game = GameEngine()
    game.run()
    
    print("Game ended.")


if __name__ == "__main__":
    main()