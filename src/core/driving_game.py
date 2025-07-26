import pygame
import math
import random
from typing import List, Tuple, Dict, Optional
from src.core.vehicle import Vehicle, GearMode
from src.core.traffic_system import TrafficSystem
from src.core.environment import Environment, WeatherSystem, WeatherType, TimeOfDay
from src.core.game_modes import CareerMode, Mission, TimeTrial, ParkingChallenge, DeliveryMission
from src.core.ui_system import DrivingHUD

# Initialize Pygame
pygame.init()
pygame.mixer.init()

# Constants
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FPS = 60

# Game States
class GameState:
    MENU = 'menu'
    PLAYING = 'playing'
    PAUSED = 'paused'
    GAME_OVER = 'game_over'

class Game:
    def __init__(self):
        # Initialize display and clock
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('Dr. Driving')
        self.clock = pygame.time.Clock()
        
        # Game state
        self.state = GameState.MENU
        self.running = True
        
        # Initialize systems
        self.environment = Environment(2000, 2000)  # Large world size
        self.traffic_system = TrafficSystem(2000, 2000)
        self.career_mode = CareerMode()
        self.hud = DrivingHUD(WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Initialize player vehicle
        self.player = Vehicle(WINDOW_WIDTH//2, WINDOW_HEIGHT//2, 'player_car')
        
        # Camera
        self.camera_x = 0
        self.camera_y = 0
        self.camera_mode = 'follow'  # 'follow', 'fixed', 'orbit'
        
        # Sound effects
        self.sounds = self._load_sounds()
        
        # Create initial missions
        self._setup_missions()

    def _load_sounds(self) -> Dict[str, Optional[pygame.mixer.Sound]]:
        sounds = {}
        sound_files = {
            'engine': 'sounds/engine.wav',
            'brake': 'sounds/brake.wav',
            'crash': 'sounds/crash.wav',
            'horn': 'sounds/horn.wav',
            'indicator': 'sounds/indicator.wav'
        }
        
        for name, path in sound_files.items():
            try:
                sounds[name] = pygame.mixer.Sound(path)
            except FileNotFoundError:
                print(f"Warning: Sound file {path} not found")
                sounds[name] = None
        
        return sounds

    def _setup_missions(self) -> None:
        # Time trial mission
        checkpoints = [
            (500, 500),
            (1500, 500),
            (1500, 1500),
            (500, 1500)
        ]
        self.career_mode.add_mission(TimeTrial(checkpoints, 120))

        # Parking mission
        self.career_mode.add_mission(ParkingChallenge((1000, 1000, 90)))

        # Delivery mission
        self.career_mode.add_mission(DeliveryMission((500, 500), (1500, 1500), 0.5))

    def _update_camera(self) -> None:
        if self.camera_mode == 'follow':
            # Smooth camera following
            target_x = self.player.x - WINDOW_WIDTH//2
            target_y = self.player.y - WINDOW_HEIGHT//2
            
            self.camera_x += (target_x - self.camera_x) * 0.1
            self.camera_y += (target_y - self.camera_y) * 0.1
        
        elif self.camera_mode == 'orbit':
            # Orbital camera
            radius = 200
            angle = math.radians(self.player.angle)
            self.camera_x = self.player.x - WINDOW_WIDTH//2 + math.cos(angle) * radius
            self.camera_y = self.player.y - WINDOW_HEIGHT//2 + math.sin(angle) * radius

    def _handle_input(self) -> None:
        keys = pygame.key.get_pressed()
        
        # Vehicle controls
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.player.engine_force = self.player.acceleration
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.player.brake_force = self.player.brake_power
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.player.steering_angle = -30
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.player.steering_angle = 30
        if keys[pygame.K_SPACE]:
            self.player.handbrake_active = True
        
        # Reset forces if keys not pressed
        if not (keys[pygame.K_w] or keys[pygame.K_UP]):
            self.player.engine_force = 0
        if not (keys[pygame.K_s] or keys[pygame.K_DOWN]):
            self.player.brake_force = 0
        if not (keys[pygame.K_a] or keys[pygame.K_LEFT]) and not (keys[pygame.K_d] or keys[pygame.K_RIGHT]):
            self.player.steering_angle = 0
        if not keys[pygame.K_SPACE]:
            self.player.handbrake_active = False

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.state == GameState.PLAYING:
                        self.state = GameState.PAUSED
                    elif self.state == GameState.PAUSED:
                        self.state = GameState.PLAYING
                
                elif event.key == pygame.K_h:
                    self.player.horn_active = True
                    if self.sounds['horn']:
                        self.sounds['horn'].play()
                
                elif event.key == pygame.K_c:
                    # Cycle camera modes
                    if self.camera_mode == 'follow':
                        self.camera_mode = 'orbit'
                    elif self.camera_mode == 'orbit':
                        self.camera_mode = 'fixed'
                    else:
                        self.camera_mode = 'follow'
                
                elif event.key == pygame.K_r:
                    if self.state == GameState.GAME_OVER:
                        self.__init__()  # Reset game
                
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_h:
                    self.player.horn_active = False
            
            # Handle UI events
            self.hud.handle_event(event)

    def update(self) -> None:
        if self.state != GameState.PLAYING:
            return

        dt = self.clock.tick(FPS) / 1000.0  # Convert to seconds

        # Update player vehicle
        self._handle_input()
        self.player.update_physics(dt)

        # Update traffic system
        self.traffic_system.update(dt)

        # Update environment
        self.environment.update(dt)

        # Update career mode and current mission
        self.career_mode.update(dt, self.player)

        # Update camera
        self._update_camera()

        # Update HUD
        self.hud.update(self.player)

        # Check for game over conditions
        if self.player.fuel_level <= 0 or self.player.current_health <= 0:
            self.state = GameState.GAME_OVER

    def draw(self) -> None:
        self.screen.fill((100, 100, 100))  # Gray background

        # Calculate camera offset
        offset_x = int(self.camera_x)
        offset_y = int(self.camera_y)

        # Draw environment
        self.environment.draw(self.screen)

        # Draw traffic
        self.traffic_system.draw(self.screen)

        # Draw player vehicle
        self.player.draw(self.screen)

        # Draw career mode elements
        self.career_mode.draw(self.screen)

        # Draw HUD
        self.hud.draw(self.screen)

        # Draw game state overlays
        if self.state == GameState.PAUSED:
            self._draw_pause_menu()
        elif self.state == GameState.GAME_OVER:
            self._draw_game_over()
        elif self.state == GameState.MENU:
            self._draw_main_menu()

        pygame.display.flip()

    def _draw_pause_menu(self) -> None:
        # Draw semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(128)
        self.screen.blit(overlay, (0, 0))

        # Draw pause menu text
        font = pygame.font.Font(None, 74)
        text = font.render('PAUSED', True, (255, 255, 255))
        text_rect = text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2))
        self.screen.blit(text, text_rect)

    def _draw_game_over(self) -> None:
        # Draw semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(192)
        self.screen.blit(overlay, (0, 0))

        # Draw game over text
        font = pygame.font.Font(None, 74)
        text = font.render('GAME OVER', True, (255, 0, 0))
        text_rect = text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2))
        self.screen.blit(text, text_rect)

        # Draw restart prompt
        font = pygame.font.Font(None, 36)
        text = font.render('Press R to Restart', True, (255, 255, 255))
        text_rect = text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + 50))
        self.screen.blit(text, text_rect)

    def _draw_main_menu(self) -> None:
        # Draw title
        font = pygame.font.Font(None, 96)
        text = font.render('Dr. Driving', True, (255, 255, 255))
        text_rect = text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//3))
        self.screen.blit(text, text_rect)

        # Draw menu options
        font = pygame.font.Font(None, 48)
        options = ['Press ENTER to Start', 'Press ESC to Quit']
        for i, option in enumerate(options):
            text = font.render(option, True, (255, 255, 255))
            text_rect = text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + i * 50))
            self.screen.blit(text, text_rect)

    def run(self) -> None:
        while self.running:
            self.handle_events()
            self.update()
            self.draw()

        pygame.quit()

if __name__ == "__main__":
    game = Game()
    game.run()
