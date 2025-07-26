import cv2
import sys
import pygame
import random
import math
import numpy as np
import psutil
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from gesture_controller import GestureController
from vehicle_customization import VehicleManager, VehicleType, PaintType, TireType
from camera import Camera
from game_modes import GameMode, GameModeManager
from game_ui import GameUI, Screen
from game_hud import GameHUD, Gear
from rewards_manager import RewardsManager
from analytics_manager import AnalyticsManager
from sound_manager import SoundManager
from config import config
from logger import logger

glutInit()

# Initialize pygame and OpenGL
pygame.init()

# Get display settings from config
W = config.get('display', 'width', 800)
H = config.get('display', 'height', 600)
FPS_LIMIT = config.get('display', 'fps_limit', 60)
VSYNC = config.get('display', 'vsync', True)

# Set up display
flags = pygame.OPENGL | pygame.DOUBLEBUF
if config.get('display', 'fullscreen', False):
    flags |= pygame.FULLSCREEN

pygame.display.set_mode((W, H), flags)
pygame.display.set_caption("3D Hand-Controlled Driving")

# OpenGL setup
glEnable(GL_DEPTH_TEST)
glEnable(GL_LIGHTING)
glEnable(GL_LIGHT0)
glEnable(GL_COLOR_MATERIAL)
glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

# Set up the perspective
glMatrixMode(GL_PROJECTION)
gluPerspective(45, (W/H), 0.1, 50.0)
glMatrixMode(GL_MODELVIEW)

# Initialize game components
vehicle_manager = VehicleManager()
camera = Camera()
gesture_controller = GestureController()
game_mode_manager = GameModeManager()
game_ui = GameUI(W, H)
game_hud = GameHUD(W, H)
rewards_manager = RewardsManager()
analytics_manager = AnalyticsManager()
sound_manager = SoundManager()

# Set audio volumes from config
sound_manager.set_volume(config.get('audio', 'sfx_volume', 0.8))
sound_manager.set_music_volume(config.get('audio', 'music_volume', 0.5))

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.log_error(Exception("Could not open camera"), "Camera initialization")
    sys.exit()

# Start analytics session
analytics_manager.start_session()
logger.log_game_event("session_start", {"timestamp": pygame.time.get_ticks()})

# Play background music
sound_manager.play_music('background_music.mp3')

def get_vehicle_state(vehicle, control_state):
    return {
        'speed': vehicle.speed,
        'gear': vehicle.gear,
        'fuel': vehicle.fuel,
        'time_elapsed': vehicle.time_elapsed,
        'current_gesture': control_state['current_gesture'],
        'next_turn': vehicle.next_turn,
        'left_indicator': control_state['left_indicator'],
        'right_indicator': control_state['right_indicator'],
        'brake_power': control_state['brake'],
        'acceleration_power': control_state['acceleration'],
        'nitro_available': vehicle.nitro_available,
        'collision_warning': vehicle.collision_warning,
        'distance_remaining': vehicle.distance_remaining,
        'distance_completed': vehicle.distance_completed
    }

def get_road_data(vehicle):
    return {
        'segments': vehicle.road_segments,
        'player_position': (vehicle.position[0], vehicle.position[2])
    }

def handle_ui_events():
    """Handle UI events and transitions"""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            logger.log_game_event("game_quit", {"timestamp": pygame.time.get_ticks()})
            cap.release()
            pygame.quit()
            sys.exit()
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if game_ui.current_screen == Screen.DRIVE_MODE:
                    game_ui.set_screen(Screen.HOME)
                    sound_manager.play_sound('menu_select')
                    logger.log_game_event("screen_change", {"from": "DRIVE_MODE", "to": "HOME"})
                else:
                    game_ui.set_screen(Screen.DRIVE_MODE)
                    sound_manager.play_sound('menu_select')
                    logger.log_game_event("screen_change", {"from": "HOME", "to": "DRIVE_MODE"})
            
            # Handle UI button clicks
            if event.type == pygame.MOUSEBUTTONDOWN:
                if game_ui.handle_click(event.pos):
                    sound_manager.play_sound('button_click')
                    if game_ui.current_screen == Screen.HOME:
                        if game_ui.clicked_button == "play":
                            game_ui.set_screen(Screen.DRIVE_MODE)
                            sound_manager.play_sound('game_start')
                            logger.log_game_event("game_start", {"mode": "DRIVE_MODE"})
                        elif game_ui.clicked_button == "garage":
                            game_ui.set_screen(Screen.GARAGE)
                            logger.log_game_event("screen_change", {"to": "GARAGE"})
                        elif game_ui.clicked_button == "store":
                            game_ui.set_screen(Screen.STORE)
                            logger.log_game_event("screen_change", {"to": "STORE"})
                        elif game_ui.clicked_button == "settings":
                            game_ui.set_screen(Screen.SETTINGS)
                            logger.log_game_event("screen_change", {"to": "SETTINGS"})
                        elif game_ui.clicked_button == "profile":
                            game_ui.set_screen(Screen.PROFILE)
                            logger.log_game_event("screen_change", {"to": "PROFILE"})
    
    return True

def main():
    clock = pygame.time.Clock()
    in_game = False
    last_time = pygame.time.get_ticks()
    last_save_time = last_time
    save_interval = config.get('gameplay', 'save_interval', 300) * 1000  # Convert to milliseconds
    
    while True:
        current_time = pygame.time.get_ticks()
        dt = (current_time - last_time) / 1000.0  # Convert to seconds
        last_time = current_time
        
        # Handle UI events
        running = handle_ui_events()
        if not running:
            break
        
        if not in_game:
            # Check if starting a game mode
            if game_ui.current_screen == Screen.DRIVE_MODE and game_ui.selected_mode:
                in_game = True
                game_mode_manager.set_mode(game_ui.selected_mode)
                analytics_manager.log_game_mode_start(game_ui.selected_mode)
                game_ui.selected_mode = None
                sound_manager.play_sound('game_start')
                logger.log_game_event("mode_start", {"mode": game_ui.selected_mode})
        else:
            # Capture frame from camera
            ret, frame = cap.read()
            if not ret:
                logger.log_error(Exception("Failed to capture frame"), "Camera")
                continue
                
            # Process hand gestures
            frame = gesture_controller.process_frame(frame)
            control_state = gesture_controller.get_control_state()
            
            # Get selected vehicle
            selected_vehicle = vehicle_manager.get_selected_vehicle()
            if selected_vehicle:
                vehicle_stats = selected_vehicle.get_current_stats()
                
                # Update vehicle based on gesture controls and stats
                sensitivity = config.get('controls', 'steering_sensitivity', 1.0)
                vehicle_stats['handling'] *= control_state['steering'] * sensitivity
                
                if control_state['acceleration'] > 0:
                    accel_sensitivity = config.get('controls', 'acceleration_sensitivity', 1.0)
                    vehicle_stats['acceleration'] *= control_state['acceleration'] * accel_sensitivity
                    sound_manager.play_sound('engine')
                
                if control_state['brake'] > 0:
                    brake_sensitivity = config.get('controls', 'brake_sensitivity', 1.0)
                    vehicle_stats['braking'] *= control_state['brake'] * brake_sensitivity
                
                # Update vehicle state
                if selected_vehicle.shift_gear(control_state['gear']):
                    sound_manager.play_sound('gear_shift')
                    logger.log_vehicle_event("gear_shift", selected_vehicle.id, {"gear": control_state['gear']})
                
                if selected_vehicle.left_indicator != control_state['left_indicator']:
                    selected_vehicle.left_indicator = control_state['left_indicator']
                    sound_manager.play_sound('indicator')
                    logger.log_vehicle_event("indicator", selected_vehicle.id, {"side": "left", "state": control_state['left_indicator']})
                
                if selected_vehicle.right_indicator != control_state['right_indicator']:
                    selected_vehicle.right_indicator = control_state['right_indicator']
                    sound_manager.play_sound('indicator')
                    logger.log_vehicle_event("indicator", selected_vehicle.id, {"side": "right", "state": control_state['right_indicator']})
                
                if selected_vehicle.headlights_state != control_state['headlights']:
                    selected_vehicle.headlights_state = control_state['headlights']
                    logger.log_vehicle_event("headlights", selected_vehicle.id, {"state": control_state['headlights']})
                
                if selected_vehicle.horn != control_state['horn']:
                    selected_vehicle.horn = control_state['horn']
                    if control_state['horn']:
                        sound_manager.play_sound('horn')
                        logger.log_vehicle_event("horn", selected_vehicle.id, {"state": True})
                
                if selected_vehicle.wipers_active != control_state['wipers']:
                    selected_vehicle.wipers_active = control_state['wipers']
                    logger.log_vehicle_event("wipers", selected_vehicle.id, {"state": control_state['wipers']})
                
                if selected_vehicle.handbrake_active != control_state['handbrake']:
                    selected_vehicle.handbrake_active = control_state['handbrake']
                    logger.log_vehicle_event("handbrake", selected_vehicle.id, {"state": control_state['handbrake']})
                
                # Update HUD
                vehicle_state = get_vehicle_state(selected_vehicle, control_state)
                road_data = get_road_data(selected_vehicle)
                game_hud.update(dt, vehicle_state, game_mode_manager.get_mode_status())
            
            # Update game mode
            game_mode_manager.update(selected_vehicle, dt)
            
            # Update camera
            camera.update(selected_vehicle)
            
            # Update rewards
            if game_mode_manager.check_achievement():
                achievement_id = game_mode_manager.get_achievement_id()
                reward = rewards_manager.achievements[achievement_id].reward
                rewards_manager.update_achievement_progress(achievement_id, 1)
                analytics_manager.log_achievement_unlock(achievement_id, reward)
                sound_manager.play_sound('achievement')
                logger.log_achievement(achievement_id, reward)
            
            # Log performance metrics
            fps = clock.get_fps()
            frame_time = 1000 / fps if fps > 0 else 0
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Convert to MB
            analytics_manager.log_performance(fps, frame_time, memory_usage)
            logger.log_performance(fps, frame_time, memory_usage)
            
            # Auto-save if enabled
            if config.get('gameplay', 'auto_save', True):
                if current_time - last_save_time >= save_interval:
                    # Save game state
                    game_mode_manager.save_state()
                    rewards_manager.save_state()
                    last_save_time = current_time
                    logger.log_game_event("auto_save", {"timestamp": current_time})
            
            # Check for game mode completion
            if game_mode_manager.is_complete():
                score = game_mode_manager.get_score()
                distance = game_mode_manager.get_distance()
                analytics_manager.log_game_mode_end(
                    game_mode_manager.get_current_mode(),
                    score,
                    distance
                )
                sound_manager.play_sound('game_over')
                logger.log_game_mode(game_mode_manager.get_current_mode(), "complete", score)
                in_game = False
                game_ui.set_screen(Screen.HOME)
            
            # Clear screen and depth buffer
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Apply camera transformation
            camera.apply()
            
            # Draw scene
            if selected_vehicle:
                selected_vehicle.draw()
            
            # Draw HUD
            screen = pygame.display.get_surface()
            game_hud.draw(screen, road_data)
            
            # Display camera feed in a separate window
            cv2.imshow('Hand Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Update UI
        game_ui.update()
        
        # Draw UI
        screen = pygame.display.get_surface()
        screen.fill((0, 0, 0))
        game_ui.draw(screen)
        
        pygame.display.flip()
        clock.tick(FPS_LIMIT)
    
    # End analytics session
    analytics_manager.end_session()
    sound_manager.cleanup()
    logger.log_game_event("session_end", {"timestamp": pygame.time.get_ticks()})
    pygame.quit()

if __name__ == "__main__":
    main() 