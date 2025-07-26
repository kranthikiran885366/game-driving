import pygame
import math
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from OpenGL.GL import *
from OpenGL.GLU import *
from enum import Enum

class Gear(Enum):
    PARK = "P"
    NEUTRAL = "N"
    REVERSE = "R"
    DRIVE = "D"
    FIRST = "1"
    SECOND = "2"
    THIRD = "3"
    FOURTH = "4"
    FIFTH = "5"

@dataclass
class HUDState:
    speed: float = 0.0
    gear: Gear = Gear.PARK
    fuel: float = 100.0
    time_elapsed: float = 0.0
    mission_goals: Dict[str, bool] = None
    current_gesture: str = ""
    next_turn: Optional[str] = None
    left_indicator: bool = False
    right_indicator: bool = False
    brake_power: float = 0.0
    acceleration_power: float = 0.0
    nitro_available: bool = False
    collision_warning: bool = False
    distance_remaining: float = 0.0
    distance_completed: float = 0.0

class GameHUD:
    def __init__(self, screen_width: int, screen_height: int):
        self.width = screen_width
        self.height = screen_height
        self.state = HUDState()
        self.fonts = {
            'large': pygame.font.Font(None, 48),
            'medium': pygame.font.Font(None, 36),
            'small': pygame.font.Font(None, 24)
        }
        self.colors = {
            'white': (255, 255, 255),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'yellow': (255, 255, 0),
            'blue': (0, 0, 255)
        }
        self.minimap_size = (200, 200)
        self.minimap_surface = pygame.Surface(self.minimap_size)
        self.collision_warning_time = 0
        self.collision_warning_duration = 1.0  # seconds
        
    def update(self, dt: float, vehicle_state: Dict, mission_state: Optional[Dict] = None):
        # Update HUD state with vehicle data
        self.state.speed = vehicle_state.get('speed', 0.0)
        self.state.gear = vehicle_state.get('gear', Gear.PARK)
        self.state.fuel = vehicle_state.get('fuel', 100.0)
        self.state.time_elapsed = vehicle_state.get('time_elapsed', 0.0)
        self.state.current_gesture = vehicle_state.get('current_gesture', "")
        self.state.next_turn = vehicle_state.get('next_turn')
        self.state.left_indicator = vehicle_state.get('left_indicator', False)
        self.state.right_indicator = vehicle_state.get('right_indicator', False)
        self.state.brake_power = vehicle_state.get('brake_power', 0.0)
        self.state.acceleration_power = vehicle_state.get('acceleration_power', 0.0)
        self.state.nitro_available = vehicle_state.get('nitro_available', False)
        self.state.distance_remaining = vehicle_state.get('distance_remaining', 0.0)
        self.state.distance_completed = vehicle_state.get('distance_completed', 0.0)
        
        # Update collision warning
        if vehicle_state.get('collision_warning', False):
            self.collision_warning_time = self.collision_warning_duration
        self.collision_warning_time = max(0, self.collision_warning_time - dt)
        self.state.collision_warning = self.collision_warning_time > 0
        
        # Update mission goals if in mission mode
        if mission_state:
            self.state.mission_goals = mission_state.get('goals', {})
    
    def draw_speedometer(self, screen: pygame.Surface, pos: Tuple[int, int]):
        # Draw 3D speedometer dial and needle using OpenGL
        glPushMatrix()
        glTranslatef(pos[0], pos[1], 0)
        # Draw dial (circle)
        glColor3f(0.2, 0.2, 0.2)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(0, 0)
        for angle in range(0, 361, 5):
            rad = angle * 3.14159 / 180
            glVertex2f(60 * np.cos(rad), 60 * np.sin(rad))
        glEnd()
        # Draw needle
        speed_angle = -120 + (self.state.speed / 240.0) * 240  # Map 0-240 km/h to -120 to +120 deg
        glRotatef(speed_angle, 0, 0, 1)
        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_QUADS)
        glVertex2f(-3, 0)
        glVertex2f(3, 0)
        glVertex2f(3, 50)
        glVertex2f(-3, 50)
        glEnd()
        glPopMatrix()
        # Overlay with Pygame for text
        speed_text = f"{int(self.state.speed)}"
        speed_surface = self.fonts['large'].render(speed_text, True, self.colors['white'])
        speed_rect = speed_surface.get_rect(center=pos)
        screen.blit(speed_surface, speed_rect)
        unit_surface = self.fonts['small'].render("KM/H", True, self.colors['white'])
        unit_rect = unit_surface.get_rect(center=(pos[0], pos[1] + 20))
        screen.blit(unit_surface, unit_rect)
    
    def draw_fuel_meter(self, screen: pygame.Surface, pos: Tuple[int, int]):
        # Draw 3D fuel gauge using OpenGL
        glPushMatrix()
        glTranslatef(pos[0], pos[1], 0)
        glColor3f(0.1, 0.1, 0.1)
        glBegin(GL_LINE_LOOP)
        for angle in range(0, 181, 5):
            rad = angle * 3.14159 / 180
            glVertex2f(40 * np.cos(rad), 40 * np.sin(rad))
        glEnd()
        # Draw fuel needle
        fuel_angle = -90 + (self.state.fuel / 100.0) * 180
        glRotatef(fuel_angle, 0, 0, 1)
        glColor3f(0.0, 1.0, 0.0) if self.state.fuel > 20 else glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_QUADS)
        glVertex2f(-2, 0)
        glVertex2f(2, 0)
        glVertex2f(2, 30)
        glVertex2f(-2, 30)
        glEnd()
        glPopMatrix()
        # Overlay with Pygame for text
        fuel_text = f"FUEL: {int(self.state.fuel)}%"
        fuel_surface = self.fonts['small'].render(fuel_text, True, self.colors['white'])
        screen.blit(fuel_surface, (pos[0] - 48, pos[1] - 30))
    
    def draw_gear_indicator(self, screen: pygame.Surface, pos: Tuple[int, int]):
        # Draw 3D gear indicator using OpenGL
        glPushMatrix()
        glTranslatef(pos[0], pos[1], 0)
        glColor3f(0.3, 0.3, 0.3)
        glBegin(GL_QUADS)
        glVertex2f(-30, -30)
        glVertex2f(30, -30)
        glVertex2f(30, 30)
        glVertex2f(-30, 30)
        glEnd()
        glPopMatrix()
        # Overlay with Pygame for text
        gear_surface = self.fonts['large'].render(self.state.gear.value, True, self.colors['white'])
        gear_rect = gear_surface.get_rect(center=pos)
        screen.blit(gear_surface, gear_rect)
    
    def draw_minimap(self, screen: pygame.Surface, pos: Tuple[int, int], road_data: Dict):
        # Placeholder for 3D minimap (could be expanded with OpenGL rendering)
        pygame.draw.rect(screen, (30, 30, 30), (pos[0], pos[1], 200, 200), border_radius=20)
        # Clear minimap surface
        self.minimap_surface.fill((0, 0, 0))
        
        # Draw road layout
        for segment in road_data.get('segments', []):
            start = segment['start']
            end = segment['end']
            # Scale coordinates to minimap size
            start_scaled = (
                int(start[0] * self.minimap_size[0]),
                int(start[1] * self.minimap_size[1])
            )
            end_scaled = (
                int(end[0] * self.minimap_size[0]),
                int(end[1] * self.minimap_size[1])
            )
            pygame.draw.line(self.minimap_surface, (100, 100, 100), start_scaled, end_scaled, 2)
        
        # Draw player position
        player_pos = road_data.get('player_position', (0.5, 0.5))
        player_scaled = (
            int(player_pos[0] * self.minimap_size[0]),
            int(player_pos[1] * self.minimap_size[1])
        )
        pygame.draw.circle(self.minimap_surface, self.colors['red'], player_scaled, 3)
        
        # Draw minimap on screen
        screen.blit(self.minimap_surface, pos)
    
    def draw_timer(self, screen: pygame.Surface, pos: Tuple[int, int]):
        minutes = int(self.state.time_elapsed // 60)
        seconds = int(self.state.time_elapsed % 60)
        time_text = f"{minutes:02d}:{seconds:02d}"
        time_surface = self.fonts['medium'].render(time_text, True, self.colors['white'])
        screen.blit(time_surface, pos)
    
    def draw_mission_objectives(self, screen: pygame.Surface, pos: Tuple[int, int]):
        # Animated overlay for mission objectives
        if not self.state.mission_goals:
            return
        y_offset = 0
        for goal, completed in self.state.mission_goals.items():
            color = self.colors['green'] if completed else self.colors['white']
            goal_surface = self.fonts['small'].render(goal, True, color)
            screen.blit(goal_surface, (pos[0], pos[1] + y_offset))
            y_offset += 25
        # Animated feedback for completion
        if all(self.state.mission_goals.values()):
            complete_surface = self.fonts['medium'].render("MISSION COMPLETE!", True, self.colors['yellow'])
            screen.blit(complete_surface, (pos[0], pos[1] + y_offset + 10))
    
    def draw_gesture_status(self, screen: pygame.Surface, pos: Tuple[int, int]):
        if not self.state.current_gesture:
            return
        # Draw semi-transparent overlay for gesture feedback
        overlay = pygame.Surface((200, 50), pygame.SRCALPHA)
        overlay.fill((50, 200, 255, 120))
        gesture_surface = self.fonts['medium'].render(
            f"Gesture: {self.state.current_gesture}",
            True,
            self.colors['white']
        )
        overlay.blit(gesture_surface, (10, 10))
        screen.blit(overlay, pos)
    
    def draw_directional_arrows(self, screen: pygame.Surface, pos: Tuple[int, int]):
        if not self.state.next_turn:
            return
        
        # Draw arrow based on next turn direction
        arrow_points = []
        if self.state.next_turn == "left":
            arrow_points = [
                (pos[0] - 20, pos[1]),
                (pos[0] + 20, pos[1] - 20),
                (pos[0] + 20, pos[1] + 20)
            ]
        elif self.state.next_turn == "right":
            arrow_points = [
                (pos[0] + 20, pos[1]),
                (pos[0] - 20, pos[1] - 20),
                (pos[0] - 20, pos[1] + 20)
            ]
        
        if arrow_points:
            pygame.draw.polygon(screen, self.colors['yellow'], arrow_points)
    
    def draw_indicator_icons(self, screen: pygame.Surface, pos: Tuple[int, int]):
        # Draw left indicator
        if self.state.left_indicator:
            left_points = [
                (pos[0] - 40, pos[1]),
                (pos[0] - 20, pos[1] - 10),
                (pos[0] - 20, pos[1] + 10)
            ]
            pygame.draw.polygon(screen, self.colors['yellow'], left_points)
        
        # Draw right indicator
        if self.state.right_indicator:
            right_points = [
                (pos[0] + 40, pos[1]),
                (pos[0] + 20, pos[1] - 10),
                (pos[0] + 20, pos[1] + 10)
            ]
            pygame.draw.polygon(screen, self.colors['yellow'], right_points)
    
    def draw_power_bars(self, screen: pygame.Surface, pos: Tuple[int, int]):
        # Draw acceleration bar
        accel_width = int(100 * self.state.acceleration_power)
        pygame.draw.rect(screen, (50, 50, 50), (pos[0], pos[1], 100, 10))
        pygame.draw.rect(screen, self.colors['green'], (pos[0], pos[1], accel_width, 10))
        
        # Draw brake bar
        brake_width = int(100 * self.state.brake_power)
        pygame.draw.rect(screen, (50, 50, 50), (pos[0], pos[1] + 20, 100, 10))
        pygame.draw.rect(screen, self.colors['red'], (pos[0], pos[1] + 20, brake_width, 10))
    
    def draw_nitro_icon(self, screen: pygame.Surface, pos: Tuple[int, int]):
        color = self.colors['blue'] if self.state.nitro_available else (50, 50, 50)
        pygame.draw.circle(screen, color, pos, 15)
        nitro_surface = self.fonts['small'].render("N", True, self.colors['white'])
        nitro_rect = nitro_surface.get_rect(center=pos)
        screen.blit(nitro_surface, nitro_rect)
    
    def draw_collision_warning(self, screen: pygame.Surface):
        if self.state.collision_warning:
            # Create semi-transparent red overlay
            overlay = pygame.Surface((self.width, self.height))
            overlay.fill(self.colors['red'])
            overlay.set_alpha(128)
            screen.blit(overlay, (0, 0))
    
    def draw_distance_tracker(self, screen: pygame.Surface, pos: Tuple[int, int]):
        # Draw completed distance
        completed_text = f"Completed: {self.state.distance_completed:.1f} km"
        completed_surface = self.fonts['small'].render(completed_text, True, self.colors['white'])
        screen.blit(completed_surface, pos)
        
        # Draw remaining distance
        remaining_text = f"Remaining: {self.state.distance_remaining:.1f} km"
        remaining_surface = self.fonts['small'].render(remaining_text, True, self.colors['white'])
        screen.blit(remaining_surface, (pos[0], pos[1] + 25))
    
    def draw(self, screen: pygame.Surface, road_data: Dict):
        # Draw all HUD elements
        self.draw_speedometer(screen, (100, self.height - 100))
        self.draw_gear_indicator(screen, (200, self.height - 100))
        self.draw_fuel_meter(screen, (300, self.height - 100))
        self.draw_minimap(screen, (self.width - 250, 50), road_data)
        self.draw_timer(screen, (self.width - 150, 20))
        self.draw_mission_objectives(screen, (20, 20))
        self.draw_gesture_status(screen, (self.width - 200, self.height - 50))
        self.draw_directional_arrows(screen, (self.width // 2, 50))
        self.draw_indicator_icons(screen, (self.width // 2, 100))
        self.draw_power_bars(screen, (self.width - 150, self.height - 150))
        self.draw_nitro_icon(screen, (self.width - 50, self.height - 50))
        self.draw_collision_warning(screen)
        self.draw_distance_tracker(screen, (20, self.height - 50)) 