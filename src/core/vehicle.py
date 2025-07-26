import pygame
import math
from enum import Enum
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from src.core.game_config import VEHICLES

class GearMode(Enum):
    PARK = 'P'
    REVERSE = 'R'
    NEUTRAL = 'N'
    DRIVE = 'D'

class Vehicle:
    def __init__(self, vehicle_type='sedan'):
        self.type = vehicle_type
        self.specs = VEHICLES[vehicle_type]
        
        # Position and movement
        self.position = np.array([0.0, 0.0, 0.0])
        self.rotation = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0
        self.steering = 0.0
        
        # Vehicle state
        self.gear = 'N'  # P, R, N, 1-6
        self.fuel = self.specs['fuel_capacity']
        self.damage = 0.0
        self.lights = False
        self.horn = False
        self.left_indicator = False
        self.right_indicator = False
        
        # Physics
        self.mass = 1000.0
        self.drag_coefficient = 0.3
        self.wheel_base = 2.5
        
        # Basic properties
        self.model_name = vehicle_type
        self.angle = 0
        self.speed = 0
        self.manual_mode = False
        self.current_gear = 1
        self.clutch_engaged = False

        # Performance specs
        self.max_speed = 200  # km/h
        self.brake_power = 0.8
        self.handling = 0.7
        self.max_health = 100
        self.current_health = 100

        # Physics properties
        self.engine_force = 0
        self.brake_force = 0
        self.steering_angle = 0
        self.max_steering_angle = 30  # degrees
        self.turn_radius = 0

        # Vehicle state
        self.engine_on = False
        self.handbrake_active = False
        self.headlights_state = 0  # 0: off, 1: low beam, 2: high beam
        self.indicators = {'left': False, 'right': False}
        self.wipers_active = False

        # Maintenance
        self.fuel_level = 100
        self.fuel_capacity = 100
        self.fuel_consumption_rate = 0.01
        self.engine_temperature = 0
        self.oil_level = 100
        self.tire_wear = [100, 100, 100, 100]  # FL, FR, RL, RR

        # Damage system
        self.damage_areas = {
            'front': 100,
            'rear': 100,
            'left': 100,
            'right': 100,
            'engine': 100,
            'transmission': 100,
            'brakes': 100,
            'steering': 100
        }

        # Visual properties
        self.color = (50, 50, 200)
        self.width = 50
        self.height = 30
        self.image = self._create_vehicle_image()
        self.rect = self.image.get_rect(center=(self.position[0], self.position[2]))

        self.tire_grip = 1.0
        self.weight_transfer = 0.0
        self.accel_smooth = 0.0
        self.brake_smooth = 0.0

    def _create_vehicle_image(self):
        """Create the vehicle's visual representation"""
        image = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.polygon(image, self.color, [
            (0, self.height/2),
            (self.width/4, 0),
            (self.width*3/4, 0),
            (self.width, self.height/2),
            (self.width*3/4, self.height),
            (self.width/4, self.height)
        ])
        return image

    def update(self, dt):
        # Smooth acceleration and braking
        target_accel = self.acceleration
        self.accel_smooth += (target_accel - self.accel_smooth) * 0.2
        target_brake = 1.0 if self.brake_force > 0 else 0.0
        self.brake_smooth += (target_brake - self.brake_smooth) * 0.2
        # Weight transfer (simple model)
        self.weight_transfer = 0.1 * self.accel_smooth - 0.1 * self.brake_smooth
        # Update position based on velocity and rotation
        self.position[0] += math.sin(math.radians(self.rotation)) * self.velocity * dt
        self.position[2] += math.cos(math.radians(self.rotation)) * self.velocity * dt
        # Update rotation based on steering and grip
        effective_steering = self.steering * self.tire_grip
        self.rotation += effective_steering * self.velocity * dt
        # Apply drag
        self.velocity *= (1.0 - self.drag_coefficient * dt)
        # Update fuel consumption
        self.fuel -= self.specs['fuel_consumption'] * abs(self.velocity) * dt
        # Collision and damage (simple logic)
        if self.velocity > self.max_speed * 0.8 and self.damage < 100:
            self.apply_damage((self.velocity - self.max_speed * 0.8) * 0.1)
            
    def accelerate(self, amount):
        if self.fuel > 0:
            self.acceleration = amount * self.specs['acceleration']
            self.velocity += self.acceleration
            
    def brake(self, amount):
        self.velocity -= amount * self.specs['braking']
        if self.velocity < 0:
            self.velocity = 0
            
    def steer(self, amount):
        self.steering = amount * self.specs['handling']
        
    def shift_gear(self, gear):
        valid_gears = ['P', 'R', 'N', '1', '2', '3', '4', '5', '6']
        if gear in valid_gears:
            self.gear = gear
            if gear == 'R':
                self.velocity = -abs(self.velocity)
            elif gear in ['1', '2', '3', '4', '5', '6', 'D']:
                self.velocity = abs(self.velocity)
            # Add realistic gear shifting delay
            import time
            time.sleep(0.1)
                
    def toggle_lights(self):
        self.lights = not self.lights
        
    def toggle_horn(self):
        self.horn = not self.horn
        
    def toggle_indicator(self, side):
        if side == 'left':
            self.left_indicator = not self.left_indicator
            self.right_indicator = False
        elif side == 'right':
            self.right_indicator = not self.right_indicator
            self.left_indicator = False
            
    def apply_damage(self, amount):
        self.damage += amount
        if self.damage > 100:
            self.damage = 100
            
    def refuel(self, amount):
        self.fuel = min(self.fuel + amount, self.specs['fuel_capacity'])
        
    def draw(self):
        glPushMatrix()
        glTranslatef(self.position[0], self.position[1], self.position[2])
        glRotatef(self.rotation, 0, 1, 0)
        
        # Draw car body
        glColor3f(0.8, 0.2, 0.2)  # Red color
        glPushMatrix()
        glScalef(1.0, 0.5, 2.0)
        glutSolidCube(1.0)
        glPopMatrix()
        
        # Draw wheels
        glColor3f(0.1, 0.1, 0.1)
        wheel_positions = [
            (-0.6, -0.3, -0.8),
            (0.6, -0.3, -0.8),
            (-0.6, -0.3, 0.8),
            (0.6, -0.3, 0.8)
        ]
        for pos in wheel_positions:
            glPushMatrix()
            glTranslatef(*pos)
            glRotatef(90, 0, 1, 0)
            glutSolidTorus(0.1, 0.2, 16, 16)
            glPopMatrix()
            
        # Draw lights if on
        if self.lights:
            glColor3f(1.0, 1.0, 0.8)
            glPushMatrix()
            glTranslatef(0, 0.3, 1.0)
            glScalef(0.8, 0.2, 0.1)
            glutSolidCube(1.0)
            glPopMatrix()
            
        # Draw indicators if on
        if self.left_indicator:
            glColor3f(1.0, 0.5, 0.0)
            glPushMatrix()
            glTranslatef(-0.6, 0.3, 0.8)
            glScalef(0.1, 0.2, 0.3)
            glutSolidCube(1.0)
            glPopMatrix()
            
        if self.right_indicator:
            glColor3f(1.0, 0.5, 0.0)
            glPushMatrix()
            glTranslatef(0.6, 0.3, 0.8)
            glScalef(0.1, 0.2, 0.3)
            glutSolidCube(1.0)
            glPopMatrix()
            
        glPopMatrix()
        
    def get_status(self):
        return {
            'position': self.position.tolist(),
            'rotation': self.rotation,
            'velocity': self.velocity,
            'gear': self.gear,
            'fuel': self.fuel,
            'damage': self.damage,
            'lights': self.lights,
            'horn': self.horn,
            'left_indicator': self.left_indicator,
            'right_indicator': self.right_indicator
        }

    def update_physics(self, dt: float):
        """Update vehicle physics"""
        if not self.engine_on and self.gear != GearMode.NEUTRAL:
            return

        # Apply engine force
        if self.gear == GearMode.DRIVE:
            acceleration = (self.engine_force / self.mass) * dt
            if not self.manual_mode or (self.manual_mode and not self.clutch_engaged):
                self.velocity += acceleration

        # Apply brake force
        if self.brake_force > 0 or self.handbrake_active:
            self.velocity = max(0, self.velocity - (self.brake_power * dt))

        # Apply drag
        drag = self.drag_coefficient * self.velocity * self.velocity * 0.5
        self.velocity = max(0, self.velocity - (drag * dt))

        # Update position
        if self.gear == GearMode.DRIVE:
            self.position[0] += math.sin(math.radians(self.rotation)) * self.velocity * dt
            self.position[2] += math.cos(math.radians(self.rotation)) * self.velocity * dt
        elif self.gear == GearMode.REVERSE:
            self.position[0] -= math.sin(math.radians(self.rotation)) * self.velocity * dt * 0.5
            self.position[2] -= math.cos(math.radians(self.rotation)) * self.velocity * dt * 0.5

        # Update steering
        if abs(self.steering_angle) > 0:
            turn_radius = self.wheel_base / math.sin(math.radians(self.steering_angle))
            angular_velocity = self.velocity / turn_radius if turn_radius != 0 else 0
            self.rotation += math.degrees(angular_velocity * dt)

        # Update fuel consumption
        if self.engine_on:
            self.fuel_level -= self.fuel_consumption_rate * abs(self.velocity) * dt
            self.fuel_level = max(0, self.fuel_level)
            if self.fuel_level == 0:
                self.engine_on = False

        # Update rect position
        self.rect.center = (self.position[0], self.position[2])

    def apply_damage(self, area: str, amount: float):
        """Apply damage to a specific area of the vehicle"""
        if area in self.damage_areas:
            self.damage_areas[area] = max(0, self.damage_areas[area] - amount)
            self._update_vehicle_performance()

    def _update_vehicle_performance(self):
        """Update vehicle performance based on damage"""
        # Engine performance
        engine_health = self.damage_areas['engine']
        self.acceleration = self.acceleration * (engine_health / 100)

        # Brake performance
        brake_health = self.damage_areas['brakes']
        self.brake_power = self.brake_power * (brake_health / 100)

        # Steering performance
        steering_health = self.damage_areas['steering']
        self.handling = self.handling * (steering_health / 100)

    def repair(self, area: str = None):
        """Repair vehicle damage"""
        if area:
            if area in self.damage_areas:
                self.damage_areas[area] = 100
        else:
            for area in self.damage_areas:
                self.damage_areas[area] = 100
        self._update_vehicle_performance()

    def toggle_engine(self):
        """Toggle engine on/off"""
        if self.fuel > 0:
            self.engine_on = not self.engine_on

    def set_manual_gear(self, gear_number: int):
        """Change manual transmission gear"""
        if self.manual_mode and self.clutch_engaged:
            self.current_gear = max(1, min(gear_number, 6))

    def toggle_headlights(self):
        """Toggle headlights state"""
        self.headlights_state = (self.headlights_state + 1) % 3

    def draw(self, screen: pygame.Surface):
        """Draw the vehicle on the screen"""
        # Draw vehicle body
        rotated_image = pygame.transform.rotate(self.image, -self.rotation)
        screen.blit(rotated_image, rotated_image.get_rect(center=(self.position[0], self.position[2])))

        # Draw headlights
        if self.headlights_state > 0:
            light_color = (255, 255, 200) if self.headlights_state == 1 else (255, 255, 255)
            front_left = (
                self.position[0] + math.sin(math.radians(self.rotation - 15)) * self.width/2,
                self.position[2] - math.cos(math.radians(self.rotation - 15)) * self.width/2
            )
            front_right = (
                self.position[0] + math.sin(math.radians(self.rotation + 15)) * self.width/2,
                self.position[2] - math.cos(math.radians(self.rotation + 15)) * self.width/2
            )
            pygame.draw.circle(screen, light_color, (int(front_left[0]), int(front_left[1])), 5)
            pygame.draw.circle(screen, light_color, (int(front_right[0]), int(front_right[1])), 5)

        # Draw indicators
        if self.left_indicator:
            indicator_pos = (
                self.position[0] + math.sin(math.radians(self.rotation - 90)) * self.width/4,
                self.position[2] - math.cos(math.radians(self.rotation - 90)) * self.width/4
            )
            pygame.draw.circle(screen, (255, 165, 0), (int(indicator_pos[0]), int(indicator_pos[1])), 3)

        if self.right_indicator:
            indicator_pos = (
                self.position[0] + math.sin(math.radians(self.rotation + 90)) * self.width/4,
                self.position[2] - math.cos(math.radians(self.rotation + 90)) * self.width/4
            )
            pygame.draw.circle(screen, (255, 165, 0), (int(indicator_pos[0]), int(indicator_pos[1])), 3)
