import pygame
import random
import math
from typing import List, Tuple, Dict
from vehicle import Vehicle, GearMode

class TrafficLight:
    def __init__(self, x: float, y: float, direction: float):
        self.x = x
        self.y = y
        self.direction = direction  # Angle in degrees
        self.state = 'red'  # 'red', 'yellow', 'green'
        self.timer = 0
        self.durations = {
            'red': 150,    # 5 seconds at 30 FPS
            'yellow': 60,  # 2 seconds
            'green': 180   # 6 seconds
        }

    def update(self):
        """Update traffic light state"""
        self.timer += 1
        if self.timer >= self.durations[self.state]:
            self.timer = 0
            if self.state == 'red':
                self.state = 'green'
            elif self.state == 'yellow':
                self.state = 'red'
            elif self.state == 'green':
                self.state = 'yellow'

    def draw(self, screen: pygame.Surface):
        """Draw traffic light"""
        colors = {
            'red': (255, 0, 0),
            'yellow': (255, 255, 0),
            'green': (0, 255, 0)
        }
        pygame.draw.rect(screen, (100, 100, 100), (self.x - 5, self.y - 30, 10, 60))
        pygame.draw.circle(screen, colors[self.state], (int(self.x), int(self.y)), 8)

class AIDriver:
    def __init__(self, vehicle: Vehicle):
        self.vehicle = vehicle
        self.target_speed = random.uniform(30, 60)
        self.safe_distance = 50
        self.reaction_time = random.uniform(0.5, 1.5)
        self.aggression = random.uniform(0.3, 0.8)
        self.path: List[Tuple[float, float]] = []
        self.current_waypoint = 0
        self.state = 'driving'  # 'driving', 'stopping', 'waiting', 'turning'
        self.stop_timer = 0

    def update(self, dt: float, traffic_lights: List[TrafficLight], other_vehicles: List[Vehicle]):
        """Update AI driver behavior"""
        if not self.path:
            return

        # Get current target waypoint
        target = self.path[self.current_waypoint]
        
        # Calculate distance and angle to target
        dx = target[0] - self.vehicle.x
        dy = target[1] - self.vehicle.y
        distance = math.sqrt(dx*dx + dy*dy)
        target_angle = math.degrees(math.atan2(dy, dx))

        # Check for traffic lights
        nearest_light = self._get_nearest_traffic_light(traffic_lights)
        if nearest_light and self._should_stop_for_light(nearest_light):
            self.state = 'stopping'
            self.vehicle.brake_force = self.vehicle.brake_power
            return

        # Check for nearby vehicles
        nearest_vehicle = self._get_nearest_vehicle(other_vehicles)
        if nearest_vehicle and self._should_brake_for_vehicle(nearest_vehicle):
            self.state = 'stopping'
            self.vehicle.brake_force = self.vehicle.brake_power * 0.5
            return

        # Normal driving behavior
        self.state = 'driving'
        
        # Adjust steering
        angle_diff = (target_angle - self.vehicle.angle + 180) % 360 - 180
        if abs(angle_diff) > 5:
            self.vehicle.steering_angle = max(-30, min(30, angle_diff * 0.5))
        else:
            self.vehicle.steering_angle = 0

        # Adjust speed
        if self.vehicle.speed < self.target_speed:
            self.vehicle.engine_force = self.vehicle.acceleration * self.aggression
            self.vehicle.brake_force = 0
        else:
            self.vehicle.engine_force = 0
            self.vehicle.brake_force = self.vehicle.brake_power * 0.1

        # Check if waypoint reached
        if distance < 20:
            self.current_waypoint = (self.current_waypoint + 1) % len(self.path)

    def _get_nearest_traffic_light(self, traffic_lights: List[TrafficLight]) -> TrafficLight:
        """Find the nearest traffic light"""
        nearest = None
        min_dist = float('inf')
        for light in traffic_lights:
            dx = light.x - self.vehicle.x
            dy = light.y - self.vehicle.y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_dist and dist < 100:  # Only consider lights within 100 pixels
                min_dist = dist
                nearest = light
        return nearest

    def _should_stop_for_light(self, light: TrafficLight) -> bool:
        """Determine if vehicle should stop for traffic light"""
        if light.state == 'red' or light.state == 'yellow':
            dx = light.x - self.vehicle.x
            dy = light.y - self.vehicle.y
            angle_to_light = math.degrees(math.atan2(dy, dx))
            angle_diff = abs((angle_to_light - light.direction + 180) % 360 - 180)
            return angle_diff < 45
        return False

    def _get_nearest_vehicle(self, vehicles: List[Vehicle]) -> Vehicle:
        """Find the nearest vehicle ahead"""
        nearest = None
        min_dist = float('inf')
        for vehicle in vehicles:
            if vehicle != self.vehicle:
                dx = vehicle.x - self.vehicle.x
                dy = vehicle.y - self.vehicle.y
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_dist and dist < 100:  # Only consider vehicles within 100 pixels
                    # Check if vehicle is ahead
                    angle_to_vehicle = math.degrees(math.atan2(dy, dx))
                    angle_diff = abs((angle_to_vehicle - self.vehicle.angle + 180) % 360 - 180)
                    if angle_diff < 45:
                        min_dist = dist
                        nearest = vehicle
        return nearest

    def _should_brake_for_vehicle(self, vehicle: Vehicle) -> bool:
        """Determine if should brake for vehicle ahead"""
        dx = vehicle.x - self.vehicle.x
        dy = vehicle.y - self.vehicle.y
        dist = math.sqrt(dx*dx + dy*dy)
        return dist < self.safe_distance

class TrafficSystem:
    def __init__(self, map_width: int, map_height: int):
        self.map_width = map_width
        self.map_height = map_height
        self.vehicles: List[Vehicle] = []
        self.ai_drivers: List[AIDriver] = []
        self.traffic_lights: List[TrafficLight] = []
        self.spawn_points: List[Tuple[float, float, float]] = []  # x, y, direction
        self.paths: List[List[Tuple[float, float]]] = []
        
    def add_spawn_point(self, x: float, y: float, direction: float):
        """Add a vehicle spawn point"""
        self.spawn_points.append((x, y, direction))

    def add_path(self, waypoints: List[Tuple[float, float]]):
        """Add a predefined path for AI vehicles"""
        self.paths.append(waypoints)

    def add_traffic_light(self, x: float, y: float, direction: float):
        """Add a traffic light"""
        self.traffic_lights.append(TrafficLight(x, y, direction))

    def spawn_vehicle(self):
        """Spawn a new AI-controlled vehicle"""
        if not self.spawn_points or not self.paths:
            return

        spawn_point = random.choice(self.spawn_points)
        vehicle = Vehicle(spawn_point[0], spawn_point[1], f"AI_Car_{len(self.vehicles)}")
        vehicle.angle = spawn_point[2]
        vehicle.engine_on = True
        vehicle.gear = GearMode.DRIVE

        ai_driver = AIDriver(vehicle)
        ai_driver.path = random.choice(self.paths)

        self.vehicles.append(vehicle)
        self.ai_drivers.append(ai_driver)

    def update(self, dt: float):
        """Update traffic system"""
        # Update traffic lights
        for light in self.traffic_lights:
            light.update()

        # Update AI vehicles
        for i, (vehicle, ai_driver) in enumerate(zip(self.vehicles, self.ai_drivers)):
            # Create list of other vehicles
            other_vehicles = self.vehicles[:i] + self.vehicles[i+1:]
            
            # Update AI behavior
            ai_driver.update(dt, self.traffic_lights, other_vehicles)
            
            # Update vehicle physics
            vehicle.update_physics(dt)

            # Remove vehicles that are off the map
            if (vehicle.x < -100 or vehicle.x > self.map_width + 100 or
                vehicle.y < -100 or vehicle.y > self.map_height + 100):
                self.vehicles.pop(i)
                self.ai_drivers.pop(i)

    def draw(self, screen: pygame.Surface):
        """Draw traffic system"""
        # Draw traffic lights
        for light in self.traffic_lights:
            light.draw(screen)

        # Draw vehicles
        for vehicle in self.vehicles:
            vehicle.draw(screen)
