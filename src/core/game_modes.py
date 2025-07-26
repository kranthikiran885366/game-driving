import pygame
import math
import time
import numpy as np
from typing import List, Tuple, Dict, Optional
from enum import Enum
from src.core.vehicle import Vehicle
from abc import ABC, abstractmethod
from src.core.mission import Mission

class MissionStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class GameMode(Enum):
    FREE_DRIVE = 'free_drive'
    PARKING_CHALLENGE = 'parking_challenge'
    HIGHWAY_DRIVE = 'highway_drive'
    FUEL_SAVER = 'fuel_saver'
    TIME_TRIAL = 'time_trial'
    TRAFFIC_MODE = 'traffic_mode'
    NAVIGATION = 'navigation'
    MISSION_MODE = 'mission_mode'
    GESTURE_PRACTICE = 'gesture_practice'

class Mission(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.status = MissionStatus.NOT_STARTED
        self.score = 0
        self.time_limit = None
        self.time_elapsed = 0
        self.objectives: List[Dict] = []
        self.rewards = {
            'coins': 0,
            'exp': 0,
            'unlocks': []
        }

    @abstractmethod
    def update(self, dt: float, player_vehicle: Vehicle) -> None:
        pass

    @abstractmethod
    def draw(self, screen: pygame.Surface) -> None:
        pass

    def start(self) -> None:
        self.status = MissionStatus.IN_PROGRESS
        self.time_elapsed = 0

    def complete(self) -> None:
        self.status = MissionStatus.COMPLETED

    def fail(self) -> None:
        self.status = MissionStatus.FAILED

class TimeTrial(Mission):
    def __init__(self, checkpoints: List[Tuple[float, float]], target_time: float):
        super().__init__(
            "Time Trial",
            "Complete the course as quickly as possible!"
        )
        self.checkpoints = checkpoints
        self.current_checkpoint = 0
        self.target_time = target_time
        self.best_time = float('inf')
        self.time_limit = target_time * 1.5
        self.rewards = {
            'coins': 1000,
            'exp': 100,
            'unlocks': ['speed_upgrade']
        }

    def update(self, dt: float, player_vehicle: Vehicle) -> None:
        if self.status != MissionStatus.IN_PROGRESS:
            return

        self.time_elapsed += dt

        # Check if player reached checkpoint
        if self.current_checkpoint < len(self.checkpoints):
            checkpoint = self.checkpoints[self.current_checkpoint]
            distance = math.sqrt(
                (player_vehicle.x - checkpoint[0])**2 + 
                (player_vehicle.y - checkpoint[1])**2
            )
            
            if distance < 50:  # Checkpoint reached
                self.current_checkpoint += 1
                if self.current_checkpoint == len(self.checkpoints):
                    self.complete()
                    if self.time_elapsed < self.best_time:
                        self.best_time = self.time_elapsed
                        self.score = int((self.target_time - self.time_elapsed) * 100)

        # Check for time limit
        if self.time_elapsed > self.time_limit:
            self.fail()

    def draw(self, screen: pygame.Surface) -> None:
        # Draw checkpoints
        for i, checkpoint in enumerate(self.checkpoints):
            color = (0, 255, 0) if i < self.current_checkpoint else (255, 0, 0)
            pygame.draw.circle(screen, color, (int(checkpoint[0]), int(checkpoint[1])), 20, 2)

        # Draw time
        font = pygame.font.Font(None, 36)
        time_text = font.render(f"Time: {self.time_elapsed:.1f}/{self.time_limit:.1f}", True, (255, 255, 255))
        screen.blit(time_text, (10, 10))

class ParkingChallenge(Mission):
    def __init__(self, parking_spot: Tuple[float, float, float]):
        super().__init__(
            "Parking Challenge",
            "Park your vehicle in the designated spot!"
        )
        self.parking_spot = parking_spot  # x, y, angle
        self.time_limit = 120
        self.max_attempts = 3
        self.attempts = 0
        self.parked = False
        self.rewards = {
            'coins': 500,
            'exp': 50,
            'unlocks': ['parking_sensor']
        }

    def update(self, dt: float, player_vehicle: Vehicle) -> None:
        if self.status != MissionStatus.IN_PROGRESS:
            return

        self.time_elapsed += dt

        # Check if vehicle is in parking spot
        distance = math.sqrt(
            (player_vehicle.x - self.parking_spot[0])**2 + 
            (player_vehicle.y - self.parking_spot[1])**2
        )
        angle_diff = abs(player_vehicle.angle - self.parking_spot[2]) % 360

        if distance < 20 and angle_diff < 15 and abs(player_vehicle.speed) < 0.1:
            self.parked = True
            self.complete()
            self.score = int((self.time_limit - self.time_elapsed) * 10)
        
        # Check for collision or out of bounds
        if player_vehicle.current_health < 100:
            self.attempts += 1
            if self.attempts >= self.max_attempts:
                self.fail()
            else:
                # Reset vehicle position
                player_vehicle.x = self.parking_spot[0] - 100
                player_vehicle.y = self.parking_spot[1]
                player_vehicle.angle = 0
                player_vehicle.speed = 0
                player_vehicle.current_health = 100

        # Check for time limit
        if self.time_elapsed > self.time_limit:
            self.fail()

    def draw(self, screen: pygame.Surface) -> None:
        # Draw parking spot
        spot_points = [
            (self.parking_spot[0] - 30, self.parking_spot[1] - 15),
            (self.parking_spot[0] + 30, self.parking_spot[1] - 15),
            (self.parking_spot[0] + 30, self.parking_spot[1] + 15),
            (self.parking_spot[0] - 30, self.parking_spot[1] + 15)
        ]
        pygame.draw.lines(screen, (255, 255, 0), True, spot_points, 2)

        # Draw attempts and time
        font = pygame.font.Font(None, 36)
        attempts_text = font.render(f"Attempts: {self.attempts}/{self.max_attempts}", True, (255, 255, 255))
        time_text = font.render(f"Time: {self.time_elapsed:.1f}/{self.time_limit:.1f}", True, (255, 255, 255))
        screen.blit(attempts_text, (10, 10))
        screen.blit(time_text, (10, 50))

class DeliveryMission(Mission):
    def __init__(self, pickup: Tuple[float, float], dropoff: Tuple[float, float], cargo_fragility: float):
        super().__init__(
            "Delivery Mission",
            "Deliver the cargo safely to its destination!"
        )
        self.pickup = pickup
        self.dropoff = dropoff
        self.cargo_fragility = cargo_fragility  # 0-1, affects damage from impacts
        self.cargo_health = 100
        self.has_cargo = False
        self.time_limit = 180
        self.rewards = {
            'coins': 800,
            'exp': 75,
            'unlocks': ['cargo_capacity']
        }

    def update(self, dt: float, player_vehicle: Vehicle) -> None:
        if self.status != MissionStatus.IN_PROGRESS:
            return

        self.time_elapsed += dt

        # Check for pickup
        if not self.has_cargo:
            pickup_distance = math.sqrt(
                (player_vehicle.x - self.pickup[0])**2 + 
                (player_vehicle.y - self.pickup[1])**2
            )
            if pickup_distance < 30 and abs(player_vehicle.speed) < 0.1:
                self.has_cargo = True
        else:
            # Check for cargo damage
            if player_vehicle.speed > 50:  # Speed impact
                self.cargo_health -= (player_vehicle.speed - 50) * self.cargo_fragility * dt
            
            if player_vehicle.current_health < 100:  # Collision impact
                self.cargo_health -= (100 - player_vehicle.current_health) * self.cargo_fragility

            # Check for delivery
            dropoff_distance = math.sqrt(
                (player_vehicle.x - self.dropoff[0])**2 + 
                (player_vehicle.y - self.dropoff[1])**2
            )
            if dropoff_distance < 30 and abs(player_vehicle.speed) < 0.1:
                self.complete()
                self.score = int(self.cargo_health)

        # Fail conditions
        if self.cargo_health <= 0 or self.time_elapsed > self.time_limit:
            self.fail()

    def draw(self, screen: pygame.Surface) -> None:
        # Draw pickup and dropoff points
        if not self.has_cargo:
            pygame.draw.circle(screen, (0, 255, 0), (int(self.pickup[0]), int(self.pickup[1])), 20, 2)
        pygame.draw.circle(screen, (255, 0, 0), (int(self.dropoff[0]), int(self.dropoff[1])), 20, 2)

        # Draw cargo health and time
        font = pygame.font.Font(None, 36)
        if self.has_cargo:
            cargo_text = font.render(f"Cargo: {int(self.cargo_health)}%", True, (255, 255, 255))
            screen.blit(cargo_text, (10, 10))
        time_text = font.render(f"Time: {self.time_elapsed:.1f}/{self.time_limit:.1f}", True, (255, 255, 255))
        screen.blit(time_text, (10, 50))

class GameModeManager:
    def __init__(self):
        self.current_mode = GameMode.FREE_DRIVE
        self.mode_start_time = time.time()
        self.score = 0
        self.fuel = 100
        self.max_fuel = 100
        self.checkpoints = []
        self.traffic_density = 0
        self.navigation_active = False
        self.destination = None
        self.practice_gestures = []
        
    def set_mode(self, mode: GameMode):
        self.current_mode = mode
        self.mode_start_time = time.time()
        self._initialize_mode()
        
    def _initialize_mode(self):
        if self.current_mode == GameMode.FREE_DRIVE:
            self.fuel = float('inf')
            self.traffic_density = 0.2
            
        elif self.current_mode == GameMode.PARKING_CHALLENGE:
            self.fuel = 100
            self.traffic_density = 0.1
            self._setup_parking_challenge()
            
        elif self.current_mode == GameMode.HIGHWAY_DRIVE:
            self.fuel = 100
            self.traffic_density = 0.3
            self._setup_highway_drive()
            
        elif self.current_mode == GameMode.FUEL_SAVER:
            self.fuel = 50
            self.max_fuel = 50
            self.traffic_density = 0.2
            self._setup_fuel_saver()
            
        elif self.current_mode == GameMode.TIME_TRIAL:
            self.fuel = 100
            self.traffic_density = 0.1
            self._setup_time_trial()
            
        elif self.current_mode == GameMode.TRAFFIC_MODE:
            self.fuel = 100
            self.traffic_density = 0.5
            self._setup_traffic_mode()
            
        elif self.current_mode == GameMode.NAVIGATION:
            self.fuel = 100
            self.traffic_density = 0.3
            self._setup_navigation()
            
        elif self.current_mode == GameMode.MISSION_MODE:
            self.fuel = 100
            self.traffic_density = 0.2
            self._setup_mission_mode()
            
        elif self.current_mode == GameMode.GESTURE_PRACTICE:
            self.fuel = float('inf')
            self.traffic_density = 0
            self._setup_gesture_practice()
            
    def _setup_parking_challenge(self):
        self.checkpoints = [
            {'position': np.array([10.0, 0.0, 10.0]), 'type': 'parking_spot'},
            {'position': np.array([-10.0, 0.0, -10.0]), 'type': 'parking_spot'},
            {'position': np.array([0.0, 0.0, 20.0]), 'type': 'parking_spot'}
        ]
        
    def _setup_highway_drive(self):
        self.checkpoints = [
            {'position': np.array([0.0, 0.0, 100.0]), 'type': 'speed_check'},
            {'position': np.array([0.0, 0.0, 200.0]), 'type': 'speed_check'},
            {'position': np.array([0.0, 0.0, 300.0]), 'type': 'speed_check'}
        ]
        
    def _setup_fuel_saver(self):
        self.checkpoints = [
            {'position': np.array([50.0, 0.0, 50.0]), 'type': 'fuel_station'},
            {'position': np.array([-50.0, 0.0, -50.0]), 'type': 'fuel_station'},
            {'position': np.array([0.0, 0.0, 100.0]), 'type': 'fuel_station'}
        ]
        
    def _setup_time_trial(self):
        self.checkpoints = [
            {'position': np.array([20.0, 0.0, 20.0]), 'type': 'checkpoint'},
            {'position': np.array([-20.0, 0.0, 20.0]), 'type': 'checkpoint'},
            {'position': np.array([0.0, 0.0, 40.0]), 'type': 'checkpoint'}
        ]
        
    def _setup_traffic_mode(self):
        self.checkpoints = [
            {'position': np.array([30.0, 0.0, 30.0]), 'type': 'traffic_light'},
            {'position': np.array([-30.0, 0.0, 30.0]), 'type': 'traffic_light'},
            {'position': np.array([0.0, 0.0, 60.0]), 'type': 'traffic_light'}
        ]
        
    def _setup_navigation(self):
        self.destination = np.array([100.0, 0.0, 100.0])
        self.navigation_active = True
        self.checkpoints = [
            {'position': np.array([25.0, 0.0, 25.0]), 'type': 'waypoint'},
            {'position': np.array([50.0, 0.0, 50.0]), 'type': 'waypoint'},
            {'position': np.array([75.0, 0.0, 75.0]), 'type': 'waypoint'}
        ]
        
    def _setup_mission_mode(self):
        self.checkpoints = [
            {'position': np.array([40.0, 0.0, 40.0]), 'type': 'pickup'},
            {'position': np.array([-40.0, 0.0, -40.0]), 'type': 'delivery'},
            {'position': np.array([0.0, 0.0, 80.0]), 'type': 'checkpoint'}
        ]
        
    def _setup_gesture_practice(self):
        self.practice_gestures = [
            'steering',
            'acceleration',
            'brake',
            'gear_shift',
            'indicator',
            'lights',
            'horn',
            'parking'
        ]
        
    def update(self, vehicle, dt):
        if self.current_mode == GameMode.FREE_DRIVE:
            self._update_free_drive(vehicle, dt)
            
        elif self.current_mode == GameMode.PARKING_CHALLENGE:
            self._update_parking_challenge(vehicle, dt)
            
        elif self.current_mode == GameMode.HIGHWAY_DRIVE:
            self._update_highway_drive(vehicle, dt)
            
        elif self.current_mode == GameMode.FUEL_SAVER:
            self._update_fuel_saver(vehicle, dt)
            
        elif self.current_mode == GameMode.TIME_TRIAL:
            self._update_time_trial(vehicle, dt)
            
        elif self.current_mode == GameMode.TRAFFIC_MODE:
            self._update_traffic_mode(vehicle, dt)
            
        elif self.current_mode == GameMode.NAVIGATION:
            self._update_navigation(vehicle, dt)
            
        elif self.current_mode == GameMode.MISSION_MODE:
            self._update_mission_mode(vehicle, dt)
            
        elif self.current_mode == GameMode.GESTURE_PRACTICE:
            self._update_gesture_practice(vehicle, dt)
            
    def _update_free_drive(self, vehicle, dt):
        # No specific objectives, just drive freely
        pass
        
    def _update_parking_challenge(self, vehicle, dt):
        for checkpoint in self.checkpoints:
            if checkpoint['type'] == 'parking_spot':
                distance = np.linalg.norm(vehicle.position - checkpoint['position'])
                if distance < 2.0 and abs(vehicle.rotation) < 10:
                    self.score += 1000
                    checkpoint['completed'] = True
                    
    def _update_highway_drive(self, vehicle, dt):
        for checkpoint in self.checkpoints:
            if checkpoint['type'] == 'speed_check':
                distance = np.linalg.norm(vehicle.position - checkpoint['position'])
                if distance < 5.0:
                    if vehicle.velocity >= 100:  # km/h
                        self.score += 500
                    checkpoint['completed'] = True
                    
    def _update_fuel_saver(self, vehicle, dt):
        # Update fuel consumption
        self.fuel -= vehicle.velocity * 0.01 * dt
        
        # Check fuel stations
        for checkpoint in self.checkpoints:
            if checkpoint['type'] == 'fuel_station':
                distance = np.linalg.norm(vehicle.position - checkpoint['position'])
                if distance < 3.0:
                    self.fuel = min(self.fuel + 20, self.max_fuel)
                    checkpoint['completed'] = True
                    
    def _update_time_trial(self, vehicle, dt):
        time_elapsed = time.time() - self.mode_start_time
        if time_elapsed > 180:  # 3 minutes time limit
            return False
            
        for checkpoint in self.checkpoints:
            if checkpoint['type'] == 'checkpoint':
                distance = np.linalg.norm(vehicle.position - checkpoint['position'])
                if distance < 3.0:
                    self.score += 500
                    checkpoint['completed'] = True
                    
    def _update_traffic_mode(self, vehicle, dt):
        for checkpoint in self.checkpoints:
            if checkpoint['type'] == 'traffic_light':
                distance = np.linalg.norm(vehicle.position - checkpoint['position'])
                if distance < 5.0:
                    # Check if vehicle stopped at red light
                    if vehicle.velocity < 1.0:
                        self.score += 100
                    checkpoint['completed'] = True
                    
    def _update_navigation(self, vehicle, dt):
        if self.navigation_active:
            distance_to_destination = np.linalg.norm(vehicle.position - self.destination)
            if distance_to_destination < 5.0:
                self.score += 2000
                self.navigation_active = False
                
        for checkpoint in self.checkpoints:
            if checkpoint['type'] == 'waypoint':
                distance = np.linalg.norm(vehicle.position - checkpoint['position'])
                if distance < 3.0:
                    self.score += 500
                    checkpoint['completed'] = True
                    
    def _update_mission_mode(self, vehicle, dt):
        for checkpoint in self.checkpoints:
            if checkpoint['type'] == 'pickup':
                distance = np.linalg.norm(vehicle.position - checkpoint['position'])
                if distance < 3.0:
                    self.score += 500
                    checkpoint['completed'] = True
                    
            elif checkpoint['type'] == 'delivery':
                distance = np.linalg.norm(vehicle.position - checkpoint['position'])
                if distance < 3.0:
                    self.score += 1000
                    checkpoint['completed'] = True
                    
    def _update_gesture_practice(self, vehicle, dt):
        # Practice mode doesn't need specific updates
        pass
        
    def get_mode_status(self):
        # Return detailed mission status for HUD
        status = {
            'mode': self.current_mode.name,
            'score': self.score,
            'fuel': self.fuel,
            'checkpoints': self.checkpoints,
            'traffic_density': self.traffic_density,
            'navigation_active': self.navigation_active,
            'destination': self.destination,
            'practice_gestures': self.practice_gestures,
            'mission_feedback': self._get_mission_feedback(),
        }
        return status

    def _get_mission_feedback(self):
        # Provide real-time feedback for current mission
        if self.current_mode == GameMode.PARKING_CHALLENGE:
            return "Park in the highlighted spot!"
        elif self.current_mode == GameMode.TIME_TRIAL:
            return "Reach all checkpoints before time runs out!"
        elif self.current_mode == GameMode.DELIVERY:
            return "Deliver the cargo safely!"
        elif self.current_mode == GameMode.TRAFFIC_MODE:
            return "Avoid collisions and obey traffic rules!"
        return "Drive safely!"

class CareerMode:
    def __init__(self):
        self.level = 1
        self.exp = 0
        self.coins = 0
        self.missions: List[Mission] = []
        self.current_mission: Optional[Mission] = None
        self.completed_missions: List[str] = []
        self.unlocked_features: List[str] = []

    def add_mission(self, mission: Mission) -> None:
        self.missions.append(mission)

    def start_mission(self, mission_index: int) -> None:
        if 0 <= mission_index < len(self.missions):
            self.current_mission = self.missions[mission_index]
            self.current_mission.start()

    def update(self, dt: float, player_vehicle: Vehicle) -> None:
        if self.current_mission:
            self.current_mission.update(dt, player_vehicle)
            
            # Handle mission completion
            if self.current_mission.status == MissionStatus.COMPLETED:
                self.coins += self.current_mission.rewards['coins']
                self.exp += self.current_mission.rewards['exp']
                self.completed_missions.append(self.current_mission.name)
                self.unlocked_features.extend(self.current_mission.rewards['unlocks'])
                
                # Level up check
                if self.exp >= self.level * 100:
                    self.level += 1
                    self.exp = 0

    def draw(self, screen: pygame.Surface) -> None:
        if self.current_mission:
            self.current_mission.draw(screen)

        # Draw career stats
        font = pygame.font.Font(None, 36)
        level_text = font.render(f"Level: {self.level}", True, (255, 255, 255))
        exp_text = font.render(f"EXP: {self.exp}/{self.level * 100}", True, (255, 255, 255))
        coins_text = font.render(f"Coins: {self.coins}", True, (255, 255, 255))
        
        screen.blit(level_text, (screen.get_width() - 150, 10))
        screen.blit(exp_text, (screen.get_width() - 150, 50))
        screen.blit(coins_text, (screen.get_width() - 150, 90))
