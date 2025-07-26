import pygame
import random
import math
from typing import List, Tuple, Dict
from enum import Enum

class WeatherType(Enum):
    CLEAR = "clear"
    RAIN = "rain"
    SNOW = "snow"
    FOG = "fog"
    STORM = "storm"

class TimeOfDay(Enum):
    DAWN = "dawn"
    DAY = "day"
    DUSK = "dusk"
    NIGHT = "night"

class WeatherSystem:
    def __init__(self):
        self.weather = WeatherType.CLEAR
        self.time_of_day = TimeOfDay.DAY
        self.time_cycle = 0  # 0-24000 (representing 24 hours)
        self.weather_particles: List[Dict] = []
        self.wind_speed = 0
        self.wind_direction = 0
        self.visibility = 1.0
        self.road_friction = 1.0
        
        # Weather transition
        self.transitioning = False
        self.transition_target = None
        self.transition_progress = 0
        self.transition_duration = 300  # frames

        # Particle settings
        self.max_particles = 1000
        self.particle_colors = {
            WeatherType.RAIN: (100, 100, 255, 150),
            WeatherType.SNOW: (255, 255, 255, 200),
            WeatherType.FOG: (200, 200, 200, 50)
        }

    def update(self, dt: float):
        """Update weather system"""
        # Update time cycle
        self.time_cycle = (self.time_cycle + dt * 10) % 24000
        self._update_time_of_day()

        # Update weather transition
        if self.transitioning:
            self.transition_progress += 1
            if self.transition_progress >= self.transition_duration:
                self.weather = self.transition_target
                self.transitioning = False
            self._update_weather_effects()

        # Update particles
        self._update_particles(dt)

    def _update_time_of_day(self):
        """Update time of day based on time cycle"""
        hour = (self.time_cycle / 1000)
        if 5 <= hour < 7:
            self.time_of_day = TimeOfDay.DAWN
        elif 7 <= hour < 19:
            self.time_of_day = TimeOfDay.DAY
        elif 19 <= hour < 21:
            self.time_of_day = TimeOfDay.DUSK
        else:
            self.time_of_day = TimeOfDay.NIGHT

    def _update_weather_effects(self):
        """Update weather effects based on current weather"""
        if self.transitioning:
            progress = self.transition_progress / self.transition_duration
            
            if self.weather == WeatherType.CLEAR:
                self.visibility = 1.0 * (1 - progress) + 0.3 * progress
                self.road_friction = 1.0 * (1 - progress) + 0.6 * progress
            elif self.weather == WeatherType.RAIN:
                self.visibility = 0.7
                self.road_friction = 0.6
            elif self.weather == WeatherType.SNOW:
                self.visibility = 0.5
                self.road_friction = 0.4
            elif self.weather == WeatherType.FOG:
                self.visibility = 0.3
                self.road_friction = 0.8
            elif self.weather == WeatherType.STORM:
                self.visibility = 0.2
                self.road_friction = 0.3
                
            # Update wind
            if self.weather in [WeatherType.RAIN, WeatherType.STORM]:
                self.wind_speed = random.uniform(5, 15)
                self.wind_direction = random.uniform(0, 360)
            else:
                self.wind_speed = random.uniform(0, 5)
                self.wind_direction = random.uniform(0, 360)

    def _update_particles(self, dt: float):
        """Update weather particles"""
        # Remove old particles
        self.weather_particles = [p for p in self.weather_particles if p['life'] > 0]

        # Add new particles if needed
        if self.weather in [WeatherType.RAIN, WeatherType.SNOW] and len(self.weather_particles) < self.max_particles:
            for _ in range(10):
                self._spawn_particle()

        # Update existing particles
        for particle in self.weather_particles:
            # Update position
            particle['x'] += (particle['speed'] * math.cos(math.radians(particle['angle'])) + 
                            self.wind_speed * math.cos(math.radians(self.wind_direction))) * dt
            particle['y'] += (particle['speed'] * math.sin(math.radians(particle['angle'])) + 
                            self.wind_speed * math.sin(math.radians(self.wind_direction))) * dt
            
            # Update life
            particle['life'] -= dt

    def _spawn_particle(self):
        """Spawn a new weather particle"""
        if self.weather == WeatherType.RAIN:
            particle = {
                'x': random.randint(0, 1280),
                'y': random.randint(-100, 0),
                'speed': random.uniform(15, 25),
                'angle': 70 + random.uniform(-10, 10),
                'size': random.uniform(2, 4),
                'life': random.uniform(1, 2),
                'color': self.particle_colors[WeatherType.RAIN]
            }
        elif self.weather == WeatherType.SNOW:
            particle = {
                'x': random.randint(0, 1280),
                'y': random.randint(-100, 0),
                'speed': random.uniform(2, 5),
                'angle': 90 + random.uniform(-20, 20),
                'size': random.uniform(2, 4),
                'life': random.uniform(3, 6),
                'color': self.particle_colors[WeatherType.SNOW]
            }
        self.weather_particles.append(particle)

    def change_weather(self, weather_type: WeatherType):
        """Start weather transition"""
        if weather_type != self.weather and not self.transitioning:
            self.transitioning = True
            self.transition_target = weather_type
            self.transition_progress = 0

    def get_ambient_light(self) -> Tuple[int, int, int]:
        """Get ambient light color based on time of day"""
        if self.time_of_day == TimeOfDay.DAWN:
            return (255, 200, 150)
        elif self.time_of_day == TimeOfDay.DAY:
            return (255, 255, 255)
        elif self.time_of_day == TimeOfDay.DUSK:
            return (255, 150, 100)
        else:  # NIGHT
            return (50, 50, 100)

    def draw(self, screen: pygame.Surface):
        """Draw weather effects"""
        # Draw sky color
        ambient_color = self.get_ambient_light()
        sky = pygame.Surface((1280, 720), pygame.SRCALPHA)
        sky.fill((*ambient_color, 50))
        screen.blit(sky, (0, 0))

        # Draw weather particles
        for particle in self.weather_particles:
            pygame.draw.circle(screen, particle['color'], 
                            (int(particle['x']), int(particle['y'])), 
                            int(particle['size']))

        # Draw fog
        if self.weather in [WeatherType.FOG, WeatherType.STORM]:
            fog = pygame.Surface((1280, 720), pygame.SRCALPHA)
            fog.fill((200, 200, 200, int(100 * (1 - self.visibility))))
            screen.blit(fog, (0, 0))

class Environment:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.weather = WeatherSystem()
        self.road_segments: List[Dict] = []
        self.buildings: List[Dict] = []
        self.decorations: List[Dict] = []

    def add_road_segment(self, start: Tuple[float, float], end: Tuple[float, float], width: float):
        """Add a road segment"""
        self.road_segments.append({
            'start': start,
            'end': end,
            'width': width
        })

    def add_building(self, x: float, y: float, width: float, height: float):
        """Add a building"""
        self.buildings.append({
            'x': x,
            'y': y,
            'width': width,
            'height': height
        })

    def add_decoration(self, x: float, y: float, type_: str):
        """Add a decoration (trees, signs, etc.)"""
        self.decorations.append({
            'x': x,
            'y': y,
            'type': type_
        })

    def update(self, dt: float):
        """Update environment"""
        self.weather.update(dt)

    def draw(self, screen: pygame.Surface):
        """Draw environment"""
        # Draw roads
        for road in self.road_segments:
            pygame.draw.line(screen, (50, 50, 50), road['start'], road['end'], int(road['width']))
            # Draw road markings
            pygame.draw.line(screen, (255, 255, 255), road['start'], road['end'], 2)

        # Draw buildings
        for building in self.buildings:
            pygame.draw.rect(screen, (100, 100, 100), 
                           (building['x'], building['y'], building['width'], building['height']))

        # Draw decorations
        for decoration in self.decorations:
            if decoration['type'] == 'tree':
                pygame.draw.circle(screen, (0, 100, 0), 
                                (int(decoration['x']), int(decoration['y'])), 10)
            elif decoration['type'] == 'sign':
                pygame.draw.rect(screen, (200, 200, 0), 
                               (decoration['x'] - 5, decoration['y'] - 20, 10, 20))

        # Draw weather effects
        self.weather.draw(screen)
