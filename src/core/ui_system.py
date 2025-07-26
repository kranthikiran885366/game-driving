import pygame
import math
from typing import List, Dict, Tuple, Optional
from src.core.vehicle import Vehicle, GearMode
from src.core.environment import WeatherType, TimeOfDay

class UIElement:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.visible = True
        self.active = True

    def draw(self, screen: pygame.Surface) -> None:
        pass

    def handle_event(self, event: pygame.event.Event) -> bool:
        return False

class Button(UIElement):
    def __init__(self, x: int, y: int, width: int, height: int, text: str, callback):
        super().__init__(x, y, width, height)
        self.text = text
        self.callback = callback
        self.hovered = False
        self.font = pygame.font.Font(None, 36)

    def draw(self, screen: pygame.Surface) -> None:
        if not self.visible:
            return

        color = (100, 100, 255) if self.hovered else (50, 50, 200)
        pygame.draw.rect(screen, color, (self.x, self.y, self.width, self.height))
        
        text_surface = self.font.render(self.text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(self.x + self.width/2, self.y + self.height/2))
        screen.blit(text_surface, text_rect)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.active:
            return False

        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.x <= event.pos[0] <= self.x + self.width and \
                          self.y <= event.pos[1] <= self.y + self.height
        
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.hovered:
            self.callback()
            return True
        
        return False

class Speedometer(UIElement):
    def __init__(self, x: int, y: int, radius: int):
        super().__init__(x, y, radius * 2, radius * 2)
        self.radius = radius
        self.max_speed = 200
        self.current_speed = 0
        self.font = pygame.font.Font(None, 36)

    def update(self, speed: float) -> None:
        self.current_speed = speed

    def draw(self, screen: pygame.Surface) -> None:
        if not self.visible:
            return

        # Draw speedometer background
        pygame.draw.circle(screen, (50, 50, 50), (self.x + self.radius, self.y + self.radius), self.radius)
        
        # Draw speed markings
        for i in range(0, self.max_speed + 1, 20):
            angle = math.radians(i * 270 / self.max_speed - 225)
            start_pos = (
                self.x + self.radius + math.cos(angle) * (self.radius - 20),
                self.y + self.radius + math.sin(angle) * (self.radius - 20)
            )
            end_pos = (
                self.x + self.radius + math.cos(angle) * self.radius,
                self.y + self.radius + math.sin(angle) * self.radius
            )
            pygame.draw.line(screen, (255, 255, 255), start_pos, end_pos, 2)

        # Draw speed needle
        angle = math.radians(self.current_speed * 270 / self.max_speed - 225)
        end_pos = (
            self.x + self.radius + math.cos(angle) * (self.radius - 10),
            self.y + self.radius + math.sin(angle) * (self.radius - 10)
        )
        pygame.draw.line(screen, (255, 0, 0),
                        (self.x + self.radius, self.y + self.radius),
                        end_pos, 3)

        # Draw speed text
        speed_text = self.font.render(f"{int(self.current_speed)} km/h", True, (255, 255, 255))
        text_rect = speed_text.get_rect(center=(self.x + self.radius, self.y + self.radius + 30))
        screen.blit(speed_text, text_rect)

class GearIndicator(UIElement):
    def __init__(self, x: int, y: int):
        super().__init__(x, y, 80, 40)
        self.current_gear = GearMode.PARK
        self.manual_gear = 1
        self.font = pygame.font.Font(None, 36)

    def update(self, gear: GearMode, manual_gear: int = 1) -> None:
        self.current_gear = gear
        self.manual_gear = manual_gear

    def draw(self, screen: pygame.Surface) -> None:
        if not self.visible:
            return

        pygame.draw.rect(screen, (50, 50, 50), (self.x, self.y, self.width, self.height))
        
        if self.current_gear == GearMode.DRIVE:
            text = f"D{self.manual_gear}"
        else:
            text = self.current_gear.value
            
        text_surface = self.font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(self.x + self.width/2, self.y + self.height/2))
        screen.blit(text_surface, text_rect)

class FuelGauge(UIElement):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        self.fuel_level = 100
        self.font = pygame.font.Font(None, 36)

    def update(self, fuel_level: float) -> None:
        self.fuel_level = fuel_level

    def draw(self, screen: pygame.Surface) -> None:
        if not self.visible:
            return

        # Draw gauge background
        pygame.draw.rect(screen, (50, 50, 50), (self.x, self.y, self.width, self.height))
        
        # Draw fuel level
        fuel_width = int(self.width * self.fuel_level / 100)
        color = (0, 255, 0) if self.fuel_level > 20 else (255, 0, 0)
        pygame.draw.rect(screen, color, (self.x, self.y, fuel_width, self.height))
        
        # Draw fuel text
        text = f"Fuel: {int(self.fuel_level)}%"
        text_surface = self.font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(self.x + self.width/2, self.y + self.height/2))
        screen.blit(text_surface, text_rect)

class Minimap(UIElement):
    def __init__(self, x: int, y: int, size: int, world_width: int, world_height: int):
        super().__init__(x, y, size, size)
        self.world_width = world_width
        self.world_height = world_height
        self.scale_x = size / world_width
        self.scale_y = size / world_height
        self.player_pos = (0, 0)
        self.waypoints: List[Tuple[float, float]] = []
        self.objects: List[Dict] = []

    def update(self, player_pos: Tuple[float, float]) -> None:
        self.player_pos = player_pos

    def add_waypoint(self, pos: Tuple[float, float]) -> None:
        self.waypoints.append(pos)

    def add_object(self, pos: Tuple[float, float], obj_type: str) -> None:
        self.objects.append({'pos': pos, 'type': obj_type})

    def draw(self, screen: pygame.Surface) -> None:
        if not self.visible:
            return

        # Draw minimap background
        pygame.draw.rect(screen, (0, 0, 0), (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, (50, 50, 50), (self.x, self.y, self.width, self.height), 2)

        # Draw objects
        for obj in self.objects:
            x = self.x + obj['pos'][0] * self.scale_x
            y = self.y + obj['pos'][1] * self.scale_y
            if obj['type'] == 'building':
                pygame.draw.rect(screen, (100, 100, 100), (x-2, y-2, 4, 4))
            elif obj['type'] == 'checkpoint':
                pygame.draw.circle(screen, (0, 255, 0), (int(x), int(y)), 3)

        # Draw waypoints
        for waypoint in self.waypoints:
            x = self.x + waypoint[0] * self.scale_x
            y = self.y + waypoint[1] * self.scale_y
            pygame.draw.circle(screen, (255, 255, 0), (int(x), int(y)), 2)

        # Draw player
        player_x = self.x + self.player_pos[0] * self.scale_x
        player_y = self.y + self.player_pos[1] * self.scale_y
        pygame.draw.circle(screen, (255, 0, 0), (int(player_x), int(player_y)), 3)

class UISystem:
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.elements: Dict[str, UIElement] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        # Create speedometer
        self.elements['speedometer'] = Speedometer(
            self.screen_width - 150,
            self.screen_height - 150,
            60
        )

        # Create gear indicator
        self.elements['gear'] = GearIndicator(
            self.screen_width - 90,
            self.screen_height - 200
        )

        # Create fuel gauge
        self.elements['fuel'] = FuelGauge(
            10,
            self.screen_height - 40,
            200,
            30
        )

        # Create minimap
        self.elements['minimap'] = Minimap(
            10,
            10,
            150,
            2000,  # world width
            2000   # world height
        )

        # Create buttons
        self.elements['pause'] = Button(
            self.screen_width - 110,
            10,
            100,
            40,
            "Pause",
            lambda: print("Pause clicked")
        )

        self.elements['reset'] = Button(
            self.screen_width - 110,
            60,
            100,
            40,
            "Reset",
            lambda: print("Reset clicked")
        )

    def update(self, vehicle: Vehicle) -> None:
        # Update speedometer
        self.elements['speedometer'].update(abs(vehicle.speed))

        # Update gear indicator
        self.elements['gear'].update(vehicle.gear, vehicle.current_gear if vehicle.manual_mode else 1)

        # Update fuel gauge
        self.elements['fuel'].update(vehicle.fuel_level)

        # Update minimap
        self.elements['minimap'].update((vehicle.x, vehicle.y))

    def handle_event(self, event: pygame.event.Event) -> bool:
        handled = False
        for element in self.elements.values():
            if element.handle_event(event):
                handled = True
                break
        return handled

    def draw(self, screen: pygame.Surface) -> None:
        for element in self.elements.values():
            element.draw(screen)

class DrivingHUD:
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.ui = UISystem(screen_width, screen_height)
        self.show_tutorial = True
        self.tutorial_step = 0
        self.font = pygame.font.Font(None, 36)

    def update(self, vehicle: Vehicle) -> None:
        self.ui.update(vehicle)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if self.show_tutorial and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.tutorial_step += 1
                if self.tutorial_step >= 5:  # Number of tutorial steps
                    self.show_tutorial = False
                return True

        return self.ui.handle_event(event)

    def draw(self, screen: pygame.Surface) -> None:
        # Draw UI elements
        self.ui.draw(screen)

        # Draw tutorial if active
        if self.show_tutorial:
            self._draw_tutorial(screen)

    def _draw_tutorial(self, screen: pygame.Surface) -> None:
        tutorial_texts = [
            "Welcome to Dr. Driving! Press SPACE to continue...",
            "Use WASD or Arrow keys to control your vehicle",
            "Press SPACE for handbrake, SHIFT for gear changes",
            "Watch your fuel gauge and avoid damage",
            "Complete missions to earn coins and unlock new features"
        ]

        if self.tutorial_step < len(tutorial_texts):
            # Draw semi-transparent background
            overlay = pygame.Surface((self.screen_width, 100))
            overlay.fill((0, 0, 0))
            overlay.set_alpha(150)
            screen.blit(overlay, (0, self.screen_height - 100))

            # Draw tutorial text
            text = tutorial_texts[self.tutorial_step]
            text_surface = self.font.render(text, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(self.screen_width/2, self.screen_height - 50))
            screen.blit(text_surface, text_rect)
