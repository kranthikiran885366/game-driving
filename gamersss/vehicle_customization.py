from enum import Enum
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass

class VehicleType(Enum):
    SPORTS = "sports"
    SUV = "suv"
    SEDAN = "sedan"
    TRUCK = "truck"

class PaintType(Enum):
    SOLID = "solid"
    METALLIC = "metallic"
    MATTE = "matte"
    CHROME = "chrome"
    CUSTOM = "custom"

class TireType(Enum):
    GRIP = "grip"
    SPEED = "speed"
    DURABILITY = "durability"

@dataclass
class VehicleStats:
    base_speed: float
    base_acceleration: float
    base_handling: float
    base_braking: float
    base_durability: float
    base_fuel_efficiency: float

@dataclass
class UpgradeLevel:
    level: int
    cost: int
    multiplier: float

class Vehicle:
    def __init__(self, vehicle_type: VehicleType):
        self.type = vehicle_type
        self.level = 1
        self.xp = 0
        self.coins = 0
        self.unlocked = False
        self.selected = False
        
        # Base stats based on vehicle type
        self.stats = self._get_base_stats()
        
        # Customization
        self.paint = {
            'type': PaintType.SOLID,
            'color': (255, 0, 0),  # Default red
            'custom_texture': None
        }
        self.tires = TireType.GRIP
        self.upgrades = {
            'engine': 0,
            'brakes': 0,
            'steering': 0,
            'suspension': 0,
            'transmission': 0
        }
        
        # Unlock requirements
        self.unlock_requirements = self._get_unlock_requirements()
        
    def _get_base_stats(self) -> VehicleStats:
        stats = {
            VehicleType.SPORTS: VehicleStats(
                base_speed=200,
                base_acceleration=12,
                base_handling=0.9,
                base_braking=15,
                base_durability=70,
                base_fuel_efficiency=0.7
            ),
            VehicleType.SUV: VehicleStats(
                base_speed=160,
                base_acceleration=8,
                base_handling=0.7,
                base_braking=12,
                base_durability=90,
                base_fuel_efficiency=0.5
            ),
            VehicleType.SEDAN: VehicleStats(
                base_speed=180,
                base_acceleration=10,
                base_handling=0.8,
                base_braking=13,
                base_durability=80,
                base_fuel_efficiency=0.6
            ),
            VehicleType.TRUCK: VehicleStats(
                base_speed=140,
                base_acceleration=6,
                base_handling=0.6,
                base_braking=10,
                base_durability=100,
                base_fuel_efficiency=0.4
            )
        }
        return stats[self.type]
    
    def _get_unlock_requirements(self) -> Dict:
        return {
            VehicleType.SPORTS: {'level': 10, 'coins': 50000},
            VehicleType.SUV: {'level': 5, 'coins': 25000},
            VehicleType.SEDAN: {'level': 1, 'coins': 10000},
            VehicleType.TRUCK: {'level': 8, 'coins': 35000}
        }
    
    def get_current_stats(self) -> Dict:
        # Calculate current stats with upgrades
        engine_multiplier = 1 + (self.upgrades['engine'] * 0.1)
        brake_multiplier = 1 + (self.upgrades['brakes'] * 0.1)
        steering_multiplier = 1 + (self.upgrades['steering'] * 0.1)
        
        # Apply tire effects
        tire_multipliers = {
            TireType.GRIP: {'handling': 1.2, 'speed': 0.9},
            TireType.SPEED: {'handling': 0.9, 'speed': 1.2},
            TireType.DURABILITY: {'handling': 1.0, 'speed': 1.0}
        }
        
        return {
            'speed': self.stats.base_speed * engine_multiplier * tire_multipliers[self.tires]['speed'],
            'acceleration': self.stats.base_acceleration * engine_multiplier,
            'handling': self.stats.base_handling * steering_multiplier * tire_multipliers[self.tires]['handling'],
            'braking': self.stats.base_braking * brake_multiplier,
            'durability': self.stats.base_durability,
            'fuel_efficiency': self.stats.base_fuel_efficiency
        }
    
    def can_upgrade(self, upgrade_type: str) -> bool:
        if upgrade_type not in self.upgrades:
            return False
        current_level = self.upgrades[upgrade_type]
        if current_level >= 5:  # Max upgrade level
            return False
        upgrade_cost = self._get_upgrade_cost(upgrade_type, current_level + 1)
        return self.coins >= upgrade_cost
    
    def _get_upgrade_cost(self, upgrade_type: str, level: int) -> int:
        base_costs = {
            'engine': 5000,
            'brakes': 3000,
            'steering': 4000,
            'suspension': 3500,
            'transmission': 4500
        }
        return base_costs[upgrade_type] * (level ** 1.5)
    
    def upgrade(self, upgrade_type: str) -> bool:
        if not self.can_upgrade(upgrade_type):
            return False
        
        cost = self._get_upgrade_cost(upgrade_type, self.upgrades[upgrade_type] + 1)
        self.coins -= cost
        self.upgrades[upgrade_type] += 1
        return True
    
    def set_paint(self, paint_type: PaintType, color: tuple, custom_texture: Optional[str] = None):
        self.paint['type'] = paint_type
        self.paint['color'] = color
        if custom_texture:
            self.paint['custom_texture'] = custom_texture
    
    def set_tires(self, tire_type: TireType):
        self.tires = tire_type
    
    def unlock(self) -> bool:
        if self.unlocked:
            return True
        
        requirements = self.unlock_requirements[self.type]
        if self.level >= requirements['level'] and self.coins >= requirements['coins']:
            self.coins -= requirements['coins']
            self.unlocked = True
            return True
        return False

class VehicleManager:
    def __init__(self):
        self.vehicles: Dict[VehicleType, Vehicle] = {
            vehicle_type: Vehicle(vehicle_type)
            for vehicle_type in VehicleType
        }
        self.selected_vehicle: Optional[VehicleType] = None
        
        # Unlock sedan by default
        self.vehicles[VehicleType.SEDAN].unlocked = True
        self.selected_vehicle = VehicleType.SEDAN
    
    def select_vehicle(self, vehicle_type: VehicleType) -> bool:
        if vehicle_type not in self.vehicles:
            return False
        
        vehicle = self.vehicles[vehicle_type]
        if not vehicle.unlocked:
            return False
        
        self.selected_vehicle = vehicle_type
        return True
    
    def get_selected_vehicle(self) -> Optional[Vehicle]:
        if self.selected_vehicle is None:
            return None
        return self.vehicles[self.selected_vehicle]
    
    def add_xp(self, amount: int):
        for vehicle in self.vehicles.values():
            vehicle.xp += amount
            # Level up if enough XP
            if vehicle.xp >= vehicle.level * 1000:
                vehicle.level += 1
                vehicle.xp = 0
    
    def add_coins(self, amount: int):
        for vehicle in self.vehicles.values():
            vehicle.coins += amount
    
    def get_available_upgrades(self, vehicle_type: VehicleType) -> Dict[str, bool]:
        vehicle = self.vehicles[vehicle_type]
        return {
            upgrade_type: vehicle.can_upgrade(upgrade_type)
            for upgrade_type in vehicle.upgrades
        }
    
    def get_vehicle_stats(self, vehicle_type: VehicleType) -> Dict:
        return self.vehicles[vehicle_type].get_current_stats()
    
    def get_unlock_status(self, vehicle_type: VehicleType) -> Dict:
        vehicle = self.vehicles[vehicle_type]
        requirements = vehicle.unlock_requirements
        return {
            'unlocked': vehicle.unlocked,
            'level_required': requirements['level'],
            'coins_required': requirements['coins'],
            'current_level': vehicle.level,
            'current_coins': vehicle.coins
        } 