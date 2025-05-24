# Game Configuration
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FPS = 60

# Camera Settings
CAMERA_DISTANCE = 5.0
CAMERA_HEIGHT = 2.0
CAMERA_ANGLE = 45.0

# Vehicle Settings
VEHICLES = {
    'sedan': {
        'name': 'Sedan',
        'max_speed': 180,
        'acceleration': 0.5,
        'handling': 0.8,
        'braking': 0.7,
        'fuel_capacity': 60,
        'fuel_consumption': 0.1,
        'price': 0,
        'unlocked': True
    },
    'sports': {
        'name': 'Sports Car',
        'max_speed': 250,
        'acceleration': 0.8,
        'handling': 0.9,
        'braking': 0.8,
        'fuel_capacity': 50,
        'fuel_consumption': 0.15,
        'price': 10000,
        'unlocked': False
    },
    'truck': {
        'name': 'Truck',
        'max_speed': 120,
        'acceleration': 0.3,
        'handling': 0.5,
        'braking': 0.6,
        'fuel_capacity': 100,
        'fuel_consumption': 0.2,
        'price': 15000,
        'unlocked': False
    }
}

# Mission Types
MISSIONS = {
    'parking': {
        'name': 'Parking Challenge',
        'description': 'Park your vehicle in the designated spot with the correct orientation',
        'time_limit': 120,  # seconds
        'reward': 2000,
        'difficulty': 'easy'
    },
    'highway': {
        'name': 'Highway Cruise',
        'description': 'Maintain high speed for a specified duration and cover the target distance',
        'time_limit': 300,  # seconds
        'reward': 3000,
        'difficulty': 'medium'
    },
    'delivery': {
        'name': 'Delivery Run',
        'description': 'Collect the package and deliver it to the destination',
        'time_limit': 180,  # seconds
        'reward': 2500,
        'difficulty': 'medium'
    }
}

# Traffic Settings
TRAFFIC_DENSITY = 0.3
TRAFFIC_SPEED_RANGE = (30, 120)
TRAFFIC_SPAWN_INTERVAL = 2.0

# Weather Effects
WEATHER_TYPES = ['clear', 'rain', 'fog', 'night']
WEATHER_CHANGE_INTERVAL = 300  # seconds

# Scoring System
SCORE_MULTIPLIERS = {
    'perfect_parking': 1.5,  # Perfect parking alignment
    'speed_bonus': 1.2,      # Maintaining high speed
    'safe_driving': 1.3,     # Safe driving without collisions
    'time_bonus': 1.1       # Completing mission under time
}

# UI Colors
COLORS = {
    'background': (0.1, 0.1, 0.1),
    'text': (1.0, 1.0, 1.0),
    'warning': (1.0, 0.0, 0.0),
    'success': (0.0, 1.0, 0.0),
    'fuel_low': (1.0, 0.5, 0.0)
}

# Sound Effects
SOUNDS = {
    'engine': 'sounds/engine.wav',
    'horn': 'sounds/horn.wav',
    'crash': 'sounds/crash.wav',
    'success': 'sounds/success.wav',
    'ambient': 'sounds/ambient.wav'
}

# Multiplayer Settings
MULTIPLAYER_PORT = 5000
MAX_PLAYERS = 4
SYNC_INTERVAL = 0.1  # seconds

# Vehicle physics parameters
VEHICLE_PHYSICS = {
    'max_speed': 150,        # km/h
    'acceleration': 10,      # km/h/s
    'braking': 15,          # km/h/s
    'handling': 0.8,        # 0-1 scale
    'mass': 1500,           # kg
    'drag_coefficient': 0.3
}

# Game settings
GAME_SETTINGS = {
    'fps': 60,
    'gravity': 9.81,        # m/s^2
    'collision_threshold': 1.0,  # meters
    'max_health': 100,
    'damage_multiplier': 1.0
} 