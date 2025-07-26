from enum import Enum

class GestureType(Enum):
    STEERING = "steering"
    ACCELERATION = "acceleration"
    BRAKE = "brake"
    GEAR_SHIFT = "gear_shift"
    INDICATOR = "indicator"
    LIGHTS = "lights"
    HORN = "horn"
    PARKING = "parking"
    GAME_ACTION = "game_action"
    CAMERA = "camera" 