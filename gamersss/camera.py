import math
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from game_config import CAMERA_DISTANCE, CAMERA_HEIGHT, CAMERA_ANGLE

class Camera:
    def __init__(self):
        self.mode = 'third_person'  # third_person, cockpit, top_down
        self.distance = CAMERA_DISTANCE
        self.height = CAMERA_HEIGHT
        self.angle = CAMERA_ANGLE
        self.position = np.array([0.0, self.height, -self.distance])
        self.target = np.array([0.0, 0.0, 0.0])
        self.up = np.array([0.0, 1.0, 0.0])
        
    def set_mode(self, mode):
        valid_modes = ['third_person', 'cockpit', 'top_down']
        if mode in valid_modes:
            self.mode = mode
            
    def update(self, vehicle):
        if self.mode == 'third_person':
            # Follow behind the vehicle
            self.position[0] = vehicle.position[0] - math.sin(math.radians(vehicle.rotation)) * self.distance
            self.position[2] = vehicle.position[2] - math.cos(math.radians(vehicle.rotation)) * self.distance
            self.position[1] = vehicle.position[1] + self.height
            
            # Look at the vehicle
            self.target = vehicle.position
            
        elif self.mode == 'cockpit':
            # Position inside the vehicle
            self.position[0] = vehicle.position[0] + math.sin(math.radians(vehicle.rotation)) * 0.5
            self.position[2] = vehicle.position[2] + math.cos(math.radians(vehicle.rotation)) * 0.5
            self.position[1] = vehicle.position[1] + 0.8
            
            # Look in the direction the vehicle is facing
            self.target[0] = vehicle.position[0] + math.sin(math.radians(vehicle.rotation)) * 10
            self.target[2] = vehicle.position[2] + math.cos(math.radians(vehicle.rotation)) * 10
            self.target[1] = vehicle.position[1] + 0.8
            
        elif self.mode == 'top_down':
            # Position above the vehicle
            self.position[0] = vehicle.position[0]
            self.position[2] = vehicle.position[2]
            self.position[1] = vehicle.position[1] + 20
            
            # Look down at the vehicle
            self.target = vehicle.position
            
    def apply(self):
        glLoadIdentity()
        gluLookAt(
            self.position[0], self.position[1], self.position[2],
            self.target[0], self.target[1], self.target[2],
            self.up[0], self.up[1], self.up[2]
        )
        
    def get_view_matrix(self):
        # Calculate view matrix
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        view_matrix = np.identity(4)
        view_matrix[0, :3] = right
        view_matrix[1, :3] = up
        view_matrix[2, :3] = -forward
        view_matrix[0, 3] = -np.dot(right, self.position)
        view_matrix[1, 3] = -np.dot(up, self.position)
        view_matrix[2, 3] = np.dot(forward, self.position)
        
        return view_matrix 