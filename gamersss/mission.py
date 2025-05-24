import time
import math
import numpy as np
from game_config import MISSIONS, SCORE_MULTIPLIERS

class Mission:
    def __init__(self, mission_type):
        self.type = mission_type
        self.specs = MISSIONS[mission_type]
        self.start_time = time.time()
        self.completed = False
        self.failed = False
        self.score = 0
        self.objectives = []
        self.setup_objectives()
        
    def setup_objectives(self):
        if self.type == 'parking':
            self.objectives = [
                {
                    'type': 'park',
                    'position': np.array([10.0, 0.0, 10.0]),
                    'rotation': 0.0,
                    'tolerance': 1.0,
                    'completed': False
                }
            ]
        elif self.type == 'highway':
            self.objectives = [
                {
                    'type': 'speed',
                    'target_speed': 100,
                    'duration': 30,
                    'current_duration': 0,
                    'completed': False
                },
                {
                    'type': 'distance',
                    'target_distance': 5000,
                    'current_distance': 0,
                    'completed': False
                }
            ]
        elif self.type == 'delivery':
            self.objectives = [
                {
                    'type': 'collect',
                    'position': np.array([20.0, 0.0, 20.0]),
                    'completed': False
                },
                {
                    'type': 'deliver',
                    'position': np.array([-20.0, 0.0, -20.0]),
                    'completed': False
                }
            ]
            
    def update(self, vehicle):
        if self.completed or self.failed:
            return
            
        # Check time limit
        if time.time() - self.start_time > self.specs['time_limit']:
            self.failed = True
            return
            
        # Update objectives
        for objective in self.objectives:
            if objective['completed']:
                continue
                
            if objective['type'] == 'park':
                distance = np.linalg.norm(vehicle.position - objective['position'])
                rotation_diff = abs(vehicle.rotation - objective['rotation'])
                if distance < objective['tolerance'] and rotation_diff < 10:
                    objective['completed'] = True
                    self.score += 1000 * SCORE_MULTIPLIERS['perfect_parking']
                    
            elif objective['type'] == 'speed':
                if vehicle.velocity >= objective['target_speed']:
                    objective['current_duration'] += 1/60  # Assuming 60 FPS
                    if objective['current_duration'] >= objective['duration']:
                        objective['completed'] = True
                        self.score += 500 * SCORE_MULTIPLIERS['speed_bonus']
                        
            elif objective['type'] == 'distance':
                objective['current_distance'] += vehicle.velocity * (1/60)  # Assuming 60 FPS
                if objective['current_distance'] >= objective['target_distance']:
                    objective['completed'] = True
                    self.score += 1000 * SCORE_MULTIPLIERS['safe_driving']
                    
            elif objective['type'] == 'collect':
                distance = np.linalg.norm(vehicle.position - objective['position'])
                if distance < 2.0:
                    objective['completed'] = True
                    self.score += 500
                    
            elif objective['type'] == 'deliver':
                distance = np.linalg.norm(vehicle.position - objective['position'])
                if distance < 2.0:
                    objective['completed'] = True
                    self.score += 1000
                    
        # Check if all objectives are completed
        if all(obj['completed'] for obj in self.objectives):
            self.completed = True
            self.score += self.specs['reward']
            
    def get_status(self):
        return {
            'type': self.type,
            'time_remaining': max(0, self.specs['time_limit'] - (time.time() - self.start_time)),
            'completed': self.completed,
            'failed': self.failed,
            'score': self.score,
            'objectives': [
                {
                    'type': obj['type'],
                    'completed': obj['completed']
                } for obj in self.objectives
            ]
        } 