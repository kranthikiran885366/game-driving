import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import json
import os

class HandType(Enum):
    LEFT = "left"
    RIGHT = "right"

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

@dataclass
class HandState:
    position: Tuple[float, float, float]
    landmarks: List[Tuple[float, float, float]]
    gesture: Optional[GestureType] = None
    confidence: float = 0.0
    fatigue_level: float = 0.0

class GestureClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

class AIEngine:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Initialize gesture classifier
        self.gesture_classifier = GestureClassifier(
            input_size=63,  # 21 landmarks * 3 coordinates
            hidden_size=128,
            num_classes=len(GestureType)
        )
        
        # Load pre-trained weights if available
        if os.path.exists('gesture_model.pth'):
            self.gesture_classifier.load_state_dict(torch.load('gesture_model.pth'))
        
        self.gesture_classifier.eval()
        
        # Initialize scaler for input normalization
        self.scaler = StandardScaler()
        
        # State tracking
        self.hand_state = HandState(
            position=(0, 0, 0),
            landmarks=[],
            gesture=None,
            confidence=0.0,
            fatigue_level=0.0
        )
        
        # Settings
        self.settings = self._load_settings()
        
        # Analytics
        self.analytics = {
            'gesture_accuracy': [],
            'calibration_attempts': 0,
            'fatigue_events': 0,
            'crash_reports': []
        }
    
    def _load_settings(self) -> Dict:
        try:
            with open('ai_settings.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'hand_type': HandType.RIGHT.value,
                'gesture_sensitivity': 1.0,
                'low_light_mode': False,
                'auto_recenter': True,
                'custom_gestures': {},
                'voice_assistance': False,
                'adaptive_difficulty': True,
                'safety_mode': True,
                'fatigue_detection': True
            }
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, HandState]:
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply low-light enhancement if enabled
        if self.settings['low_light_mode']:
            rgb_frame = self._enhance_low_light(rgb_frame)
        
        # Process hands
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Get primary hand based on settings
            hand_idx = 0 if self.settings['hand_type'] == HandType.RIGHT.value else 1
            if hand_idx < len(results.multi_hand_landmarks):
                landmarks = results.multi_hand_landmarks[hand_idx]
                
                # Update hand state
                self.hand_state.landmarks = [
                    (lm.x, lm.y, lm.z) for lm in landmarks.landmark
                ]
                self.hand_state.position = self._calculate_hand_position()
                
                # Detect gesture
                gesture, confidence = self._detect_gesture()
                self.hand_state.gesture = gesture
                self.hand_state.confidence = confidence
                
                # Check for fatigue
                if self.settings['fatigue_detection']:
                    self._check_hand_fatigue()
                
                # Auto-recenter if enabled
                if self.settings['auto_recenter']:
                    self._auto_recenter_hand()
                
                # Draw hand tracking
                self._draw_hand_tracking(frame, landmarks)
        
        return frame, self.hand_state
    
    def _enhance_low_light(self, frame: np.ndarray) -> np.ndarray:
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    def _calculate_hand_position(self) -> Tuple[float, float, float]:
        if not self.hand_state.landmarks:
            return (0, 0, 0)
        
        # Calculate average position of all landmarks
        x = sum(lm[0] for lm in self.hand_state.landmarks) / len(self.hand_state.landmarks)
        y = sum(lm[1] for lm in self.hand_state.landmarks) / len(self.hand_state.landmarks)
        z = sum(lm[2] for lm in self.hand_state.landmarks) / len(self.hand_state.landmarks)
        
        return (x, y, z)
    
    def _detect_gesture(self) -> Tuple[Optional[GestureType], float]:
        if not self.hand_state.landmarks:
            return None, 0.0
        
        # Prepare input for classifier
        landmarks_flat = np.array(self.hand_state.landmarks).flatten()
        landmarks_normalized = self.scaler.transform([landmarks_flat])
        
        # Get prediction
        with torch.no_grad():
            input_tensor = torch.FloatTensor(landmarks_normalized)
            output = self.gesture_classifier(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            gesture = GestureType(predicted.item()) if confidence > 0.7 else None
            return gesture, confidence.item()
    
    def _check_hand_fatigue(self):
        if not self.hand_state.landmarks:
            return
        
        # Calculate movement variance
        if hasattr(self, '_last_landmarks'):
            movement = np.mean([
                np.linalg.norm(np.array(curr) - np.array(prev))
                for curr, prev in zip(self.hand_state.landmarks, self._last_landmarks)
            ])
            
            # Update fatigue level based on movement patterns
            if movement < 0.01:  # Very small movements
                self.hand_state.fatigue_level += 0.1
            else:
                self.hand_state.fatigue_level = max(0, self.hand_state.fatigue_level - 0.05)
            
            # Log fatigue event if threshold exceeded
            if self.hand_state.fatigue_level > 0.8:
                self.analytics['fatigue_events'] += 1
        
        self._last_landmarks = self.hand_state.landmarks
    
    def _auto_recenter_hand(self):
        if not self.hand_state.landmarks:
            return
        
        # Calculate center of hand
        center_x = sum(lm[0] for lm in self.hand_state.landmarks) / len(self.hand_state.landmarks)
        center_y = sum(lm[1] for lm in self.hand_state.landmarks) / len(self.hand_state.landmarks)
        
        # If hand is too far from center, adjust sensitivity
        if abs(center_x - 0.5) > 0.3 or abs(center_y - 0.5) > 0.3:
            self.settings['gesture_sensitivity'] *= 0.9
    
    def _draw_hand_tracking(self, frame: np.ndarray, landmarks):
        # Draw landmarks
        for landmark in landmarks.landmark:
            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # Draw gesture label if detected
        if self.hand_state.gesture:
            cv2.putText(
                frame,
                f"{self.hand_state.gesture.value} ({self.hand_state.confidence:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
    
    def calibrate(self, calibration_frames: List[np.ndarray]):
        """Perform gesture calibration using provided frames"""
        self.analytics['calibration_attempts'] += 1
        
        # Collect landmarks from calibration frames
        calibration_data = []
        for frame in calibration_frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                calibration_data.append([
                    (lm.x, lm.y, lm.z) for lm in landmarks.landmark
                ])
        
        if calibration_data:
            # Update scaler with calibration data
            self.scaler.fit(np.array(calibration_data).reshape(-1, 63))
            
            # Save calibration data
            with open('calibration_data.json', 'w') as f:
                json.dump({
                    'scaler_mean': self.scaler.mean_.tolist(),
                    'scaler_scale': self.scaler.scale_.tolist()
                }, f)
    
    def update_settings(self, new_settings: Dict):
        """Update AI engine settings"""
        self.settings.update(new_settings)
        with open('ai_settings.json', 'w') as f:
            json.dump(self.settings, f)
    
    def log_analytics(self, event_type: str, data: Dict):
        """Log analytics data"""
        if event_type == 'gesture_accuracy':
            self.analytics['gesture_accuracy'].append(data)
        elif event_type == 'crash':
            self.analytics['crash_reports'].append(data)
        
        # Save analytics periodically
        if len(self.analytics['gesture_accuracy']) % 100 == 0:
            with open('analytics.json', 'w') as f:
                json.dump(self.analytics, f)
    
    def get_adaptive_difficulty(self, player_level: int) -> Dict:
        """Calculate adaptive difficulty settings based on player level"""
        return {
            'traffic_density': min(0.8, 0.3 + (player_level * 0.05)),
            'ai_driver_skill': min(0.9, 0.4 + (player_level * 0.05)),
            'obstacle_frequency': min(0.7, 0.2 + (player_level * 0.05))
        }
    
    def check_safety(self) -> Tuple[bool, str]:
        """Check if current hand position/gesture is safe"""
        if not self.hand_state.landmarks:
            return True, "No hand detected"
        
        # Check for extreme positions
        center_x = sum(lm[0] for lm in self.hand_state.landmarks) / len(self.hand_state.landmarks)
        if center_x < 0.1 or center_x > 0.9:
            return False, "Hand position too extreme"
        
        # Check for rapid movements
        if hasattr(self, '_last_position'):
            movement = np.linalg.norm(
                np.array(self.hand_state.position) - np.array(self._last_position)
            )
            if movement > 0.5:
                return False, "Movement too rapid"
        
        self._last_position = self.hand_state.position
        return True, "Safe" 