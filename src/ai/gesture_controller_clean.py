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
import time
from src.core.color_system import ThemeManager
import pygame
from src.ai.gesture_types import GestureType
from src.ai.gesture_classifier import GestureClassifier

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
    is_left: bool
    landmarks: List[Tuple[float, float, float]]
    fingers_up: List[bool]
    palm_center: Tuple[float, float, float]
    palm_normal: Tuple[float, float, float]

@dataclass
class GestureState:
    type: Optional[GestureType] = None
    confidence: float = 0.0
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    speed: float = 0.0
    duration: float = 0.0
    left_hand: Optional[HandState] = None
    right_hand: Optional[HandState] = None

class GestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.left_classifier = GestureClassifier(63, 128, len(GestureType))
        self.right_classifier = GestureClassifier(63, 128, len(GestureType))
        if os.path.exists('models/gesture_model_left.pth'):
            self.left_classifier.load_model('models/gesture_model_left.pth')
        if os.path.exists('models/gesture_model_right.pth'):
            self.right_classifier.load_model('models/gesture_model_right.pth')
        self.left_classifier.eval()
        self.right_classifier.eval()
        self.control_state = {
            'steering': 0.0,
            'acceleration': 0.0,
            'brake': 0.0,
            'gear': 'D',
            'left_indicator': False,
            'right_indicator': False,
            'headlights': False,
            'horn': False,
            'wipers': False,
            'handbrake': False
        }
        
        # Initialize gesture classifiers for both hands
        self.left_hand_classifier = GestureClassifier(
            input_size=63,  # 21 landmarks * 3 coordinates
            hidden_size=128,
            num_classes=len(GestureType)
        )
        
        self.right_hand_classifier = GestureClassifier(
            input_size=63,  # 21 landmarks * 3 coordinates
            hidden_size=128,
            num_classes=len(GestureType)
        )
        
        # Load pre-trained weights if available
        if os.path.exists('models/gesture_model_left.pth'):
            self.left_hand_classifier.load_model('models/gesture_model_left.pth')
        if os.path.exists('models/gesture_model_right.pth'):
            self.right_hand_classifier.load_model('models/gesture_model_right.pth')
        
        self.left_hand_classifier.eval()
        self.right_hand_classifier.eval()
        
        # Initialize scalers for input normalization
        self.left_hand_scaler = StandardScaler()
        self.right_hand_scaler = StandardScaler()
        
        # State tracking
        self.left_hand_state = None
        self.right_hand_state = None
        self.current_gesture = None
        self.gesture_start_time = 0
        self.last_gesture_position = None
        self.gesture_history = []
        
        # Theme manager for gesture feedback colors
        self.theme_manager = ThemeManager()
        
        # Load calibration data
        self._load_calibration()
        
        self.current_gesture = "none"
        self.gesture_confidence = 0.0
        self.hand_position = (0, 0)
        self.use_keyboard = True  # Use keyboard controls by default
        
        # Keyboard state
        self.keys = {
            'steering': 0.0,  # -1.0 to 1.0
            'acceleration': 0.0,  # 0.0 to 1.0
            'brake': 0.0,  # 0.0 to 1.0
            'left_indicator': False,
            'right_indicator': False,
            'horn': False,
            'nitro': False
        }
    
    def _load_calibration(self):
        try:
            with open('gesture_calibration.json', 'r') as f:
                self.calibration_data = json.load(f)
        except FileNotFoundError:
            self._save_calibration()
    
    def _save_calibration(self):
        with open('gesture_calibration.json', 'w') as f:
            json.dump(self.calibration_data, f)
    
    def _process_hands(self, results) -> Tuple[Optional[HandState], Optional[HandState]]:
        """Process hand landmarks and determine left/right hand states"""
        left_hand = None
        right_hand = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Get hand landmarks
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                
                # Determine if it's left or right hand
                is_left = handedness.classification[0].label == "Left"
                
                # Get finger states
                fingers_up = self._get_finger_states(landmarks)
                
                # Calculate palm center and normal
                palm_center = self._calculate_palm_center(landmarks)
                palm_normal = self._calculate_palm_normal(landmarks)
                
                # Create hand state
                hand_state = HandState(
                    is_left=is_left,
                    landmarks=landmarks,
                    fingers_up=fingers_up,
                    palm_center=palm_center,
                    palm_normal=palm_normal
                )
                
                if is_left:
                    left_hand = hand_state
                else:
                    right_hand = hand_state
        
        return left_hand, right_hand

    def _get_finger_states(self, landmarks: List[Tuple[float, float, float]]) -> List[bool]:
        """Determine which fingers are up"""
        # Finger tip indices
        thumb_tip = 4
        index_tip = 8
        middle_tip = 12
        ring_tip = 16
        pinky_tip = 20
        
        # Finger base indices
        thumb_base = 2
        index_base = 6
        middle_base = 10
        ring_base = 14
        pinky_base = 18
        
        # Check if each finger is up
        thumb_up = landmarks[thumb_tip][1] < landmarks[thumb_base][1]
        index_up = landmarks[index_tip][1] < landmarks[index_base][1]
        middle_up = landmarks[middle_tip][1] < landmarks[middle_base][1]
        ring_up = landmarks[ring_tip][1] < landmarks[ring_base][1]
        pinky_up = landmarks[pinky_tip][1] < landmarks[pinky_base][1]
        
        return [thumb_up, index_up, middle_up, ring_up, pinky_up]

    def _calculate_palm_center(self, landmarks: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        """Calculate the center of the palm"""
        # Use wrist and middle finger base as reference points
        wrist = landmarks[0]
        middle_base = landmarks[9]
        
        # Calculate center as midpoint
        center_x = (wrist[0] + middle_base[0]) / 2
        center_y = (wrist[1] + middle_base[1]) / 2
        center_z = (wrist[2] + middle_base[2]) / 2
        
        return (center_x, center_y, center_z)

    def _calculate_palm_normal(self, landmarks: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        """Calculate the normal vector of the palm"""
        # Use wrist and middle finger base to calculate palm normal
        wrist = landmarks[0]
        middle_base = landmarks[9]
        
        # Calculate normal vector
        dx = middle_base[0] - wrist[0]
        dy = middle_base[1] - wrist[1]
        dz = middle_base[2] - wrist[2]
        
        # Normalize
        length = np.sqrt(dx*dx + dy*dy + dz*dz)
        if length > 0:
            dx /= length
            dy /= length
            dz /= length
        
        return (dx, dy, dz)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hands
        results = self.hands.process(rgb_frame)
        
        # Process left and right hands
        self.left_hand_state, self.right_hand_state = self._process_hands(results)
        
        if results.multi_hand_landmarks:
            # Detect gesture using both hands
            gesture, confidence = self._detect_gesture(results)
            
            if gesture:
                # Update gesture state
                self._update_gesture_state(gesture, confidence, results)
                
                # Process the gesture
                self._process_gesture(gesture, results)
                
                # Draw feedback
                self._draw_gesture_feedback(frame, gesture, results)
        
        # Draw finger states and control state
        self._draw_finger_states(frame)
        self._draw_control_state(frame)
        
        return frame

    def _detect_gesture(self, results) -> Tuple[Optional[GestureType], float]:
        """Detect gesture using both hands"""
        if not results.multi_hand_landmarks:
            return None, 0.0
            
        # Get hand landmarks
        left_landmarks = None
        right_landmarks = None
        
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            if handedness.classification[0].label == "Left":
                left_landmarks = hand_landmarks.landmark
            else:
                right_landmarks = hand_landmarks.landmark
        
        # Process each hand separately
        left_gesture = None
        right_gesture = None
        left_confidence = 0.0
        right_confidence = 0.0
        
        if left_landmarks:
            left_features = self._extract_features(left_landmarks)
            left_gesture, left_confidence = self.left_classifier.predict(left_features)
            
        if right_landmarks:
            right_features = self._extract_features(right_landmarks)
            right_gesture, right_confidence = self.right_classifier.predict(right_features)
        
        # Combine gestures based on confidence
        if left_confidence > right_confidence:
            return left_gesture, left_confidence
        else:
            return right_gesture, right_confidence

    def _extract_features(self, landmarks) -> np.ndarray:
        """Extract features from hand landmarks"""
        features = []
        for landmark in landmarks:
            features.extend([landmark.x, landmark.y, landmark.z])
        return np.array(features)

    def _update_gesture_state(self, gesture: GestureType, confidence: float, results):
        """Update the current gesture state"""
        if self.current_gesture != gesture:
            self.current_gesture = gesture
            self.gesture_start_time = time.time()
            self.last_gesture_position = self._calculate_hand_position(results)
        
        self.gesture_confidence = confidence
        self.hand_position = self._calculate_hand_position(results)

    def _process_gesture(self, gesture: GestureType, results):
        """Process the detected gesture"""
        if gesture == GestureType.STEERING:
            self._process_steering(results)
        elif gesture == GestureType.ACCELERATION:
            self._process_acceleration(results)
        elif gesture == GestureType.BRAKE:
            self._process_brake(results)
        elif gesture == GestureType.GEAR_SHIFT:
            self._process_gear_shift(results)
        elif gesture == GestureType.INDICATOR:
            self._process_indicator(results)
        elif gesture == GestureType.LIGHTS:
            self._process_lights(results)
        elif gesture == GestureType.HORN:
            self._process_horn(results)
        elif gesture == GestureType.PARKING:
            self._process_parking(results)
        elif gesture == GestureType.GAME_ACTION:
            self._process_game_action(results)
        elif gesture == GestureType.CAMERA:
            self._process_camera(results)

    def _process_steering(self, results):
        """Process steering gesture"""
        if self.right_hand_state:
            # Calculate steering based on hand position relative to center
            hand_x = self.right_hand_state.palm_center[0]
            steering = (hand_x - 0.5) * 2  # Convert to -1 to 1 range
            self.control_state['steering'] = np.clip(steering, -1.0, 1.0)

    def _process_acceleration(self, results):
        """Process acceleration gesture"""
        if self.right_hand_state:
            # Calculate acceleration based on hand position
            hand_y = self.right_hand_state.palm_center[1]
            acceleration = 1.0 - hand_y  # Convert to 0 to 1 range
            self.control_state['acceleration'] = np.clip(acceleration, 0.0, 1.0)

    def _process_brake(self, results):
        """Process brake gesture"""
        if self.left_hand_state:
            # Calculate brake based on hand position
            hand_y = self.left_hand_state.palm_center[1]
            brake = 1.0 - hand_y  # Convert to 0 to 1 range
            self.control_state['brake'] = np.clip(brake, 0.0, 1.0)

    def _process_gear_shift(self, results):
        """Process gear shift gesture"""
        if self.right_hand_state:
            # Detect gear shift gestures
            if self._is_hand_raised(self.right_hand_state.landmarks):
                self._shift_gear_up()
            elif self._is_hand_down(self.right_hand_state.landmarks):
                self._shift_gear_down()

    def _process_indicator(self, results):
        """Process indicator gesture"""
        if self.left_hand_state:
            # Detect left indicator
            if self._is_two_fingers_left(self.left_hand_state.landmarks):
                self.control_state['left_indicator'] = True
                self.control_state['right_indicator'] = False
            # Detect right indicator
            elif self._is_two_fingers_right(self.left_hand_state.landmarks):
                self.control_state['left_indicator'] = False
                self.control_state['right_indicator'] = True

    def _process_lights(self, results):
        """Process light control gesture"""
        if self.left_hand_state:
            # Toggle headlights
            if self._is_palm_forward(self.left_hand_state.landmarks):
                self.control_state['headlights'] = not self.control_state['headlights']

    def _process_horn(self, results):
        """Process horn gesture"""
        if self.right_hand_state:
            # Activate horn
            if self._is_fist(self.right_hand_state.landmarks):
                self.control_state['horn'] = True
            else:
                self.control_state['horn'] = False

    def _process_parking(self, results):
        """Process parking gesture"""
        if self.left_hand_state and self.right_hand_state:
            # Activate handbrake
            if self._is_stop_pose(self.left_hand_state.landmarks):
                self.control_state['handbrake'] = True
            else:
                self.control_state['handbrake'] = False

    def _process_game_action(self, results):
        """Process game action gesture"""
        if self.right_hand_state:
            # Detect game actions
            if self._is_punch_motion(self.right_hand_state.landmarks):
                self.control_state['nitro'] = True
            else:
                self.control_state['nitro'] = False

    def _process_camera(self, results):
        """Process camera control gesture"""
        if self.left_hand_state:
            # Detect camera controls
            if self._is_hand_wave(self.left_hand_state.landmarks):
                # Toggle camera view
                pass

    def _draw_gesture_feedback(self, frame: np.ndarray, gesture: Optional[GestureType], results):
        """Draw gesture feedback on frame"""
        if not gesture:
            return
            
        # Get theme colors
        colors = self.theme_manager.get_colors()
        
        # Draw gesture name and confidence
        gesture_text = f"{gesture.value}: {self.gesture_confidence:.2f}"
        cv2.putText(frame, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, colors['text'], 2)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, colors['landmark'], -1)

    def _draw_finger_states(self, frame: np.ndarray):
        """Draw finger states on frame"""
        if self.left_hand_state:
            self._draw_hand_fingers(frame, self.left_hand_state, (10, 60))
        if self.right_hand_state:
            self._draw_hand_fingers(frame, self.right_hand_state, (10, 120))

    def _draw_hand_fingers(self, frame: np.ndarray, hand_state: HandState, position: Tuple[int, int]):
        """Draw finger states for a single hand"""
        hand_type = "Left" if hand_state.is_left else "Right"
        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        
        for i, (name, is_up) in enumerate(zip(finger_names, hand_state.fingers_up)):
            color = (0, 255, 0) if is_up else (0, 0, 255)
            text = f"{hand_type} {name}: {'Up' if is_up else 'Down'}"
            cv2.putText(frame, text, (position[0], position[1] + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _draw_control_state(self, frame: np.ndarray):
        """Draw current control state on frame"""
        y = 180
        for control, value in self.control_state.items():
            text = f"{control}: {value}"
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 20

    def get_control_state(self) -> Dict:
        """Get current control state"""
        return self.control_state.copy()

    def calibrate(self, calibration_frames: List[np.ndarray]):
        """Calibrate gesture detection"""
        # Process calibration frames
        for frame in calibration_frames:
            results = self.hands.process(frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self._calibrate_hand_center(hand_landmarks.landmark)
        
        # Save calibration data
        self._save_calibration()

    def _calibrate_hand_center(self, landmarks):
        """Calibrate hand center position"""
        # Calculate average hand position
        center_x = sum(lm.x for lm in landmarks) / len(landmarks)
        center_y = sum(lm.y for lm in landmarks) / len(landmarks)
        center_z = sum(lm.z for lm in landmarks) / len(landmarks)
        
        # Update calibration data
        if 'hand_center' not in self.calibration_data:
            self.calibration_data['hand_center'] = []
        self.calibration_data['hand_center'].append([center_x, center_y, center_z])

    def _is_fist(self, landmarks) -> bool:
        """Check if hand is in a fist position"""
        return all(not finger for finger in self._get_finger_states(landmarks))

    def _is_palm_forward(self, landmarks) -> bool:
        """Check if palm is facing forward"""
        return landmarks[9].z < landmarks[0].z

    def _is_hand_raised(self, landmarks) -> bool:
        """Check if hand is raised"""
        return landmarks[0].y < 0.5

    def _is_palm_down(self, landmarks) -> bool:
        """Check if palm is facing down"""
        return landmarks[9].y > landmarks[0].y

    def _is_fist_down(self, landmarks) -> bool:
        """Check if fist is pulled down"""
        return self._is_fist(landmarks) and landmarks[0].y > 0.5

    def _is_stop_pose(self, landmarks) -> bool:
        """Check if both hands are in stop pose"""
        return (self._is_palm_forward(landmarks) and
                all(finger for finger in self._get_finger_states(landmarks)))

    def _is_finger_swipe_forward(self, landmarks) -> bool:
        """Check if index finger is swiping forward"""
        return (landmarks[8].x > landmarks[6].x and
                landmarks[8].z < landmarks[6].z)

    def _is_finger_swipe_backward(self, landmarks) -> bool:
        """Check if index finger is swiping backward"""
        return (landmarks[8].x < landmarks[6].x and
                landmarks[8].z > landmarks[6].z)

    def _is_fist_backward(self, landmarks) -> bool:
        """Check if fist is moving backward"""
        return self._is_fist(landmarks) and landmarks[0].z > 0.5

    def _is_hand_down(self, landmarks) -> bool:
        """Check if hand is pushed down"""
        return landmarks[0].y > 0.7

    def _is_two_fingers_left(self, landmarks) -> bool:
        """Check if two fingers are pointing left"""
        fingers = self._get_finger_states(landmarks)
        return fingers[1] and fingers[2] and not any(fingers[3:])

    def _is_two_fingers_right(self, landmarks) -> bool:
        """Check if two fingers are pointing right"""
        fingers = self._get_finger_states(landmarks)
        return fingers[1] and fingers[2] and not any(fingers[3:])

    def _is_v_shape_both_hands(self, landmarks) -> bool:
        """Check if both hands are in V shape"""
        fingers = self._get_finger_states(landmarks)
        return fingers[1] and fingers[2] and not any(fingers[3:])

    def _is_index_up(self, landmarks) -> bool:
        """Check if index finger is raised"""
        fingers = self._get_finger_states(landmarks)
        return fingers[1] and not any(fingers[2:])

    def _is_two_fingers_up(self, landmarks) -> bool:
        """Check if two fingers are raised"""
        fingers = self._get_finger_states(landmarks)
        return fingers[1] and fingers[2] and not any(fingers[3:])

    def _is_hand_wave(self, landmarks) -> bool:
        """Check if hand is waving"""
        # Check for side-to-side motion
        if len(self.gesture_history) < 5:
            return False
            
        recent_positions = [pos[0] for pos in self.gesture_history[-5:]]
        return max(recent_positions) - min(recent_positions) > 0.1

    def _is_punch_motion(self, landmarks) -> bool:
        """Check if hand is in punch motion"""
        return (self._is_fist(landmarks) and
                landmarks[0].z < 0.3)

    def _is_pinch_hold(self, landmarks) -> bool:
        """Check if fingers are pinched together"""
        return (landmarks[4].x - landmarks[8].x) < 0.05

    def _is_both_hands_up(self, landmarks) -> bool:
        """Check if both hands are up"""
        return landmarks[0].y < 0.3

    def _is_hand_up_freeze(self, landmarks) -> bool:
        """Check if hand is up and frozen"""
        if len(self.gesture_history) < 10:
            return False
            
        recent_positions = [pos for pos in self.gesture_history[-10:]]
        return (landmarks[0].y < 0.3 and
                max(p[1] for p in recent_positions) - min(p[1] for p in recent_positions) < 0.01)

    def _is_spock_hand(self, landmarks) -> bool:
        """Check if hand is in Spock position"""
        fingers = self._get_finger_states(landmarks)
        return fingers[1] and fingers[2] and not fingers[3] and fingers[4]

    def _is_both_palms_up(self, landmarks) -> bool:
        """Check if both palms are up"""
        return landmarks[9].y < landmarks[0].y

    def _is_fast_wave(self, landmarks) -> bool:
        """Check if hand is waving fast"""
        if len(self.gesture_history) < 3:
            return False
            
        recent_positions = [pos[0] for pos in self.gesture_history[-3:]]
        return max(recent_positions) - min(recent_positions) > 0.15

    def _is_two_peace_signs(self, landmarks) -> bool:
        """Check if both hands are in peace sign"""
        fingers = self._get_finger_states(landmarks)
        return fingers[1] and fingers[2] and not any(fingers[3:])

    def _is_crossed_fingers(self, landmarks) -> bool:
        """Check if fingers are crossed"""
        return (landmarks[8].x > landmarks[12].x and
                landmarks[8].y < landmarks[12].y)

    def _is_palm_five_fingers(self, landmarks) -> bool:
        """Check if palm is facing camera with five fingers spread"""
        return all(self._get_finger_states(landmarks))

    def _is_hold_palm(self, landmarks) -> bool:
        """Check if palm is held for 3 seconds"""
        if len(self.gesture_history) < 30:  # Assuming 10 fps
            return False
            
        recent_positions = [pos for pos in self.gesture_history[-30:]]
        return (self._is_palm_five_fingers(landmarks) and
                max(p[1] for p in recent_positions) - min(p[1] for p in recent_positions) < 0.01)

    def _is_circle_motion(self, landmarks) -> bool:
        """Check if hand is making a circle motion"""
        if len(self.gesture_history) < 20:
            return False
            
        recent_positions = [pos for pos in self.gesture_history[-20:]]
        center_x = sum(p[0] for p in recent_positions) / len(recent_positions)
        center_y = sum(p[1] for p in recent_positions) / len(recent_positions)
        
        # Check if points form a rough circle
        distances = [np.sqrt((p[0]-center_x)**2 + (p[1]-center_y)**2) for p in recent_positions]
        return max(distances) - min(distances) < 0.1

    def _calculate_hand_position(self, results) -> Tuple[float, float, float]:
        """Calculate current hand position"""
        if not results.multi_hand_landmarks:
            return (0.0, 0.0, 0.0)
            
        # Use the first detected hand
        landmarks = results.multi_hand_landmarks[0].landmark
        return (landmarks[0].x, landmarks[0].y, landmarks[0].z)

    def _calculate_hand_direction(self, results) -> Tuple[float, float, float]:
        """Calculate hand movement direction"""
        if len(self.gesture_history) < 2:
            return (0.0, 0.0, 0.0)
            
        current = self.gesture_history[-1]
        previous = self.gesture_history[-2]
        
        dx = current[0] - previous[0]
        dy = current[1] - previous[1]
        dz = current[2] - previous[2]
        
        # Normalize
        length = np.sqrt(dx*dx + dy*dy + dz*dz)
        if length > 0:
            dx /= length
            dy /= length
            dz /= length
            
        return (dx, dy, dz)

    def _calculate_hand_speed(self, current_position: Tuple[float, float, float]) -> float:
        """Calculate hand movement speed"""
        if len(self.gesture_history) < 2:
            return 0.0
            
        previous = self.gesture_history[-1]
        dx = current_position[0] - previous[0]
        dy = current_position[1] - previous[1]
        dz = current_position[2] - previous[2]
        
        return np.sqrt(dx*dx + dy*dy + dz*dz)

    def _shift_gear_up(self):
        """Shift gear up"""
        current_gear = self.control_state['gear']
        if current_gear == 'R':
            self.control_state['gear'] = 'N'
        elif current_gear == 'N':
            self.control_state['gear'] = 'D'
        elif current_gear == 'D':
            self.control_state['gear'] = '1'
        elif current_gear in ['1', '2', '3', '4', '5']:
            self.control_state['gear'] = str(int(current_gear) + 1)

    def _shift_gear_down(self):
        """Shift gear down"""
        current_gear = self.control_state['gear']
        if current_gear in ['2', '3', '4', '5', '6']:
            self.control_state['gear'] = str(int(current_gear) - 1)
        elif current_gear == '1':
            self.control_state['gear'] = 'D'
        elif current_gear == 'D':
            self.control_state['gear'] = 'N'
        elif current_gear == 'N':
            self.control_state['gear'] = 'R'

    def update(self) -> Dict:
        """Update gesture controller state"""
        # Update gesture history
        if self.hand_position != (0, 0):
            self.gesture_history.append(self.hand_position)
            if len(self.gesture_history) > 100:  # Keep last 100 positions
                self.gesture_history.pop(0)
        
        # Return current control state
        return self.get_control_state()

    def get_gesture_state(self) -> Tuple[str, float]:
        """Get current gesture state"""
        return self.current_gesture, self.gesture_confidence

    def get_hand_position(self) -> Tuple[float, float]:
        """Get current hand position"""
        return self.hand_position

    def toggle_control_mode(self):
        """Toggle between keyboard and gesture control"""
        self.use_keyboard = not self.use_keyboard
