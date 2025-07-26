import os
import torch
import json
from src.ai.gesture_classifier import GestureClassifier
from src.ai.gesture_types import GestureType

class AdvancedGestureController:
    def __init__(self, model_path='models/advanced_gesture_model.pth', 
                 mapping_path='advanced_gesture_mapping.json'):
        """
        Initialize the advanced gesture controller with our trained model.
        
        Args:
            model_path: Path to the trained PyTorch model
            mapping_path: Path to the gesture mapping JSON file
        """
        # Load gesture mapping
        with open(mapping_path, 'r') as f:
            self.gesture_mapping = json.load(f)
        
        # Reverse the mapping for label to gesture type
        self.label_to_gesture = {v: k for k, v in self.gesture_mapping.items()}
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    
    def _load_model(self, model_path):
        """Load the trained model with the correct architecture."""
        # Load the saved model state
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model with correct architecture
        model = GestureClassifier(
            input_size=63,  # 21 landmarks * 3 coordinates (x, y, z)
            hidden_size=512,  # Match the size used in training
            num_classes=len(self.gesture_mapping)
        ).to(self.device)
        
        # Load the trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def process_frame(self, frame):
        """
        Process a single frame and detect gestures.
        
        Args:
            frame: Input BGR image frame
            
        Returns:
            dict: Dictionary containing control state based on detected gestures
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = self.hands.process(rgb_frame)
        
        control_state = {
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
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                               results.multi_handedness):
                # Extract landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                # Convert to tensor and predict
                landmarks_tensor = torch.FloatTensor(landmarks).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = self.model(landmarks_tensor)
                    _, predicted = torch.max(outputs, 1)
                    gesture_label = predicted.item()
                    
                # Map prediction to gesture type
                gesture_name = self.label_to_gesture.get(gesture_label, "unknown")
                
                # Update control state based on gesture
                self._update_control_state(control_state, gesture_name, hand_landmarks)
        
        return control_state
    
    def _update_control_state(self, control_state, gesture_name, landmarks):
        """Update control state based on detected gesture."""
        # Get the center of the hand (wrist landmark)
        wrist = landmarks.landmark[0]
        
        # Map gesture names to control actions
        if gesture_name == "closedFist":
            control_state['brake'] = 1.0
        elif gesture_name == "openPalm":
            control_state['acceleration'] = 1.0
        elif gesture_name == "semiOpenFist":
            # Use x-coordinate of wrist for steering
            if wrist.x < 0.4:  # Left side of frame
                control_state['steering'] = -1.0
            elif wrist.x > 0.6:  # Right side of frame
                control_state['steering'] = 1.0
        elif gesture_name == "fingerCircle":
            control_state['horn'] = True
        elif gesture_name == "singleFingerBend":
            control_state['left_indicator'] = True
        elif gesture_name == "fingerSymbols":
            control_state['right_indicator'] = True
        # Add more gesture mappings as needed

def main():
    # Example usage
    import cv2
    
    # Initialize controller
    controller = AdvancedGestureController()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        control_state = controller.process_frame(frame)
        print("Control State:", control_state)
        
        # Display the frame
        cv2.imshow('Gesture Control', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
