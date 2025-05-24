import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from gesture_classifier import GestureClassifier
from gesture_types import GestureType
import cv2
import mediapipe as mp
from tqdm import tqdm
import time

class GestureDataset(Dataset):
    def __init__(self, data, labels, hand_type):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.hand_type = hand_type
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def try_open_camera(max_indices=5):
    for camera_index in range(max_indices):
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Successfully opened camera {camera_index}")
                return cap
            else:
                cap.release()
    return None

def calibrate_hand(frame, hand_landmarks):
    # Check if hand is centered and of reasonable size
    h, w, _ = frame.shape
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bbox_w = (max_x - min_x) * w
    bbox_h = (max_y - min_y) * h
    center_x = (min_x + max_x) / 2 * w
    center_y = (min_y + max_y) / 2 * h
    # Criteria
    size_ok = 100 < bbox_w < w * 0.8 and 100 < bbox_h < h * 0.8
    center_ok = w * 0.2 < center_x < w * 0.8 and h * 0.2 < center_y < h * 0.8
    return size_ok, center_ok, bbox_w, bbox_h, center_x, center_y

def draw_gesture_guide(frame, gesture_type):
    """Draw enhanced visual guide for the current gesture with step-by-step instructions"""
    h, w = frame.shape[:2]
    guide_text = ""
    steps = []
    
    if gesture_type == GestureType.STEERING:
        guide_text = "Make a steering wheel gesture"
        steps = [
            "1. Hold hands like holding a steering wheel",
            "2. Keep hands at chest level",
            "3. Maintain a comfortable distance"
        ]
        # Draw steering wheel icon with animation
        center_x, center_y = w - 100, 100
        cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), 2)
        cv2.line(frame, (center_x-30, center_y), (center_x+30, center_y), (255, 255, 255), 2)
        cv2.line(frame, (center_x, center_y-30), (center_x, center_y+30), (255, 255, 255), 2)
        # Add rotating effect
        angle = int(time.time() * 100) % 360
        rad = np.radians(angle)
        cv2.line(frame, 
                (center_x, center_y),
                (int(center_x + 30 * np.cos(rad)), int(center_y + 30 * np.sin(rad))),
                (0, 255, 255), 2)
    elif gesture_type == GestureType.ACCELERATION:
        guide_text = "Show palm facing up"
        steps = [
            "1. Open your palm",
            "2. Face palm upward",
            "3. Keep fingers slightly spread"
        ]
        # Draw palm up icon with animation
        center_x, center_y = w - 100, 100
        cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), 2)
        cv2.ellipse(frame, (center_x, center_y+10), (20, 10), 0, 0, 180, (255, 255, 255), 2)
        # Add pulsing effect
        pulse = int(10 * np.sin(time.time() * 5)) + 20
        cv2.circle(frame, (center_x, center_y), pulse, (0, 255, 255), 1)
    elif gesture_type == GestureType.BRAKE:
        guide_text = "Show palm facing down"
        # Draw palm down icon
        center_x, center_y = w - 100, 100
        cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), 2)
        cv2.ellipse(frame, (center_x, center_y-10), (20, 10), 0, 180, 360, (255, 255, 255), 2)
    elif gesture_type == GestureType.GEAR_SHIFT:
        guide_text = "Make a gear shift gesture"
        # Draw gear shift icon
        center_x, center_y = w - 100, 100
        cv2.line(frame, (center_x, center_y-30), (center_x, center_y+30), (255, 255, 255), 2)
        cv2.circle(frame, (center_x, center_y), 10, (255, 255, 255), 2)
    elif gesture_type == GestureType.INDICATOR:
        guide_text = "Point left or right"
        # Draw pointing hand icon
        center_x, center_y = w - 100, 100
        cv2.line(frame, (center_x-20, center_y), (center_x+20, center_y), (255, 255, 255), 2)
        cv2.line(frame, (center_x+20, center_y), (center_x+30, center_y-10), (255, 255, 255), 2)
    elif gesture_type == GestureType.LIGHTS:
        guide_text = "Show open palm"
        # Draw open palm icon
        center_x, center_y = w - 100, 100
        cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), 2)
        for i in range(5):
            angle = i * 72
            rad = np.radians(angle)
            x = int(center_x + 25 * np.cos(rad))
            y = int(center_y + 25 * np.sin(rad))
            cv2.line(frame, (center_x, center_y), (x, y), (255, 255, 255), 2)
    elif gesture_type == GestureType.HORN:
        guide_text = "Make a fist"
        # Draw fist icon
        center_x, center_y = w - 100, 100
        cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), 2)
    elif gesture_type == GestureType.PARKING:
        guide_text = "Show 'P' gesture"
        # Draw P gesture icon
        center_x, center_y = w - 100, 100
        cv2.putText(frame, "P", (center_x-10, center_y+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    elif gesture_type == GestureType.GAME_ACTION:
        guide_text = "Show victory sign"
        # Draw victory sign icon
        center_x, center_y = w - 100, 100
        cv2.line(frame, (center_x-20, center_y), (center_x, center_y-20), (255, 255, 255), 2)
        cv2.line(frame, (center_x+20, center_y), (center_x, center_y-20), (255, 255, 255), 2)
    elif gesture_type == GestureType.CAMERA:
        guide_text = "Make a camera gesture"
        # Draw camera icon
        center_x, center_y = w - 100, 100
        cv2.rectangle(frame, (center_x-20, center_y-15), (center_x+20, center_y+15), (255, 255, 255), 2)
        cv2.circle(frame, (center_x, center_y), 10, (255, 255, 255), 2)
    
    # Draw guide text with background
    text_size = cv2.getTextSize(guide_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = 10
    text_y = frame.shape[0] - 120
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (text_x-5, text_y-text_size[1]-5), 
                 (text_x+text_size[0]+5, text_y+5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw main guide text
    cv2.putText(frame, f"Gesture Guide: {guide_text}", (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw step-by-step instructions
    for i, step in enumerate(steps):
        step_y = text_y + 30 + i * 25
        cv2.putText(frame, step, (text_x, step_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

def draw_position_guide(frame, hand_landmarks):
    """Draw enhanced position guide with dynamic feedback"""
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Draw dynamic concentric circles
    time_val = time.time()
    pulse = int(10 * np.sin(time_val * 3))  # Pulsing effect
    
    # Outer circle (target area) with pulsing effect
    cv2.circle(frame, (center_x, center_y), 150 + pulse, (0, 255, 255), 2)
    # Middle circle (optimal area)
    cv2.circle(frame, (center_x, center_y), 100, (0, 255, 0), 2)
    # Inner circle (perfect position)
    cv2.circle(frame, (center_x, center_y), 50, (255, 255, 255), 2)
    # Center point with pulsing effect
    cv2.circle(frame, (center_x, center_y), 5 + pulse//2, (0, 0, 255), -1)
    
    if hand_landmarks:
        # Calculate hand center and metrics
        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]
        hand_center_x = int(sum(xs) / len(xs) * w)
        hand_center_y = int(sum(ys) / len(ys) * h)
        
        # Draw hand center with pulsing effect
        cv2.circle(frame, (hand_center_x, hand_center_y), 8 + pulse//2, (0, 255, 0), -1)
        
        # Draw dynamic line to center
        line_color = (0, 255, 255)
        if ((hand_center_x - center_x) ** 2 + (hand_center_y - center_y) ** 2) ** 0.5 < 100:
            line_color = (0, 255, 0)  # Green when close to center
        cv2.line(frame, (hand_center_x, hand_center_y), (center_x, center_y), line_color, 2)
        
        # Draw hand bounding box with dynamic color
        min_x = int(min(xs) * w)
        max_x = int(max(xs) * w)
        min_y = int(min(ys) * h)
        max_y = int(max(ys) * h)
        box_color = (0, 255, 0) if 100 < (max_x - min_x) < 300 else (0, 0, 255)
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), box_color, 2)
        
        # Draw distance indicator with arrow
        dx = center_x - hand_center_x
        dy = center_y - hand_center_y
        distance = ((dx) ** 2 + (dy) ** 2) ** 0.5
        angle = np.degrees(np.arctan2(dy, dx))
        
        if distance > 150:
            # Draw animated arrow
            arrow_length = 50 + int(10 * np.sin(time_val * 5))
            arrow_angle = np.radians(angle)
            end_x = int(hand_center_x + arrow_length * np.cos(arrow_angle))
            end_y = int(hand_center_y + arrow_length * np.sin(arrow_angle))
            cv2.arrowedLine(frame, (hand_center_x, hand_center_y), (end_x, end_y), (0, 0, 255), 2)
            
            # Draw distance text with background
            text = f"Move {int(distance)}px to center"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, 
                        (hand_center_x - text_size[0]//2, hand_center_y - 40),
                        (hand_center_x + text_size[0]//2, hand_center_y - 20),
                        (0, 0, 0), -1)
            cv2.putText(frame, text, 
                       (hand_center_x - text_size[0]//2, hand_center_y - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def collect_gesture_data(num_samples_per_gesture=100):
    """Collect gesture data with enhanced feedback and guidance"""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = try_open_camera(5)
    while not cap or not cap.isOpened():
        print("[ERROR] Could not open any camera. Press 'r' to retry or 'q' to quit.")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            return {'left': (np.array([]), np.array([])), 'right': (np.array([]), np.array([]))}
        elif key == ord('r'):
            cap = try_open_camera(5)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")
    
    left_hand_data = []
    left_hand_labels = []
    right_hand_data = []
    right_hand_labels = []
    
    current_gesture = None
    samples_collected = 0
    current_hand = None
    last_collection_time = 0
    collection_interval = 0.1
    preview_saved = False
    
    print("\n=== ENHANCED GESTURE COLLECTION INSTRUCTIONS ===")
    print("1. First, select which hand to use:")
    print("   - Press 'l' for LEFT hand")
    print("   - Press 'r' for RIGHT hand")
    print("2. Then select a gesture (1-10):")
    for i, gesture in enumerate(GestureType):
        print(f"   {i+1}: {gesture.value}")
    print("\n3. Follow the on-screen guide:")
    print("   - Watch the gesture icon on the right")
    print("   - Read the step-by-step instructions")
    print("   - Position your hand in the center target")
    print("   - Keep your hand within the yellow circle")
    print("   - Wait for the green border before moving")
    print("\n4. Navigation:")
    print("   - Press 'n' to skip to next gesture")
    print("   - Press 'b' to go back")
    print("   - Press 'q' to quit")
    print("   - Press 'h' to show this help again")
    print("=============================================\n")
    
    gesture_list = list(GestureType)
    gesture_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera")
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Draw status information
        y_offset = 30
        cv2.putText(frame, "=== GESTURE COLLECTION STATUS ===", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        if current_hand:
            cv2.putText(frame, f"Current Hand: {current_hand.upper()}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Select Hand: Press 'l' for LEFT or 'r' for RIGHT", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30
        
        if current_gesture:
            cv2.putText(frame, f"Current Gesture: {current_gesture.value}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            cv2.putText(frame, f"Samples Collected: {samples_collected}/{num_samples_per_gesture}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Draw gesture guide
            draw_gesture_guide(frame, current_gesture)
        else:
            cv2.putText(frame, "Select Gesture: Press 1-10", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        hand_detected = False
        hand_feedback = ""
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label.lower()
                confidence = handedness.classification[0].score
                color = (0, 255, 0) if hand_label == current_hand else (0, 0, 255)
                
                # Draw hand landmarks
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, color, -1)
                
                cv2.putText(frame, f"{hand_label.upper()} ({confidence:.2f})", (10, y_offset + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                if current_hand and hand_label == current_hand:
                    size_ok, center_ok, bbox_w, bbox_h, center_x, center_y = calibrate_hand(frame, hand_landmarks)
                    
                    # Draw position guide with concentric circles
                    draw_position_guide(frame, hand_landmarks)
                    
                    if not size_ok:
                        if bbox_w < 100 or bbox_h < 100:
                            hand_feedback = "Hand too small. Move closer to camera."
                        else:
                            hand_feedback = "Hand too large. Move farther from camera."
                    elif not center_ok:
                        hand_feedback = "Move hand to center target."
                    else:
                        hand_feedback = "Perfect! Hold still..."
                        hand_detected = True
                        if current_gesture and time.time() - last_collection_time > collection_interval:
                            landmarks_flat = np.array([
                                (lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark
                            ]).flatten()
                            
                            if current_hand == 'left':
                                left_hand_data.append(landmarks_flat)
                                left_hand_labels.append(gesture_list.index(current_gesture))
                            else:
                                right_hand_data.append(landmarks_flat)
                                right_hand_labels.append(gesture_list.index(current_gesture))
                            
                            samples_collected += 1
                            last_collection_time = time.time()
                            print(f"[DEBUG] Collected sample {samples_collected}/{num_samples_per_gesture} for {current_hand} hand, gesture {current_gesture.value}")
                            
                            if samples_collected == 1 and not preview_saved:
                                preview_path = f"data/preview_{current_hand}_{current_gesture.value}.jpg"
                                cv2.imwrite(preview_path, frame)
                                print(f"Saved preview: {preview_path}")
                                preview_saved = True
                            
                            if samples_collected >= num_samples_per_gesture:
                                print(f"\nCompleted collecting {num_samples_per_gesture} samples for {current_gesture.value} with {current_hand} hand!")
                                current_gesture = None
                                samples_collected = 0
                                preview_saved = False
                                gesture_idx += 1
                                if gesture_idx < len(gesture_list):
                                    current_gesture = gesture_list[gesture_idx]
                                else:
                                    current_gesture = None
                else:
                    hand_feedback = "Show the correct hand."
        else:
            hand_feedback = "No hands detected!"
        
        # Draw feedback
        cv2.putText(frame, hand_feedback, (10, y_offset + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw progress bar
        if current_hand and current_gesture and samples_collected > 0:
            progress = samples_collected / num_samples_per_gesture
            bar_width = int(frame.shape[1] * 0.8)
            bar_height = 20
            x = int(frame.shape[1] * 0.1)
            y = frame.shape[0] - 40
            
            cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (0, 0, 0), -1)
            progress_width = int(bar_width * progress)
            cv2.rectangle(frame, (x, y), (x + progress_width, y + bar_height), (0, 255, 0), -1)
            cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (255, 255, 255), 2)
            cv2.putText(frame, f"{int(progress * 100)}%", (x + bar_width + 10, y + bar_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw border
        if current_hand and current_gesture:
            box_color = (0, 255, 0) if hand_detected else (0, 0, 255)
            cv2.rectangle(frame, (5, 5), (frame.shape[1]-5, frame.shape[0]-5), box_color, 4)
        
        cv2.imshow('Gesture Collection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            current_hand = 'left'
            current_gesture = gesture_list[gesture_idx] if gesture_idx < len(gesture_list) else None
            samples_collected = 0
            preview_saved = False
            print("\nSelected LEFT hand mode")
        elif key == ord('r'):
            current_hand = 'right'
            current_gesture = gesture_list[gesture_idx] if gesture_idx < len(gesture_list) else None
            samples_collected = 0
            preview_saved = False
            print("\nSelected RIGHT hand mode")
        elif key in [ord(str(i)) for i in range(1, 10)] or key == ord('0'):
            if key == ord('0'):
                gesture_idx = 9  # 10th gesture (index 9)
            else:
                gesture_idx = key - ord('1')
            if gesture_idx < len(gesture_list):
                current_gesture = gesture_list[gesture_idx]
                samples_collected = 0
                preview_saved = False
                print(f"\nSelected gesture: {current_gesture.value}")
        elif key == ord('n'):
            gesture_idx += 1
            if gesture_idx < len(gesture_list):
                current_gesture = gesture_list[gesture_idx]
                samples_collected = 0
                preview_saved = False
                print(f"\nSkipped to gesture: {current_gesture.value}")
            else:
                print("\nNo more gestures to collect.")
                current_gesture = None
        elif key == ord('b'):
            gesture_idx = max(0, gesture_idx - 1)
            current_gesture = gesture_list[gesture_idx]
            samples_collected = 0
            preview_saved = False
            print(f"\nBack to gesture: {current_gesture.value}")
        elif key == ord('h'):
            print("\n=== QUICK HELP ===")
            print("1. Select hand (l/r)")
            print("2. Select gesture (1-10)")
            print("3. Follow on-screen guide")
            print("4. Keep hand in yellow circle")
            print("5. Wait for green border")
            print("==================\n")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n=== Collection Summary ===")
    print(f"Left hand samples: {len(left_hand_data)}")
    print(f"Right hand samples: {len(right_hand_data)}")
    print("========================\n")
    
    return {
        'left': (np.array(left_hand_data), np.array(left_hand_labels)),
        'right': (np.array(right_hand_data), np.array(right_hand_labels))
    }

def train_model(data_dict, batch_size=32, epochs=50, learning_rate=0.001):
    """Train separate models for left and right hands"""
    models = {}
    
    for hand_type, (data, labels) in data_dict.items():
        if data.shape[0] == 0 or labels.shape[0] == 0:
            print(f"[WARNING] No data for {hand_type} hand. Skipping training for this hand.")
            continue
        print(f"\nTraining model for {hand_type} hand...")
        
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            data, labels, test_size=0.2, random_state=42
        )
        
        # Create datasets and dataloaders
        train_dataset = GestureDataset(X_train, y_train, hand_type)
        val_dataset = GestureDataset(X_val, y_val, hand_type)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        model = GestureClassifier(
            input_size=data.shape[1],
            hidden_size=128,
            num_classes=len(GestureType)
        )
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_data, batch_labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()
            
            # Print statistics
            train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_model(f'models/gesture_model_{hand_type}.pth')
                print(f'Model saved for {hand_type} hand!')
            
            models[hand_type] = model
    
    return models

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Always collect new gesture data
    print("Collecting new gesture data...")
    data_dict = collect_gesture_data(num_samples_per_gesture=5)
    np.savez('data/gesture_data.npz',
             left_X=data_dict['left'][0], left_y=data_dict['left'][1],
             right_X=data_dict['right'][0], right_y=data_dict['right'][1])
    
    # Train models
    print("Training models...")
    models = train_model(data_dict)
    print("Training complete!")

if __name__ == "__main__":
    main() 