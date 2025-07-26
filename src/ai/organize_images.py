import os
import shutil
import cv2
import mediapipe as mp
from tqdm import tqdm
import numpy as np

# Define the source and target directories
source_dir = r"C:\Users\cg515\Documents\kranthi\dr drivings\images"
target_base = r"C:\Users\cg515\Documents\kranthi\dr drivings\organized_images"

# Create target directories for each gesture type
gesture_types = [
    "steering", "acceleration", "brake", "gear_shift",
    "indicator", "lights", "horn", "parking", "game_action", "camera"
]

# Create target directories
for gesture in gesture_types:
    os.makedirs(os.path.join(target_base, gesture), exist_ok=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

def analyze_gesture(image_path):
    """Analyze an image to determine the gesture type"""
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None
            
        # For simplicity, we'll use some basic heuristics based on hand landmarks
        # You may need to adjust these based on your specific gestures
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Get hand landmarks
        landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
        
        # Calculate finger states (simple heuristic)
        # Thumb
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        thumb_open = thumb_tip[1] < thumb_mcp[1]
        
        # Other fingers
        fingers_open = []
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:  # Tips and PIP joints
            if landmarks[tip][1] < landmarks[pip][1]:  # If tip is above PIP
                fingers_open.append(True)
            else:
                fingers_open.append(False)
        
        # Simple gesture classification
        num_fingers = sum(fingers_open)
        
        if num_fingers == 5:
            return "steering"  # Open hand
        elif num_fingers == 0:
            return "brake"     # Fist
        elif num_fingers == 1 and fingers_open[0]:  # Index finger
            return "acceleration"
        elif num_fingers == 2 and fingers_open[0] and fingers_open[1]:  # Victory sign
            return "gear_shift"
        elif num_fingers == 3:
            return "indicator"
        else:
            return "game_action"  # Default
            
    except Exception as e:
        print(f"Error analyzing {image_path}: {e}")
        return None

def main():
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(source_dir) 
                  if f.lower().endswith(image_extensions)]
    
    print(f"Found {len(image_files)} images to organize...")
    
    # Process each image
    organized = 0
    for img_file in tqdm(image_files, desc="Organizing images"):
        src_path = os.path.join(source_dir, img_file)
        
        # Try to determine gesture type
        gesture_type = analyze_gesture(src_path)
        
        if gesture_type and gesture_type in gesture_types:
            # Move to the appropriate directory
            dest_dir = os.path.join(target_base, gesture_type)
            shutil.copy2(src_path, os.path.join(dest_dir, img_file))
            organized += 1
        else:
            # Move to an 'uncategorized' folder if we can't determine the gesture
            uncategorized_dir = os.path.join(target_base, "uncategorized")
            os.makedirs(uncategorized_dir, exist_ok=True)
            shutil.copy2(src_path, os.path.join(uncategorized_dir, img_file))
    
    print(f"\nOrganization complete!")
    print(f"- Total images processed: {len(image_files)}")
    print(f"- Successfully organized: {organized}")
    print(f"- Unclassified images: {len(image_files) - organized}")
    print(f"\nOrganized images are in: {target_base}")

if __name__ == "__main__":
    main()
