import os
import shutil
import json

def integrate_gesture_model():
    """
    Integrate the advanced gesture model with the game by:
    1. Copying the model file to the correct location
    2. Updating the gesture mapping
    3. Creating a backup of existing files
    """
    print("Starting gesture model integration...")
    
    # Define file paths
    model_src = "models/advanced_gesture_model.pth"
    mapping_src = "advanced_gesture_mapping.json"
    
    # Backup existing files
    print("Creating backups...")
    if os.path.exists("models/gesture_model.pth"):
        shutil.copy2("models/gesture_model.pth", "models/gesture_model.pth.bak")
    if os.path.exists("gesture_mapping.json"):
        shutil.copy2("gesture_mapping.json", "gesture_mapping.json.bak")
    
    # Copy new model and mapping
    print("Copying new model and mapping...")
    try:
        # Copy model file
        shutil.copy2(model_src, "models/gesture_model.pth")
        
        # Copy and update gesture mapping
        with open(mapping_src, 'r') as f:
            mapping = json.load(f)
        
        # Ensure all required gesture types are in the mapping
        required_gestures = [
            "closedFist", "openPalm", "semiOpenFist", "semiOpenPalm",
            "fingerCircle", "fingerSymbols", "multiFingerBend", "singleFingerBend"
        ]
        
        # Update mapping with default values if any are missing
        updated = False
        for i, gesture in enumerate(required_gestures):
            if str(i) not in mapping.values():
                for k, v in list(mapping.items()):
                    if v == str(i):
                        del mapping[k]
                mapping[gesture] = str(i)
                updated = True
        
        # Save updated mapping if needed
        if updated:
            with open("gesture_mapping.json", 'w') as f:
                json.dump(mapping, f, indent=2)
        else:
            shutil.copy2(mapping_src, "gesture_mapping.json")
        
        print("Gesture model integration complete!")
        print("Backups created: gesture_model.pth.bak, gesture_mapping.json.bak")
        
    except Exception as e:
        print(f"Error during integration: {str(e)}")
        print("Attempting to restore from backups...")
        
        # Restore from backups if available
        if os.path.exists("models/gesture_model.pth.bak"):
            shutil.copy2("models/gesture_model.pth.bak", "models/gesture_model.pth")
        if os.path.exists("gesture_mapping.json.bak"):
            shutil.copy2("gesture_mapping.json.bak", "gesture_mapping.json")
        
        print("Original files have been restored from backups.")
        raise

if __name__ == "__main__":
    integrate_gesture_model()
