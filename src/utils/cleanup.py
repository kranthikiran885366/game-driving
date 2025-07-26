import os
import shutil
from pathlib import Path

def cleanup_project():
    """Clean up unnecessary files from the project directory."""
    print("Starting cleanup...")
    
    # Files to remove
    files_to_remove = [
        # Backup files
        "driving_hand.py.bak",
        "run_game.py.bak",
        "train_gesture_classifier.py.bak",
        "gesture_mapping.json.bak",
        "models/gesture_model.pth.bak",
        
        # Temporary training files
        "improved_gesture_mapping.json",
        "confusion_matrix.png",
        
        # Old training scripts (keep train_improved.py for reference)
        "train_gesture_classifier.py",
        "train_gesture_classifier_fixed.py",
        "train_from_folders.py",
        "train_from_images.py",
        "train_from_nested_folders.py",
    ]
    
    # Directories to remove
    dirs_to_remove = [
        # Cache directories
        "__pycache__",
        "runs",  # TensorBoard logs
        "logs",  # Application logs
    ]
    
    # Remove files
    for file_path in files_to_remove:
        try:
            if os.path.exists(file_path):
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"Removed directory: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")
    
    # Remove directories
    for dir_path in dirs_to_remove:
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                print(f"Removed directory: {dir_path}")
        except Exception as e:
            print(f"Error removing {dir_path}: {e}")
    
    # Clean up empty directories
    for root, dirs, _ in os.walk('.', topdown=False):
        for dir_name in dirs:
            try:
                dir_path = os.path.join(root, dir_name)
                if os.path.isdir(dir_path) and not os.listdir(dir_path):
                    os.rmdir(dir_path)
                    print(f"Removed empty directory: {dir_path}")
            except Exception as e:
                print(f"Error removing empty directory {dir_path}: {e}")
    
    print("\nCleanup complete! The following files remain for gesture control:")
    print("- models/gesture_model.pth (the trained model)")
    print("- gesture_mapping.json (gesture to control mapping)")
    print("- update_gesture_controller.py (advanced gesture controller)")
    print("- integrate_gesture_model.py (model integration script)")

if __name__ == "__main__":
    cleanup_project()
