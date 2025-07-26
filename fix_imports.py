import os
import re
from pathlib import Path

# Define the import mappings
IMPORT_MAPPINGS = {
    # Core imports
    'from driving_game': 'from src.core.driving_game',
    'from environment': 'from src.core.environment',
    'from game_config': 'from src.core.game_config',
    'from game_modes': 'from src.core.game_modes',
    'from mission': 'from src.core.mission',
    'from traffic_system': 'from src.core.traffic_system',
    'from vehicle': 'from src.core.vehicle',
    'from vehicle_customization': 'from src.core.vehicle_customization',
    'from camera': 'from src.core.camera',
    'from color_system': 'from src.core.color_system',
    'from ui_system': 'from src.core.ui_system',
    
    # AI imports
    'from ai_engine': 'from src.ai.ai_engine',
    'from analytics_manager': 'from src.ai.analytics_manager',
    'from analyze_performance': 'from src.ai.analyze_performance',
    'from gesture_classifier': 'from src.ai.gesture_classifier',
    'from gesture_controller': 'from src.ai.gesture_controller',
    'from gesture_types': 'from src.ai.gesture_types',
    'from train_advanced': 'from src.ai.train_advanced',
    'from train_improved': 'from src.ai.train_improved',
    'from integrate_gesture_model': 'from src.ai.integrate_gesture_model',
    'from organize_images': 'from src.ai.organize_images',
    
    # UI imports
    'from game_hud': 'from src.ui.game_hud',
    'from game_ui': 'from src.ui.game_ui',
    'from settings_manager': 'from src.ui.settings_manager',
    
    # Utils imports
    'from config': 'from src.utils.config',
    'from logger': 'from src.utils.logger',
    'from rewards_manager': 'from src.utils.rewards_manager',
    'from sound_manager': 'from src.utils.sound_manager',
    'from cleanup': 'from src.utils.cleanup',
}

def update_file_imports(file_path):
    """Update imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Update imports
        for old_import, new_import in IMPORT_MAPPINGS.items():
            # Handle 'from x import' pattern
            content = re.sub(
                fr'(\s*)from\s+{re.escape(old_import)}\b',
                fr'\1{new_import}',
                content
            )
            # Handle 'import x' pattern
            content = re.sub(
                fr'(\s*)import\s+{re.escape(old_import)}\b',
                fr'\1import {new_import.split(".")[-1]}',
                content
            )
        
        # Update relative imports
        content = re.sub(
            r'from \s*\.\s*([a-zA-Z0-9_]*)',
            lambda m: f'from .{m.group(1).strip()}',
            content
        )
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def update_config_paths(file_path):
    """Update file paths in configuration files."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update model paths
        content = content.replace(
            'models/gesture_model.pth', 
            'models/gesture_model.pth'
        )
        content = content.replace(
            'gesture_mapping.json', 
            'config/gesture_mapping.json'
        )
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error updating config paths in {file_path}: {e}")
        return False

def main():
    print("Starting to update imports and file paths...")
    
    # Process all Python files
    python_files = []
    for root, _, files in os.walk('.'):
        if 'venv' in root or '__pycache__' in root:
            continue
        for file in files:
            if file.endswith('.py') and file != 'fix_imports.py':
                python_files.append(os.path.join(root, file))
    
    # Process all Python files
    updated_count = 0
    for file_path in python_files:
        if update_file_imports(file_path):
            print(f"Updated imports in: {file_path}")
            updated_count += 1
    
    # Update config files
    config_files = [
        'config/config.json',
        'config/gesture_mapping.json',
        'src/ai/gesture_controller.py',
        'src/ai/gesture_controller_clean.py',
        'src/ai/train_advanced.py',
        'src/ai/train_improved.py',
        'src/core/driving_hand.py',
        'src/core/game_config.py'
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            if update_config_paths(config_file):
                print(f"Updated paths in: {config_file}")
    
    print(f"\nUpdate complete! Processed {len(python_files)} files, updated {updated_count} files.")
    print("\nNext steps:")
    print("1. Run 'python -m pytest' to run tests")
    print("2. Run 'python src/core/driving_hand.py' to test the game")
    print("3. Check for any remaining import errors in the console output")

if __name__ == "__main__":
    main()
