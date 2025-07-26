import os
import shutil
from pathlib import Path

def create_directories():
    """Create the target directory structure."""
    dirs = [
        'src/core',
        'src/ai',
        'src/graphics',
        'src/ui',
        'src/utils',
        'config',
        'models',
        'assets/images',
        'assets/sounds',
        'tests',
        'docs'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def move_files():
    """Move files to their appropriate directories."""
    # Core game files
    core_files = [
        'driving_game.py',
        'environment.py',
        'game_config.py',
        'game_modes.py',
        'mission.py',
        'traffic_system.py',
        'vehicle.py',
        'vehicle_customization.py',
        'camera.py',
        'color_system.py',
        'ui_system.py'
    ]
    
    # AI and ML files
    ai_files = [
        'ai_engine.py',
        'analytics_manager.py',
        'analyze_performance.py',
        'gesture_classifier.py',
        'gesture_controller.py',
        'gesture_controller_clean.py',
        'gesture_types.py',
        'train_advanced.py',
        'train_improved.py',
        'integrate_gesture_model.py',
        'organize_images.py'
    ]
    
    # Graphics files
    graphics_files = [
        # Any OpenGL specific files would go here
    ]
    
    # UI files
    ui_files = [
        'game_hud.py',
        'game_ui.py',
        'settings_manager.py'
    ]
    
    # Utility files
    utils_files = [
        'config.py',
        'logger.py',
        'rewards_manager.py',
        'sound_manager.py',
        'cleanup.py'
    ]
    
    # Config files
    config_files = [
        'config.json',
        'gesture_mapping.json',
        'advanced_gesture_mapping.json'
    ]
    
    # Model files
    model_files = [
        'models/gesture_model.pth',
        'models/improved_gesture_model.pth',
        'models/advanced_gesture_model.pth'
    ]
    
    # Move files to their new locations
    file_mappings = {
        'src/core/': core_files,
        'src/ai/': ai_files,
        'src/graphics/': graphics_files,
        'src/ui/': ui_files,
        'src/utils/': utils_files,
        'config/': config_files,
        'models/': [f for f in model_files if os.path.exists(f)]
    }
    
    for target_dir, files in file_mappings.items():
        for file_path in files:
            if os.path.exists(file_path):
                target_path = os.path.join(target_dir, os.path.basename(file_path))
                shutil.move(file_path, target_path)
                print(f"Moved: {file_path} -> {target_path}")
    
    # Move test files
    if os.path.exists('tests'):
        for test_file in os.listdir('tests'):
            if test_file.endswith('.py'):
                shutil.move(f'tests/{test_file}', f'tests/{test_file}')
                print(f"Moved test file: tests/{test_file}")
    
    # Move documentation
    if os.path.exists('README.md'):
        shutil.copy('README.md', 'docs/README.md')
        print("Copied README.md to docs/")
    
    if os.path.exists('LICENSE'):
        shutil.copy('LICENSE', 'docs/LICENSE')
        print("Copied LICENSE to docs/")

def update_imports():
    """Update import statements in Python files to reflect the new structure."""
    # This is a simplified version - in a real scenario, you'd need to parse and update imports
    print("\nNote: You'll need to update import statements in the Python files to reflect the new structure.")
    print("This typically involves updating the module paths in import statements.")

def create_init_files():
    """Create __init__.py files to make Python treat directories as packages."""
    for root, dirs, _ in os.walk('src'):
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
        for dir_name in dirs:
            init_file = os.path.join(root, dir_name, '__init__.py')
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write('# Package initialization\n')
                print(f"Created: {init_file}")

def main():
    print("Starting project reorganization...")
    
    # Create the new directory structure
    create_directories()
    
    # Move files to their new locations
    move_files()
    
    # Create __init__.py files
    create_init_files()
    
    # Notify about import updates needed
    update_imports()
    
    print("\nReorganization complete!")
    print("Next steps:")
    print("1. Review and update import statements in Python files")
    print("2. Update any file paths in configuration files")
    print("3. Test the application to ensure everything works")

if __name__ == "__main__":
    main()
