import os

def fix_file_imports(file_path):
    """Fix relative imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Get the directory depth to determine the number of parent references needed
        rel_depth = len(file_path.split(os.sep)) - 2  # -2 for filename and src/
        parent_ref = '.'.join(['..'] * rel_depth)
        
        # Update relative imports
        content = content.replace(
            'from src.core.vehicle import', 
            'from src.core.vehicle import'
        )
        content = content.replace(
            'from src.core.environment import', 
            'from src.core.environment import'
        )
        content = content.replace(
            'from src.core.game_config import', 
            'from src.core.game_config import'
        )
        content = content.replace(
            'from src.core.camera import', 
            'from src.core.camera import'
        )
        content = content.replace(
            'from src.core.color_system import', 
            'from src.core.color_system import'
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

def main():
    print("Fixing relative imports in core modules...")
    
    # Fix imports in core modules
    core_files = [
        'src/core/driving_game.py',
        'src/core/environment.py',
        'src/core/game_modes.py',
        'src/core/mission.py',
        'src/core/traffic_system.py',
        'src/core/vehicle.py',
        'src/core/vehicle_customization.py',
        'src/core/camera.py',
        'src/core/color_system.py',
        'src/core/ui_system.py',
    ]
    
    for file_path in core_files:
        if os.path.exists(file_path):
            if fix_file_imports(file_path):
                print(f"Fixed imports in: {file_path}")
    
    print("\nImport fixes complete!")
    print("Try running the test script again with: python test_imports.py")

if __name__ == "__main__":
    main()
