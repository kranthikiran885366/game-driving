"""Test script to verify all imports are working correctly."""

def test_imports():
    """Test importing all modules to catch any import errors."""
    print("Testing imports...")
    
    # Core imports
    from src.core import driving_game, environment, game_config, game_modes
    from src.core import mission, traffic_system, vehicle, vehicle_customization
    from src.core import camera, color_system, ui_system
    
    # AI imports
    from src.ai import ai_engine, analytics_manager, analyze_performance
    from src.ai import gesture_classifier, gesture_controller, gesture_types
    from src.ai import train_advanced, train_improved, integrate_gesture_model, organize_images
    
    # UI imports
    from src.ui import game_hud, game_ui, settings_manager
    
    # Utils imports
    from src.utils import config, logger, rewards_manager, sound_manager, cleanup
    
    print("All imports successful!")

if __name__ == "__main__":
    test_imports()
