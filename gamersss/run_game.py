import os
import sys
import subprocess
import time
import psutil
from config import config
from logger import logger

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import cv2
        import pygame
        import numpy
        import torch
        import mediapipe
        import OpenGL
        import requests
        import tqdm
        import psutil
        return True
    except ImportError as e:
        logger.log_error(e, "Dependency check")
        return False

def check_system_requirements():
    """Check if system meets minimum requirements"""
    try:
        # Check CPU
        cpu_count = psutil.cpu_count()
        if cpu_count < 2:
            logger.log_error(Exception("Insufficient CPU cores"), "System check")
            return False
        
        # Check memory
        memory = psutil.virtual_memory()
        if memory.total < 4 * 1024 * 1024 * 1024:  # 4GB
            logger.log_error(Exception("Insufficient RAM"), "System check")
            return False
        
        # Check disk space
        disk = psutil.disk_usage('/')
        if disk.free < 1 * 1024 * 1024 * 1024:  # 1GB
            logger.log_error(Exception("Insufficient disk space"), "System check")
            return False
        
        return True
    except Exception as e:
        logger.log_error(e, "System requirements check")
        return False

def setup_game():
    """Set up the game environment"""
    logger.log_game_event("setup_start", {"timestamp": time.time()})
    
    # Get the directory where run_game.py is located
    game_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create necessary directories
    directories = ['data', 'sounds', 'models', 'logs', 'saves']
    for directory in directories:
        dir_path = os.path.join(game_dir, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.log_game_event("directory_created", {"directory": directory})
    
    # Download sound files if not present
    sounds_dir = os.path.join(game_dir, 'sounds')
    if not os.path.exists(os.path.join(sounds_dir, 'background_music.mp3')):
        logger.log_game_event("sound_download_start", {"timestamp": time.time()})
        try:
            download_script = os.path.join(game_dir, 'download_sounds.py')
            subprocess.run([sys.executable, download_script], check=True)
            logger.log_game_event("sound_download_complete", {"timestamp": time.time()})
        except subprocess.CalledProcessError as e:
            logger.log_error(e, "Sound download")
            return False
    
    # Train gesture classifier if model doesn't exist
    models_dir = os.path.join(game_dir, 'models')
    if not os.path.exists(os.path.join(models_dir, 'gesture_model.pth')):
        logger.log_game_event("model_training_start", {"timestamp": time.time()})
        try:
            training_script = os.path.join(game_dir, 'train_gesture_classifier.py')
            subprocess.run([sys.executable, training_script], check=True)
            logger.log_game_event("model_training_complete", {"timestamp": time.time()})
        except subprocess.CalledProcessError as e:
            logger.log_error(e, "Model training")
            return False
    
    logger.log_game_event("setup_complete", {"timestamp": time.time()})
    return True

def main():
    """Main function to run the game"""
    logger.log_game_event("game_start", {"timestamp": time.time()})
    
    # Check dependencies
    if not check_dependencies():
        logger.log_error(Exception("Missing dependencies"), "Game startup")
        print("Please install all required dependencies:")
        print("pip install -r requirements.txt")
        return
    
    # Check system requirements
    if not check_system_requirements():
        logger.log_error(Exception("System requirements not met"), "Game startup")
        print("Your system does not meet the minimum requirements to run the game.")
        return
    
    # Setup game environment
    if not setup_game():
        logger.log_error(Exception("Game setup failed"), "Game startup")
        print("Failed to set up the game environment.")
        return
    
    # Run the game
    logger.log_game_event("game_launch", {"timestamp": time.time()})
    try:
        game_dir = os.path.dirname(os.path.abspath(__file__))
        game_script = os.path.join(game_dir, 'driving_hand.py')
        subprocess.run([sys.executable, game_script])
    except KeyboardInterrupt:
        logger.log_game_event("game_stopped", {"reason": "user_interrupt", "timestamp": time.time()})
        print("\nGame stopped by user")
    except Exception as e:
        logger.log_error(e, "Game execution")
        print(f"Error running game: {e}")
    finally:
        logger.log_game_event("game_end", {"timestamp": time.time()})
        print("Game ended")

if __name__ == "__main__":
    main() 