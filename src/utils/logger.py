import logging
import os
from datetime import datetime
from src.utils.config import config

class GameLogger:
    def __init__(self):
        self.log_dir = 'logs'
        self.setup_logger()
    
    def setup_logger(self):
        """Set up the logging system"""
        # Create logs directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.log_dir, f'game_{timestamp}.log')
        
        # Fix: Get log level as integer, not function
        log_level_str = config.get('debug', 'log_level', 'INFO')
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # Log system information
        self.log_system_info()
    
    def log_system_info(self):
        """Log system information"""
        import platform
        import sys
        import pygame
        import cv2
        import torch
        
        logging.info("=== System Information ===")
        logging.info(f"OS: {platform.system()} {platform.release()}")
        logging.info(f"Python: {sys.version}")
        logging.info(f"Pygame: {pygame.version.ver}")
        logging.info(f"OpenCV: {cv2.__version__}")
        logging.info(f"PyTorch: {torch.__version__}")
        logging.info("========================")
    
    def log_game_event(self, event_type: str, event_data: dict):
        """Log a game event"""
        logging.info(f"Game Event: {event_type} - {event_data}")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log an error"""
        logging.error(f"Error in {context}: {str(error)}", exc_info=True)
    
    def log_performance(self, fps: float, frame_time: float, memory_usage: float):
        """Log performance metrics"""
        if config.get('debug', 'show_fps', True):
            logging.debug(f"Performance - FPS: {fps:.1f}, Frame Time: {frame_time:.2f}ms, Memory: {memory_usage:.1f}MB")
    
    def log_gesture(self, gesture_type: str, confidence: float):
        """Log gesture detection"""
        logging.debug(f"Gesture Detected: {gesture_type} (confidence: {confidence:.2f})")
    
    def log_achievement(self, achievement_id: str, reward: int):
        """Log achievement unlock"""
        logging.info(f"Achievement Unlocked: {achievement_id} (reward: {reward})")
    
    def log_game_mode(self, mode: str, action: str, score: int = None):
        """Log game mode events"""
        if score is not None:
            logging.info(f"Game Mode: {mode} - {action} (score: {score})")
        else:
            logging.info(f"Game Mode: {mode} - {action}")
    
    def log_vehicle_event(self, event_type: str, vehicle_id: str, data: dict = None):
        """Log vehicle-related events"""
        if data:
            logging.info(f"Vehicle Event: {event_type} - Vehicle: {vehicle_id} - Data: {data}")
        else:
            logging.info(f"Vehicle Event: {event_type} - Vehicle: {vehicle_id}")
    
    def log_analytics(self, event_type: str, data: dict):
        """Log analytics events"""
        logging.info(f"Analytics: {event_type} - {data}")

# Create global logger instance
logger = GameLogger() 