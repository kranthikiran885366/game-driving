import os
import shutil
from pathlib import Path

def create_sound_files():
    # Create sounds directory if it doesn't exist
    sounds_dir = Path("sounds")
    sounds_dir.mkdir(exist_ok=True)
    
    # Create empty sound files
    sound_files = [
        "engine.wav",
        "crash.wav",
        "horn.wav",
        "gear_shift.wav",
        "nitro.wav",
        "achievement.wav",
        "menu_select.wav",
        "countdown.wav"
    ]
    
    for sound_file in sound_files:
        file_path = sounds_dir / sound_file
        if not file_path.exists():
            # Create an empty WAV file with basic header
            with open(file_path, 'wb') as f:
                # Write a minimal WAV header
                f.write(b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
    
    print("Created empty sound files in sounds directory")

if __name__ == "__main__":
    create_sound_files() 