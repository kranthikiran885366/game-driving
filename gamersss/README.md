# Gesture-Controlled 3D Driving Game

A modern 3D driving game controlled entirely through hand gestures using computer vision and machine learning.

## Features

### Core Game Features
- Real-time hand gesture recognition
- 3D driving physics
- Multiple game modes
- Vehicle customization
- Achievement system
- Daily rewards
- Challenge system
- Economy system

### Technical Features
- OpenGL-based 3D rendering
- MediaPipe hand tracking
- PyTorch gesture classification
- Real-time performance monitoring
- Comprehensive logging system
- Auto-save functionality
- Configurable settings
- CI/CD Pipeline with GitHub Actions
- Automated deployment to Netlify

### Controls
- **Steering**: Hand rotation
- **Acceleration**: Palm forward
- **Brake**: Fist
- **Gear Shift**: Two fingers up/down
- **Indicators**: Two fingers left/right
- **Lights**: Three fingers up
- **Horn**: Four fingers
- **Parking**: Five fingers spread
- **Game Actions**: Various hand gestures
- **Camera Control**: Hand position tracking

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kranthikiran885366/game-driving.git
cd game-driving
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the game:
```bash
python run_game.py
```

## System Requirements

- Python 3.8 or higher
- Webcam for gesture control
- OpenGL-compatible graphics card
- Minimum 4GB RAM
- 1GB free disk space

## Project Structure

```
game-driving/
├── driving_hand.py      # Main game file
├── gesture_controller.py # Hand gesture detection
├── gesture_classifier.py # ML model for gestures
├── vehicle_customization.py # Vehicle management
├── game_modes.py        # Game mode definitions
├── game_ui.py          # User interface
├── game_hud.py         # Heads-up display
├── rewards_manager.py  # Achievement system
├── analytics_manager.py # Game analytics
├── sound_manager.py    # Audio system
├── config.py           # Configuration
├── logger.py           # Logging system
├── run_game.py         # Game launcher
├── requirements.txt    # Dependencies
├── tests/             # Test directory
├── .github/           # GitHub Actions workflows
├── netlify.toml       # Netlify configuration
└── README.md          # Documentation
```

## Configuration

The game can be configured through `config.json` or in-game settings:

- Display settings (resolution, fullscreen, vsync)
- Audio settings (volumes, mute)
- Control sensitivity
- Gameplay options
- Graphics quality
- Debug options

## Development

### Running Tests
```bash
pytest
```

### Training the Gesture Classifier
```bash
python train_gesture_classifier.py
```

### Adding New Features
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Deployment

The game is automatically deployed to Netlify when changes are pushed to the main branch. The deployment process includes:

1. Running tests
2. Building the executable
3. Deploying to Netlify

### Manual Deployment
1. Build the game:
```bash
pyinstaller --onefile --windowed run_game.py
```

2. Deploy to Netlify:
```bash
netlify deploy --prod
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Kranthi Kiran - mallelakranthikiran@gmail.com

Project Link: [https://github.com/kranthikiran885366/game-driving](https://github.com/kranthikiran885366/game-driving)

Live Demo: [https://game-driving.netlify.app](https://game-driving.netlify.app)

## Acknowledgments

- OpenGL for 3D rendering
- MediaPipe for hand tracking
- PyTorch for machine learning
- Pygame for game development
- OpenCV for computer vision
- GitHub Actions for CI/CD
- Netlify for hosting 