# Gesture-Controlled 3D Driving Game

A modern 3D driving game controlled entirely through hand gestures using computer vision and machine learning.

## 🚀 Features

### 🎮 Gameplay
- Real-time hand gesture recognition with deep learning
- Immersive 3D driving physics
- Multiple game modes (Free Roam, Time Trial, Challenge)
- Vehicle customization system
- Progressive difficulty system
- Dynamic weather and day/night cycles

### ✨ Technical Highlights
- **Computer Vision**: MediaPipe for real-time hand tracking
- **Deep Learning**: PyTorch-based gesture classification (76%+ accuracy)
- **Graphics**: OpenGL-accelerated 3D rendering
- **Performance**: Optimized for real-time processing
- **Modular Design**: Clean, maintainable codebase
- **Cross-Platform**: Works on Windows, macOS, and Linux

## 🎯 Gesture Controls

| Gesture | Action |
|---------|--------|
| 👊 Closed Fist | Brake |
| 🖐️ Open Palm | Accelerate |
| 🤏 Semi-Open Fist (Left/Right) | Steering |
| 👉 Single Finger | Left Indicator |
| ✌️ Two Fingers | Right Indicator |
| 👌 Finger Circle | Horn |
| 🤙 Hand Wave | Toggle Headlights |
| 🖐️✊ Open/Close Hand | Handbrake |

## 🛠️ Project Structure

```
game-driving/
├── models/                      # Trained gesture models
│   └── gesture_model.pth        # Pre-trained gesture classifier
├── config/                      # Configuration files
│   ├── config.json             # Game settings
│   └── gesture_mapping.json    # Gesture to control mapping
├── src/                        # Source code
│   ├── core/                   # Core game engine
│   ├── ai/                     # AI and machine learning
│   ├── graphics/               # 3D rendering
│   ├── ui/                     # User interface
│   └── utils/                  # Utility functions
├── tests/                      # Test cases
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam
- Modern GPU (recommended for best performance)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/game-driving.git
   cd game-driving
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the game**
   ```bash
   python run_game.py
   ```

## 🎮 How to Play

1. Position yourself in front of your webcam
2. Use hand gestures to control the vehicle
3. Try different game modes and challenges
4. Customize your vehicle and settings

## 🤖 Gesture Training (Optional)

The game comes with a pre-trained model, but you can train your own:

```bash
# Train a new gesture model
python train_advanced.py --data_path path/to/gesture/images --epochs 50
```

## 📊 Performance

- **Model Accuracy**: 76.19% validation accuracy
- **Inference Speed**: <10ms per frame (with GPU acceleration)
- **Memory Usage**: ~500MB (excluding PyTorch and OpenGL)

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MediaPipe for hand tracking
- PyTorch for deep learning
- Pygame for game development
- OpenGL for 3D rendering
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