# 🚗 Gesture-Controlled 3D Driving Game

A cutting-edge 3D driving simulation that uses hand gestures for control, powered by computer vision and deep learning. Experience intuitive, natural interaction as you navigate through various environments using just your hand movements.

## 🎮 Features

### 🕹️ Core Gameplay
- **Gesture-Based Controls**: Intuitive hand gesture recognition system
- **Realistic Physics**: Advanced vehicle dynamics and collision detection
- **Multiple Game Modes**:
  - 🏁 Free Roam: Explore open environments at your own pace
  - ⏱️ Time Trial: Race against the clock on challenging tracks
  - 🏆 Challenge Mode: Complete specific objectives and missions
- **Dynamic Environment**:
  - 🌦️ Weather system (rain, fog, clear)
  - 🌓 Day/Night cycles
  - 🏙️ Interactive traffic system

### 🤖 AI & Machine Learning
- **Gesture Recognition**:
  - 8+ gesture classes with 76.19% validation accuracy
  - Real-time processing at 30+ FPS
  - Robust to lighting and background variations
- **Traffic AI**:
  - Smart NPC vehicles with realistic behavior
  - Adaptive difficulty based on player performance

### 🎨 Graphics & Audio
- **3D Rendering**:
  - OpenGL-accelerated graphics
  - Dynamic lighting and shadows
  - Particle effects (dust, rain, headlights)
- **Immersive Audio**:
  - Engine sounds based on RPM
  - Environmental audio (traffic, weather, collisions)
  - Spatial audio for 3D positioning

## ✋ Gesture Controls

| Gesture | Action | Description |
|---------|--------|-------------|
| 👊 Closed Fist | Brake | Make a fist to slow down or stop the vehicle |
| 🖐️ Open Palm | Accelerate | Show open palm to accelerate forward |
| 🤏 Pinch (Left/Right) | Steering | Pinch left/right to steer the vehicle |
| 👉 Point Left | Left Turn | Point left to activate left turn signal |
| 👉 Point Right | Right Turn | Point right to activate right turn signal |
| 👌 OK Sign | Horn | Make an "OK" sign to honk the horn |
| ✋ Stop | Handbrake | Show stop sign to engage handbrake |
| 🤙 Call Me | Toggle Lights | Make "call me" gesture to toggle headlights |
| 🤲 Both Palms | Reset Position | Show both palms to reset vehicle position |

> **Tip**: Keep your hand visible to the camera and maintain good lighting for best gesture recognition.

## 🏗️ Project Structure

```
gesture-driving-game/
├── .github/                    # GitHub workflows and templates
├── assets/                    # Game assets (images, sounds, models)
│   ├── images/               # UI and texture images
│   └── sounds/               # Audio files
├── config/                    # Configuration files
│   ├── config.json           # Game settings
│   └── gesture_mapping.json  # Gesture to control mapping
├── docs/                     # Documentation
│   ├── api/                 # API documentation
│   └── guides/              # Tutorials and guides
├── models/                   # Trained ML models
│   ├── gesture_model.pth     # Pre-trained gesture classifier
│   └── advanced_model.pth    # Advanced gesture model
├── src/                      # Source code
│   ├── ai/                  # AI and machine learning
│   │   ├── __init__.py
│   │   ├── gesture_classifier.py
│   │   ├── gesture_controller.py
│   │   └── train_advanced.py
│   ├── core/                # Core game engine
│   │   ├── __init__.py
│   │   ├── driving_game.py
│   │   ├── vehicle.py
│   │   └── environment.py
│   ├── graphics/            # 3D rendering
│   │   ├── __init__.py
│   │   ├── renderer.py
│   │   └── shaders/
│   ├── ui/                  # User interface
│   │   ├── __init__.py
│   │   ├── game_hud.py
│   │   └── menus.py
│   └── utils/               # Utility functions
│       ├── __init__.py
│       ├── logger.py
│       └── helpers.py
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── .gitignore             # Git ignore rules
├── LICENSE               # License file
├── README.md             # This file
├── requirements.txt      # Python dependencies
└── setup.py             # Package configuration
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8 or newer**
- **Webcam** (built-in or external)
- **OpenGL 3.3+ compatible GPU**
- **Windows 10/11, macOS 10.15+, or Linux**
- **At least 4GB free RAM**

### 🛠 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/gesture-driving-game.git
   cd gesture-driving-game
   ```

2. **Set up a virtual environment** (recommended)
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   
   # For development (optional)
   pip install -e .
   ```

4. **Run the game**
   ```bash
   # Method 1: Using Python module
   python -m src.core
   
   # Method 2: Using entry point (after installing with -e)
   gesture-driving-game
   ```

5. **Calibrate your camera** (first run only)
   - Follow the on-screen instructions to calibrate gesture detection
   - Ensure good lighting and clear view of your hand

## 🎮 Gameplay Guide

### Getting Started
1. **Positioning**: Sit 2-3 feet from your webcam with good lighting
2. **Calibration**: Complete the one-time calibration when first launching
3. **Controls**: Use the gesture controls table above to navigate

### Game Modes
- **Free Roam**: Explore open worlds at your own pace
- **Time Attack**: Beat the clock on challenging tracks
- **Mission Mode**: Complete specific objectives and challenges
- **Multiplayer**: Race against AI or friends (local only)

### Tips for Best Performance
- Ensure even lighting on your hands
- Keep your hand visible to the camera
- Use deliberate, clear gestures
- Adjust camera angle if needed in settings

## 🤖 Advanced: Training Your Own Gesture Model

### Prerequisites
- NVIDIA GPU with CUDA support (recommended)
- At least 50 sample images per gesture
- Python environment with PyTorch and OpenCV

### Training Process

1. **Prepare Your Dataset**
   - Organize images in this structure:
     ```
     dataset/
     ├── closed_fist/
     │   ├── image1.jpg
     │   └── ...
     ├── open_palm/
     │   ├── image1.jpg
     │   └── ...
     └── ...
     ```

2. **Start Training**
   ```bash
   python src/ai/train_advanced.py \
     --data_path path/to/dataset \
     --epochs 100 \
     --batch_size 32 \
     --learning_rate 0.001
   ```

3. **Monitor Training**
   - TensorBoard logs are saved to `runs/`
   - Model checkpoints are saved to `models/`

### Hyperparameter Tuning
- Adjust learning rate, batch size, and model architecture
- Use data augmentation for better generalization
- Try different optimizers (Adam, SGD with momentum)

## ⚙️ Performance & Optimization

### System Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| GPU | Integrated Graphics | NVIDIA GTX 1060 / AMD RX 580 |
| RAM | 4GB | 8GB+ |
| Storage | 2GB free space | SSD recommended |

### Performance Metrics
- **Gesture Recognition**: 
  - Accuracy: 76.19% (validation)
  - Latency: <10ms per frame (GPU)
  - FPS: 30+ (depends on hardware)

- **Memory Usage**:
  - Game: ~500MB
  - ML Model: ~50MB
  - GPU Memory: 1-2GB (with acceleration)

### Optimization Tips
- Lower resolution for better performance
- Reduce shadow quality if experiencing lag
- Close background applications
- Update graphics drivers

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Report Bugs**
   - Open an issue with detailed reproduction steps
   - Include system information and error logs

2. **Suggest Features**
   - Propose new features or improvements
   - Discuss implementation approaches

3. **Submit Code**
   - Fork the repository
   - Create a feature branch
   - Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Build documentation
cd docs && make html
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe** for robust hand tracking
- **PyTorch** for deep learning framework
- **Pygame** for game development
- **OpenGL** for 3D rendering
- **NumPy** for numerical computing
- **SciKit-Learn** for machine learning utilities

## 📚 Resources

- [Documentation](docs/README.md)
- [API Reference](docs/api/README.md)
- [Troubleshooting Guide](docs/guides/TROUBLESHOOTING.md)
- [Changelog](CHANGELOG.md)

## 📞 Support

For support, please:
1. Check the [FAQ](docs/guides/FAQ.md)
2. Search existing issues
3. Open a new issue if needed
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