# Gesture-Controlled Driving Game

A 3D driving game controlled entirely through hand gestures using computer vision and machine learning.

## Features

- Real-time hand gesture recognition using MediaPipe and PyTorch
- 3D driving environment with OpenGL
- Multiple game modes and vehicle customization
- Intuitive gesture controls for steering, acceleration, braking, and more
- Support for both left and right hand gestures
- Real-time visual feedback and guidance

## Gesture Controls

1. **Steering**: Hold hands like a steering wheel
2. **Acceleration**: Show palm facing up
3. **Brake**: Show palm facing down
4. **Gear Shift**: Make a gear shift gesture
5. **Indicator**: Point left or right
6. **Lights**: Show open palm
7. **Horn**: Make a fist
8. **Parking**: Show 'P' gesture
9. **Game Action**: Show victory sign
10. **Camera**: Make a camera gesture

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- MediaPipe
- PyGame
- OpenGL
- NumPy
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gesture-driving-game.git
cd gesture-driving-game
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the gesture recognition model:
```bash
python gamersss/train_gesture_classifier.py
```

2. Run the game:
```bash
python gamersss/run_game.py
```

## Project Structure

```
gamersss/
├── assets/           # Game assets (textures, models)
├── cars/            # Vehicle models and configurations
├── data/            # Training data and saved models
├── maps/            # Game maps and environments
├── models/          # Trained gesture recognition models
├── sounds/          # Game audio files
├── textures/        # Game textures
├── ui/              # UI elements and layouts
├── gesture_classifier.py    # Neural network model
├── gesture_controller.py    # Gesture processing
├── gesture_types.py         # Gesture definitions
├── train_gesture_classifier.py  # Training script
└── run_game.py      # Main game launcher
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MediaPipe for hand tracking
- PyTorch for machine learning
- PyGame for game development
- OpenGL for 3D rendering 