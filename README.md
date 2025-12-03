# Thymio Robot Navigation System

Autonomous navigation system for a Thymio robot using computer vision, path planning, and Kalman filtering.

## Overview

The robot navigates from any starting position to a goal location while avoiding obstacles. An overhead camera provides localization via ArUco markers, and the system gracefully handles camera occlusion by switching to odometry-based estimation.

## Components

| Module | Description |
|--------|-------------|
| `main.py` | Main navigation loop |
| `vision_system.py` | Camera calibration, ArUco detection, obstacle detection |
| `kalman_filter.py` | Extended Kalman Filter for pose estimation |
| `path_planner.py` | A* path planning with visibility graphs |
| `motion_controller.py` | Motor control and local obstacle avoidance |
| `visualizer.py` | Real-time visualization with status panel |
| `utils.py` | Robot physical parameters |

## Hardware Setup

- **Robot**: Thymio II
- **Camera**: Overhead camera with bird's-eye view
- **ArUco Markers**:
  - IDs 0, 2, 3, 5: Map corners
  - ID 1: Goal position
  - ID 4: Robot (mounted on top)
- **Obstacles**: Blue rectangular objects

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Position the camera above the arena
2. Place ArUco markers at corners (IDs 0, 2, 3, 5) and goal (ID 1)
3. Attach marker ID 4 to the Thymio
4. Connect Thymio via USB or wireless dongle
5. Run:

```bash
python main.py
```

The system will auto-calibrate when all markers are detected, then begin navigation.

## Calibration Files

The Kalman filter requires pre-computed noise covariance matrices:
- `Q.npy`: Process noise covariance (3x3)
- `R.npy`: Measurement noise covariance (3x3)

## Controls

- Press `q` during calibration to quit

## Authors

Daniel Ataíde \
Nicholas Thole \
Rachid Kahia