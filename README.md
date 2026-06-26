# CDIO Golf Robot

Autonomous robot software for the CDIO golf robot project. The system uses a PC camera and vision model to detect the robot, balls, goals, and red-cross obstacles, then sends movement and claw commands to an EV3 robot over TCP.

## Project Structure

```text
src/
  pc/      PC-side autonomy, camera vision, pathfinding, pickup, delivery, and calibration
  ev3/     EV3-side socket server, motor control, command handling, and claw control
  main.py  Convenience entry point for the EV3 program
```

## Main Features

- Camera-based arena detection and perspective warp
- YOLO-based detection of robot, claw, balls, goals, and red-cross obstacles
- Pathfinding and obstacle avoidance
- Autonomous pickup and delivery sequence
- Camera-to-EV3 coordinate synchronization
- Motion and camera calibration tools
- EV3 command protocol for movement and claw control

## Requirements

PC-side:

- Python 3
- OpenCV
- NumPy
- Ultralytics
- python-dotenv

EV3-side:

- ev3dev
- ev3dev2 Python library

Install PC dependencies with something like:

```bash
pip install opencv-python numpy ultralytics python-dotenv
```

## Running the Robot

Connect to robot through SSH:
```bash
ssh robot@ev3dev.local
password: maker
```

Start the EV3 server on the robot:

```bash
cd src/ev3
python3 main.py
```

Then start the autonomous PC program:

```bash
cd src/pc
python3 main.py
```

By default, the PC connects to:

```text
ev3dev.local:5000
```

This can be changed in `src/pc/com_protocol.py`.

## Calibration

Camera warp calibration:

```bash
cd src/pc
python3 calibrate_warp.py
```

Motion calibration:

```bash
cd src/pc
python3 calibrate_motion.py
```

Calibration files are stored in:

```text
src/pc/warp_calibration.txt
src/ev3/calibration.txt
src/ev3/motion_calibration.txt
```

## Configuration

Most PC-side settings are in:

```text
src/pc/settings.py
```

Many settings can also be changed with environment variables starting with `GOLFBOT_`.

## Notes

- The vision model files are stored in `src/pc/models/`.
- The default active model is configured in `src/pc/settings.py`.
- The EV3 program must be running before the PC autonomy program connects.
- The PC and EV3 must be on the same network.
