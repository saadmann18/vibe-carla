# AV Perception Portfolio

This repository contains projects and exercises for Autonomous Vehicle Perception.

## Structure
- `src/`: Source code for weekly projects.
- `docs/`: Documentation.
- `docker/`: Docker configurations.
- `tests/`: Tests.

## Week 1: Kalman Filter Sensor Fusion
Located in `src/week1_kalman_fusion/`.
Fuses IMU and GPS data from CARLA to estimate vehicle pose.

### Running
1. **Start CARLA Simulator**:
   ```bash
   cd /home/saad/dev/carlaSim/CARLA_0.9.15
   ./CarlaUE4.sh -quality-level=Low
   ```
2. **Start Driver Client** (Required for motion):
   ```bash
   cd /home/saad/dev/carlaSim/CARLA_0.9.15/PythonAPI/examples
   python manual_control.py
   ```
3. **Run Sensor Fusion**:
   ```bash
   python src/week1_kalman_fusion/kalman_fusion.py
   ```

## Documentation
- [Technical Report (PDF)](docs/tex/report.pdf): Detailed mathematical formulation of the Kalman Filter.
