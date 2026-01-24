"""
Week 1: Kalman Filter Sensor Fusion
-----------------------------------
Fuses GPS and IMU data from CARLA using an Extended Kalman Filter (EKF) 
to estimate vehicle 2D pose and velocity.
"""

import os
import time
import math
import random
import datetime
import uuid
import numpy as np
import matplotlib.pyplot as plt
import carla
from filterpy.kalman import ExtendedKalmanFilter

# --- Configuration Constants ---
SIM_HOST = 'localhost'
SIM_PORT = 2000
SIM_TIMEOUT = 10.0
# Filter Parameters
SIGMA_GPS = 2.0         # Standard deviation of GPS noise (meters)
DT = 0.05               # Time step (seconds)
EKF_STEPS = 1000        # Number of simulation steps to collect

def ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)

class Sensor:
    """Wrapper for CARLA sensors to handle data callbacks."""
    def __init__(self, actor):
        self.actor = actor
        self.data = None
        self.actor.listen(self._callback)

    def _callback(self, data):
        self.data = data
    
    def get_data(self):
        return self.data
    
    def destroy(self):
        if self.actor.is_alive:
            self.actor.stop()
            self.actor.destroy()

def state_transition(x: np.ndarray, dt: float) -> np.ndarray:
    """
    CTRV (Constant Turn Rate and Velocity) Motion Model.
    x: [px, py, vx, vy, yaw, omega]
    """
    px, py, vx, vy, yaw, omega = x
    c, s = np.cos(yaw), np.sin(yaw)
    
    # Kinematic update assuming constant velocity in body frame
    px_new = px + (vx * c - vy * s) * dt
    py_new = py + (vx * s + vy * c) * dt
    yaw_new = yaw + omega * dt
    
    return np.array([px_new, py_new, vx, vy, yaw_new, omega])

def calculate_jacobian_F(x: np.ndarray, dt: float) -> np.ndarray:
    """
    Computes linearization of the motion model (Jacobian F).
    """
    _, _, vx, vy, yaw, _ = x
    c, s = np.cos(yaw), np.sin(yaw)
    
    F = np.eye(6)
    # Derivatives of px_new w.r.t vx, vy, yaw
    F[0, 2] = c * dt 
    F[0, 3] = -s * dt
    F[0, 4] = (-vx * s - vy * c) * dt

    # Derivatives of py_new w.r.t vx, vy, yaw
    F[1, 2] = s * dt
    F[1, 3] = c * dt
    F[1, 4] = (vx * c - vy * s) * dt
    
    # Derivative of yaw_new w.r.t omega
    F[4, 5] = dt
    
    return F

def measurement_function(x: np.ndarray) -> np.ndarray:
    """
    Observation model h(x).
    Maps state [px, py, vx, vy, yaw, omega] to [px, py, yaw, omega]
    """
    return np.array([x[0], x[1], x[4], x[5]])

def calculate_jacobian_H(x: np.ndarray) -> np.ndarray:
    """
    Computes Jacobian H (Observation Matrix).
    """
    H = np.zeros((4, 6))
    H[0, 0] = 1 # px
    H[1, 1] = 1 # py
    H[2, 4] = 1 # yaw
    H[3, 5] = 1 # omega
    return H

def main():
    # 1. Setup Logging
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_results_dir = os.path.join(script_dir, 'results')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{timestamp}_{str(uuid.uuid4())[:8]}"
    experiment_dir = os.path.join(base_results_dir, experiment_id)
    ensure_dir(experiment_dir)
    print(f"📂 Results will be saved to: {experiment_dir}")

    # 2. Connect to CARLA
    try:
        client = carla.Client(SIM_HOST, SIM_PORT)
        client.set_timeout(SIM_TIMEOUT)
        world = client.get_world()
    except RuntimeError as e:
        print(f"❌ Error connecting to CARLA: {e}")
        return

    # 3. Find 'Hero' Vehicle
    vehicle = None
    print("🔍 Searching for 'hero' vehicle...")
    for _ in range(5):
        actors = world.get_actors().filter('vehicle.*')
        for actor in actors:
            if actor.attributes.get('role_name') == 'hero':
                vehicle = actor
                break
        if vehicle: break
        time.sleep(1.0)
    
    if not vehicle:
        if len(actors) > 0:
            print("⚠️ 'hero' role not found. Attaching to arbitrary vehicle.")
            vehicle = actors[0]
        else:
            print("❌ No vehicles found in simulation. Run manual_control.py first.")
            return

    print(f"✅ Attached to: {vehicle.type_id} (ID: {vehicle.id})")

    # 4. Sensor Calibration & Spawning
    bp_lib = world.get_blueprint_library()
    imu_bp = bp_lib.find('sensor.other.imu')
    gps_bp = bp_lib.find('sensor.other.gnss')

    imu_actor = world.spawn_actor(imu_bp, carla.Transform(carla.Location(0,0,0)), attach_to=vehicle)
    gps_actor = world.spawn_actor(gps_bp, carla.Transform(carla.Location(0,0,0)), attach_to=vehicle)
    
    imu_sensor = Sensor(imu_actor)

    # 5. Initialize EKF
    kf = ExtendedKalmanFilter(dim_x=6, dim_z=4)
    kf.P *= 10.0
    # Measurement Noise R: Tuned for [GPS_x, GPS_y, IMU_yaw, IMU_omega]
    var_gps = SIGMA_GPS**2
    kf.R = np.diag([var_gps, var_gps, 0.1, 0.05])
    kf.Q = np.eye(6) * 0.1 # Process noise

    # Data Storage
    fused_poses, gps_meas = [], []

    try:
        print("Waiting for sensor data stream...")
        while imu_sensor.get_data() is None:
            world.wait_for_tick()
        
        # Initialize State
        start_gps = gps_actor.get_location()
        start_imu = imu_sensor.get_data()
        start_yaw = math.radians(start_imu.compass)
        kf.x = np.array([start_gps.x, start_gps.y, 0, 0, start_yaw, 0])

        print(f"🚀 Starting EKF Loop ({EKF_STEPS} steps)...")
        
        for i in range(EKF_STEPS):
            world.wait_for_tick()
            
            # --- Get Observations ---
            # 1. GPS (Simulate Noise)
            true_loc = gps_actor.get_location()
            z_gps_x = true_loc.x + random.gauss(0, SIGMA_GPS)
            z_gps_y = true_loc.y + random.gauss(0, SIGMA_GPS)
            gps_meas.append([z_gps_x, z_gps_y])

            # 2. IMU
            imu_data = imu_sensor.get_data()
            if not imu_data: continue
            z_yaw = math.radians(imu_data.compass)
            z_omega = imu_data.gyroscope.z

            # --- EKF Prediction Step ---
            kf.F = calculate_jacobian_F(kf.x, DT)
            kf.x = state_transition(kf.x, DT)
            kf.predict()

            # --- EKF Correction Step ---
            z = np.array([z_gps_x, z_gps_y, z_yaw, z_omega])
            kf.update(z, calculate_jacobian_H, measurement_function)

            # Store Result
            fused_poses.append(kf.x[:2].copy())

            if i % 100 == 0:
                print(f"Step {i:4d}: Pos=({kf.x[0]:.1f}, {kf.x[1]:.1f})")

    except Exception as e:
        print(f"❌ Runtime Error: {e}")
    finally:
        imu_sensor.destroy()
        gps_actor.destroy()
        print("🛑 ROI Sensors destroyed.")

    # 6. Visualization
    if not fused_poses:
        print("⚠️ No data collected.")
        return

    fused = np.array(fused_poses)
    gps_only = np.array(gps_meas)

    plt.figure(figsize=(15, 6))
    
    # Subplot 1: Trajectory
    plt.subplot(121)
    if len(gps_only) > 0:
        plt.plot(gps_only[:,0], gps_only[:,1], 'g.', markersize=2, alpha=0.3, label='GPS Raw (Simulated)')
    plt.plot(fused[:,0], fused[:,1], 'b-', linewidth=2, label='EKF Fused')
    plt.legend()
    plt.title('2D Vehicle Trajectory Estimation')
    plt.xlabel('Global X (m)')
    plt.ylabel('Global Y (m)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')

    # Subplot 2: Error Analysis
    plt.subplot(122)
    min_len = min(len(fused), len(gps_only))
    if min_len > 0:
        error = np.linalg.norm(fused[:min_len] - gps_only[:min_len], axis=1)
        plt.plot(np.arange(min_len) * DT, error, 'r-', linewidth=1)
        plt.title(f'Position Deviation (Filter vs Noisy GPS)')
        plt.xlabel('Time (s)')
        plt.ylabel('Euclidean Distance (m)')
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(experiment_dir, 'week1_trajectory.png')
    plt.savefig(save_path)
    print(f"📊 Visualization saved: {save_path}")

if __name__ == "__main__":
    main()
