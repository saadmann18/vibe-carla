"""
TC-003: End-to-End Integration Tests
Test cases for SF-EKF-003 requirement (GPS + IMU Fusion)
Requires CARLA simulator running
"""

import pytest
import numpy as np
import carla
import time
import math
import random
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.week1_kalman_fusion.kalman_fusion import (
    state_transition,
    calculate_jacobian_F,
    measurement_function,
    calculate_jacobian_H,
    Sensor
)
from filterpy.kalman import ExtendedKalmanFilter


class TestEndToEndFusion:
    """TC-003: End-to-End Fusion Tests"""
    
    @classmethod
    def setup_class(cls):
        """Setup CARLA connection once for all tests"""
        try:
            cls.client = carla.Client('localhost', 2000)
            cls.client.set_timeout(10.0)
            cls.world = cls.client.get_world()
            print("✅ Connected to CARLA")
        except RuntimeError as e:
            pytest.skip(f"CARLA not available: {e}")
    
    def setup_method(self):
        """Setup for each test"""
        self.vehicle = None
        self.imu_sensor = None
        self.gps_actor = None
    
    def teardown_method(self):
        """Cleanup after each test"""
        if self.imu_sensor:
            self.imu_sensor.destroy()
        if self.gps_actor and self.gps_actor.is_alive:
            self.gps_actor.destroy()
        # Note: Don't destroy vehicle if it's the 'hero' from manual_control
    
    def _setup_sensors(self):
        """Helper to setup sensors on vehicle"""
        # Find hero vehicle
        actors = self.world.get_actors().filter('vehicle.*')
        for actor in actors:
            if actor.attributes.get('role_name') == 'hero':
                self.vehicle = actor
                break
        
        if not self.vehicle and len(actors) > 0:
            self.vehicle = actors[0]
        
        assert self.vehicle is not None, "No vehicle found. Run manual_control.py first"
        
        # Spawn sensors
        bp_lib = self.world.get_blueprint_library()
        imu_bp = bp_lib.find('sensor.other.imu')
        gps_bp = bp_lib.find('sensor.other.gnss')
        
        imu_actor = self.world.spawn_actor(imu_bp, carla.Transform(), attach_to=self.vehicle)
        self.gps_actor = self.world.spawn_actor(gps_bp, carla.Transform(), attach_to=self.vehicle)
        
        self.imu_sensor = Sensor(imu_actor)
        
        # Wait for initial data
        for _ in range(10):
            self.world.wait_for_tick()
            if self.imu_sensor.get_data():
                break
    
    def _run_ekf_fusion(self, num_steps=100, sigma_gps=2.0):
        """
        Run EKF fusion for specified steps
        Returns: (gps_rmse, ekf_rmse, fused_poses, gps_meas)
        """
        # Initialize EKF
        kf = ExtendedKalmanFilter(dim_x=6, dim_z=4)
        kf.P *= 10.0
        kf.R = np.diag([sigma_gps**2, sigma_gps**2, 0.1, 0.05])
        kf.Q = np.eye(6) * 0.1
        
        # Initialize state
        start_gps = self.gps_actor.get_location()
        start_imu = self.imu_sensor.get_data()
        start_yaw = math.radians(start_imu.compass)
        kf.x = np.array([start_gps.x, start_gps.y, 0, 0, start_yaw, 0])
        
        fused_poses = []
        gps_meas = []
        true_poses = []
        dt = 0.05
        
        for i in range(num_steps):
            self.world.wait_for_tick()
            
            # Get true position
            true_loc = self.gps_actor.get_location()
            true_poses.append([true_loc.x, true_loc.y])
            
            # Simulate noisy GPS
            z_gps_x = true_loc.x + random.gauss(0, sigma_gps)
            z_gps_y = true_loc.y + random.gauss(0, sigma_gps)
            gps_meas.append([z_gps_x, z_gps_y])
            
            # Get IMU
            imu_data = self.imu_sensor.get_data()
            if not imu_data:
                continue
            
            z_yaw = math.radians(imu_data.compass)
            z_omega = imu_data.gyroscope.z
            
            # EKF Predict
            kf.F = calculate_jacobian_F(kf.x, dt)
            kf.x = state_transition(kf.x, dt)
            kf.predict()
            
            # EKF Update
            z = np.array([z_gps_x, z_gps_y, z_yaw, z_omega])
            kf.update(z, calculate_jacobian_H, measurement_function)
            
            fused_poses.append(kf.x[:2].copy())
        
        # Calculate RMSE
        true_poses = np.array(true_poses)
        fused_poses = np.array(fused_poses)
        gps_meas = np.array(gps_meas)
        
        min_len = min(len(true_poses), len(fused_poses), len(gps_meas))
        
        gps_errors = np.linalg.norm(gps_meas[:min_len] - true_poses[:min_len], axis=1)
        ekf_errors = np.linalg.norm(fused_poses[:min_len] - true_poses[:min_len], axis=1)
        
        gps_rmse = np.sqrt(np.mean(gps_errors**2))
        ekf_rmse = np.sqrt(np.mean(ekf_errors**2))
        
        return gps_rmse, ekf_rmse, fused_poses, gps_meas
    
    def test_tc_003_1_straight_drive(self):
        """
        TC-003.1: Straight drive scenario
        Duration: 100 steps
        Pass Criteria: EKF RMSE ≥30% better than GPS RMSE
        Coverage: SF-EKF-003
        """
        self._setup_sensors()
        
        print("\n🚗 TC-003.1: Drive straight for best results...")
        gps_rmse, ekf_rmse, _, _ = self._run_ekf_fusion(num_steps=100, sigma_gps=2.0)
        
        improvement = ((gps_rmse - ekf_rmse) / gps_rmse) * 100
        
        print(f"GPS RMSE: {gps_rmse:.3f} m")
        print(f"EKF RMSE: {ekf_rmse:.3f} m")
        print(f"Improvement: {improvement:.1f}%")
        
        assert improvement >= 30.0, \
            f"Expected ≥30% improvement, got {improvement:.1f}%"
        
        print(f"✅ TC-003.1 PASS: {improvement:.1f}% improvement (≥30% required)")
    
    def test_tc_003_2_turns_and_curves(self):
        """
        TC-003.2: Turns + curves scenario
        Duration: 100 steps
        Pass Criteria: EKF RMSE ≥25% better than GPS RMSE
        Coverage: SF-EKF-003
        """
        self._setup_sensors()
        
        print("\n🔄 TC-003.2: Drive with turns and curves...")
        gps_rmse, ekf_rmse, _, _ = self._run_ekf_fusion(num_steps=100, sigma_gps=2.0)
        
        improvement = ((gps_rmse - ekf_rmse) / gps_rmse) * 100
        
        print(f"GPS RMSE: {gps_rmse:.3f} m")
        print(f"EKF RMSE: {ekf_rmse:.3f} m")
        print(f"Improvement: {improvement:.1f}%")
        
        assert improvement >= 25.0, \
            f"Expected ≥25% improvement, got {improvement:.1f}%"
        
        print(f"✅ TC-003.2 PASS: {improvement:.1f}% improvement (≥25% required)")
    
    def test_tc_003_3_gps_noise_only(self):
        """
        TC-003.3: GPS noise only scenario (high noise)
        Duration: 100 steps
        Pass Criteria: EKF RMSE ≥30% better than GPS RMSE
        Coverage: SF-EKF-003
        """
        self._setup_sensors()
        
        print("\n📡 TC-003.3: Testing with high GPS noise...")
        # Use higher GPS noise to test filter robustness
        gps_rmse, ekf_rmse, _, _ = self._run_ekf_fusion(num_steps=100, sigma_gps=3.0)
        
        improvement = ((gps_rmse - ekf_rmse) / gps_rmse) * 100
        
        print(f"GPS RMSE: {gps_rmse:.3f} m")
        print(f"EKF RMSE: {ekf_rmse:.3f} m")
        print(f"Improvement: {improvement:.1f}%")
        
        assert improvement >= 30.0, \
            f"Expected ≥30% improvement, got {improvement:.1f}%"
        
        print(f"✅ TC-003.3 PASS: {improvement:.1f}% improvement (≥30% required)")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
