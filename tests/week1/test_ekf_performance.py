"""
TC-004, TC-005, TC-006: Performance and Validation Tests
Test cases for SF-EKF-004, SF-EKF-005, SF-EKF-006 requirements
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


class TestPerformanceMetrics:
    """TC-004: Performance Metrics Tests"""
    
    @classmethod
    def setup_class(cls):
        """Setup CARLA connection"""
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
        
        # Find and setup sensors
        actors = self.world.get_actors().filter('vehicle.*')
        for actor in actors:
            if actor.attributes.get('role_name') == 'hero':
                self.vehicle = actor
                break
        
        if not self.vehicle and len(actors) > 0:
            self.vehicle = actors[0]
        
        assert self.vehicle is not None, "No vehicle found"
        
        bp_lib = self.world.get_blueprint_library()
        imu_bp = bp_lib.find('sensor.other.imu')
        gps_bp = bp_lib.find('sensor.other.gnss')
        
        imu_actor = self.world.spawn_actor(imu_bp, carla.Transform(), attach_to=self.vehicle)
        self.gps_actor = self.world.spawn_actor(gps_bp, carla.Transform(), attach_to=self.vehicle)
        
        self.imu_sensor = Sensor(imu_actor)
        
        # Wait for data
        for _ in range(10):
            self.world.wait_for_tick()
            if self.imu_sensor.get_data():
                break
    
    def teardown_method(self):
        """Cleanup"""
        if self.imu_sensor:
            self.imu_sensor.destroy()
        if self.gps_actor and self.gps_actor.is_alive:
            self.gps_actor.destroy()
    
    def _run_performance_test(self, num_steps=1000):
        """Run EKF for performance measurement"""
        kf = ExtendedKalmanFilter(dim_x=6, dim_z=4)
        kf.P *= 10.0
        kf.R = np.diag([4.0, 4.0, 0.1, 0.05])
        kf.Q = np.eye(6) * 0.1
        
        start_gps = self.gps_actor.get_location()
        start_imu = self.imu_sensor.get_data()
        start_yaw = math.radians(start_imu.compass)
        kf.x = np.array([start_gps.x, start_gps.y, 0, 0, start_yaw, 0])
        
        fused_poses = []
        gps_meas = []
        true_poses = []
        yaw_history = []
        dt = 0.05
        
        for i in range(num_steps):
            self.world.wait_for_tick()
            
            true_loc = self.gps_actor.get_location()
            true_poses.append([true_loc.x, true_loc.y])
            
            z_gps_x = true_loc.x + random.gauss(0, 2.0)
            z_gps_y = true_loc.y + random.gauss(0, 2.0)
            gps_meas.append([z_gps_x, z_gps_y])
            
            imu_data = self.imu_sensor.get_data()
            if not imu_data:
                continue
            
            z_yaw = math.radians(imu_data.compass)
            z_omega = imu_data.gyroscope.z
            
            kf.F = calculate_jacobian_F(kf.x, dt)
            kf.x = state_transition(kf.x, dt)
            kf.predict()
            
            z = np.array([z_gps_x, z_gps_y, z_yaw, z_omega])
            kf.update(z, calculate_jacobian_H, measurement_function)
            
            fused_poses.append(kf.x[:2].copy())
            yaw_history.append(kf.x[4])
        
        return np.array(true_poses), np.array(fused_poses), np.array(gps_meas), np.array(yaw_history)
    
    def test_tc_004_1_rmse_improvement(self):
        """
        TC-004.1: RMSE Improvement ≥30%
        Metric: Fused RMSE < GPS RMSE by ≥30%
        Coverage: SF-EKF-004
        """
        print("\n📊 TC-004.1: Measuring RMSE improvement over 1000 steps...")
        true_poses, fused_poses, gps_meas, _ = self._run_performance_test(1000)
        
        min_len = min(len(true_poses), len(fused_poses), len(gps_meas))
        
        gps_errors = np.linalg.norm(gps_meas[:min_len] - true_poses[:min_len], axis=1)
        ekf_errors = np.linalg.norm(fused_poses[:min_len] - true_poses[:min_len], axis=1)
        
        gps_rmse = np.sqrt(np.mean(gps_errors**2))
        ekf_rmse = np.sqrt(np.mean(ekf_errors**2))
        
        improvement_pct = ((gps_rmse - ekf_rmse) / gps_rmse) * 100
        
        print(f"GPS RMSE: {gps_rmse:.3f} m")
        print(f"EKF RMSE: {ekf_rmse:.3f} m")
        print(f"Improvement: {improvement_pct:.1f}%")
        
        assert improvement_pct >= 30.0, \
            f"Expected ≥30% improvement, got {improvement_pct:.1f}%"
        
        print(f"✅ TC-004.1 PASS: {improvement_pct:.1f}% improvement")
    
    def test_tc_004_2_max_position_error(self):
        """
        TC-004.2: Max Position Error <5m
        Metric: max(|fused - true|) < 5.0m
        Coverage: SF-EKF-004
        """
        print("\n📏 TC-004.2: Measuring maximum position error...")
        true_poses, fused_poses, _, _ = self._run_performance_test(1000)
        
        min_len = min(len(true_poses), len(fused_poses))
        errors = np.linalg.norm(fused_poses[:min_len] - true_poses[:min_len], axis=1)
        max_error = np.max(errors)
        
        print(f"Max Position Error: {max_error:.3f} m")
        
        assert max_error < 5.0, \
            f"Expected <5.0m, got {max_error:.3f}m"
        
        print(f"✅ TC-004.2 PASS: {max_error:.3f} m < 5.0 m")
    
    def test_tc_004_3_yaw_drift(self):
        """
        TC-004.3: Yaw Drift <10° (1000 steps)
        Metric: |yaw_final - yaw_initial| < 10°
        Coverage: SF-EKF-004
        """
        print("\n🧭 TC-004.3: Measuring yaw drift over 1000 steps...")
        _, _, _, yaw_history = self._run_performance_test(1000)
        
        yaw_initial = yaw_history[0]
        yaw_final = yaw_history[-1]
        
        # Calculate drift in degrees
        yaw_drift_rad = abs(yaw_final - yaw_initial)
        yaw_drift_deg = math.degrees(yaw_drift_rad)
        
        print(f"Yaw Drift: {yaw_drift_deg:.2f}°")
        
        assert yaw_drift_deg < 10.0, \
            f"Expected <10°, got {yaw_drift_deg:.2f}°"
        
        print(f"✅ TC-004.3 PASS: {yaw_drift_deg:.2f}° < 10°")


class TestStressTest:
    """TC-005: Stress Test (10Hz Processing)"""
    
    @classmethod
    def setup_class(cls):
        """Setup CARLA connection"""
        try:
            cls.client = carla.Client('localhost', 2000)
            cls.client.set_timeout(10.0)
            cls.world = cls.client.get_world()
        except RuntimeError as e:
            pytest.skip(f"CARLA not available: {e}")
    
    def test_tc_005_1_process_10hz(self):
        """
        TC-005.1: Process CARLA data @ 10Hz
        Verify system can handle 10Hz update rate
        Coverage: SF-EKF-005
        """
        print("\n⚡ TC-005.1: Stress testing at 10Hz...")
        
        # Find vehicle
        actors = self.world.get_actors().filter('vehicle.*')
        vehicle = None
        for actor in actors:
            if actor.attributes.get('role_name') == 'hero':
                vehicle = actor
                break
        if not vehicle and len(actors) > 0:
            vehicle = actors[0]
        
        assert vehicle is not None
        
        # Setup sensors
        bp_lib = self.world.get_blueprint_library()
        imu_bp = bp_lib.find('sensor.other.imu')
        gps_bp = bp_lib.find('sensor.other.gnss')
        
        imu_actor = self.world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
        gps_actor = self.world.spawn_actor(gps_bp, carla.Transform(), attach_to=vehicle)
        
        imu_sensor = Sensor(imu_actor)
        
        # Wait for data
        for _ in range(10):
            self.world.wait_for_tick()
            if imu_sensor.get_data():
                break
        
        # Initialize EKF
        kf = ExtendedKalmanFilter(dim_x=6, dim_z=4)
        kf.P *= 10.0
        kf.R = np.diag([4.0, 4.0, 0.1, 0.05])
        kf.Q = np.eye(6) * 0.1
        
        start_gps = gps_actor.get_location()
        start_imu = imu_sensor.get_data()
        kf.x = np.array([start_gps.x, start_gps.y, 0, 0, math.radians(start_imu.compass), 0])
        
        # Run at 10Hz (0.1s per iteration)
        num_iterations = 100
        processing_times = []
        
        for i in range(num_iterations):
            start_time = time.time()
            
            self.world.wait_for_tick()
            
            true_loc = gps_actor.get_location()
            z_gps_x = true_loc.x + random.gauss(0, 2.0)
            z_gps_y = true_loc.y + random.gauss(0, 2.0)
            
            imu_data = imu_sensor.get_data()
            if not imu_data:
                continue
            
            z_yaw = math.radians(imu_data.compass)
            z_omega = imu_data.gyroscope.z
            
            dt = 0.1  # 10Hz
            kf.F = calculate_jacobian_F(kf.x, dt)
            kf.x = state_transition(kf.x, dt)
            kf.predict()
            
            z = np.array([z_gps_x, z_gps_y, z_yaw, z_omega])
            kf.update(z, calculate_jacobian_H, measurement_function)
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
        
        # Cleanup
        imu_sensor.destroy()
        gps_actor.destroy()
        
        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)
        
        print(f"Avg Processing Time: {avg_processing_time*1000:.2f} ms")
        print(f"Max Processing Time: {max_processing_time*1000:.2f} ms")
        print(f"Target: <100 ms (10Hz)")
        
        # Should process faster than 100ms to maintain 10Hz
        assert max_processing_time < 0.1, \
            f"Processing too slow: {max_processing_time*1000:.2f} ms"
        
        print(f"✅ TC-005.1 PASS: Can process at 10Hz")


class TestPositionErrorValidation:
    """TC-006: Position Error Validation"""
    
    @classmethod
    def setup_class(cls):
        """Setup CARLA connection"""
        try:
            cls.client = carla.Client('localhost', 2000)
            cls.client.set_timeout(10.0)
            cls.world = cls.client.get_world()
        except RuntimeError as e:
            pytest.skip(f"CARLA not available: {e}")
    
    def test_tc_006_1_position_error_threshold(self):
        """
        TC-006.1: Position error σ < 2.0m (1000 steps)
        Metric: std(|fused - true|) < 2.0m
        Coverage: SF-EKF-006
        """
        print("\n🎯 TC-006.1: Validating position error over 1000 steps...")
        
        # Find vehicle and setup sensors
        actors = self.world.get_actors().filter('vehicle.*')
        vehicle = None
        for actor in actors:
            if actor.attributes.get('role_name') == 'hero':
                vehicle = actor
                break
        if not vehicle and len(actors) > 0:
            vehicle = actors[0]
        
        assert vehicle is not None
        
        bp_lib = self.world.get_blueprint_library()
        imu_bp = bp_lib.find('sensor.other.imu')
        gps_bp = bp_lib.find('sensor.other.gnss')
        
        imu_actor = self.world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
        gps_actor = self.world.spawn_actor(gps_bp, carla.Transform(), attach_to=vehicle)
        
        imu_sensor = Sensor(imu_actor)
        
        for _ in range(10):
            self.world.wait_for_tick()
            if imu_sensor.get_data():
                break
        
        # Initialize EKF
        kf = ExtendedKalmanFilter(dim_x=6, dim_z=4)
        kf.P *= 10.0
        kf.R = np.diag([4.0, 4.0, 0.1, 0.05])
        kf.Q = np.eye(6) * 0.1
        
        start_gps = gps_actor.get_location()
        start_imu = imu_sensor.get_data()
        kf.x = np.array([start_gps.x, start_gps.y, 0, 0, math.radians(start_imu.compass), 0])
        
        fused_poses = []
        true_poses = []
        
        for i in range(1000):
            self.world.wait_for_tick()
            
            true_loc = gps_actor.get_location()
            true_poses.append([true_loc.x, true_loc.y])
            
            z_gps_x = true_loc.x + random.gauss(0, 2.0)
            z_gps_y = true_loc.y + random.gauss(0, 2.0)
            
            imu_data = imu_sensor.get_data()
            if not imu_data:
                continue
            
            z_yaw = math.radians(imu_data.compass)
            z_omega = imu_data.gyroscope.z
            
            dt = 0.05
            kf.F = calculate_jacobian_F(kf.x, dt)
            kf.x = state_transition(kf.x, dt)
            kf.predict()
            
            z = np.array([z_gps_x, z_gps_y, z_yaw, z_omega])
            kf.update(z, calculate_jacobian_H, measurement_function)
            
            fused_poses.append(kf.x[:2].copy())
        
        # Cleanup
        imu_sensor.destroy()
        gps_actor.destroy()
        
        # Calculate error statistics
        true_poses = np.array(true_poses)
        fused_poses = np.array(fused_poses)
        
        min_len = min(len(true_poses), len(fused_poses))
        errors = np.linalg.norm(fused_poses[:min_len] - true_poses[:min_len], axis=1)
        
        error_std = np.std(errors)
        error_mean = np.mean(errors)
        
        print(f"Mean Error: {error_mean:.3f} m")
        print(f"Std Error: {error_std:.3f} m")
        
        assert error_std < 2.0, \
            f"Expected σ < 2.0m, got {error_std:.3f}m"
        
        print(f"✅ TC-006.1 PASS: σ = {error_std:.3f} m < 2.0 m")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
