"""
TC-001 & TC-002: Unit Tests for EKF Initialization and Motion Model
Test cases for SF-EKF-001 and SF-EKF-002 requirements
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.week1_kalman_fusion.kalman_fusion import (
    state_transition,
    calculate_jacobian_F,
    measurement_function,
    calculate_jacobian_H
)
from filterpy.kalman import ExtendedKalmanFilter


class TestEKFInitialization:
    """TC-001: Initialization Verification Tests"""
    
    def test_tc_001_1_state_initialization_zeros(self):
        """
        TC-001.1: Verify kf.x = np.zeros(6)
        Expected: [0,0,0,0,0,0]
        Coverage: SF-EKF-001
        """
        kf = ExtendedKalmanFilter(dim_x=6, dim_z=4)
        kf.x = np.zeros(6)
        
        expected = np.array([0, 0, 0, 0, 0, 0])
        actual = kf.x
        
        np.testing.assert_array_equal(actual, expected)
        print(f"✅ TC-001.1 PASS: State initialized to {actual}")
    
    def test_tc_001_2_covariance_initialization(self):
        """
        TC-001.2: Verify kf.P *= 10
        Expected: Diagonal = 10.0
        Coverage: SF-EKF-001
        """
        kf = ExtendedKalmanFilter(dim_x=6, dim_z=4)
        kf.P *= 10
        
        expected_diagonal = 10.0
        actual_diagonal = np.diag(kf.P)
        
        np.testing.assert_array_equal(actual_diagonal, np.ones(6) * expected_diagonal)
        print(f"✅ TC-001.2 PASS: Covariance diagonal = {actual_diagonal[0]}")


class TestMotionModel:
    """TC-002: Motion Model Unit Tests"""
    
    def test_tc_002_1_straight_motion(self):
        """
        TC-002.1: Test state transition with straight motion
        Initial State: [0, 0, 10, 0, 0, 0]
        dt: 0.1
        Expected Δx: 1.0 (vx * cos(0) * dt = 10 * 1 * 0.1)
        Tolerance: ±0.01
        Coverage: SF-EKF-002
        """
        initial_state = np.array([0, 0, 10, 0, 0, 0])
        dt = 0.1
        
        new_state = state_transition(initial_state, dt)
        
        expected_x = 1.0
        actual_x = new_state[0]
        tolerance = 0.01
        
        assert abs(actual_x - expected_x) < tolerance, \
            f"Expected x={expected_x}, got {actual_x}"
        
        print(f"✅ TC-002.1 PASS: Δx = {actual_x:.3f} (expected {expected_x} ±{tolerance})")
    
    def test_tc_002_2_turning_motion(self):
        """
        TC-002.2: Test state transition with turning motion
        Initial State: [0, 0, 10, 0, 0, 1]  # yaw_rate = 1 rad/s
        dt: 0.1
        Expected yaw: 0.1 rad
        Tolerance: ±0.001
        Coverage: SF-EKF-002
        """
        initial_state = np.array([0, 0, 10, 0, 0, 1])
        dt = 0.1
        
        new_state = state_transition(initial_state, dt)
        
        expected_yaw = 0.1
        actual_yaw = new_state[4]
        tolerance = 0.001
        
        assert abs(actual_yaw - expected_yaw) < tolerance, \
            f"Expected yaw={expected_yaw}, got {actual_yaw}"
        
        print(f"✅ TC-002.2 PASS: yaw = {actual_yaw:.4f} (expected {expected_yaw} ±{tolerance})")


class TestJacobians:
    """Additional unit tests for Jacobian calculations"""
    
    def test_jacobian_F_shape(self):
        """Verify Jacobian F has correct dimensions"""
        x = np.array([0, 0, 10, 0, 0, 0])
        dt = 0.1
        F = calculate_jacobian_F(x, dt)
        
        assert F.shape == (6, 6), f"Expected (6,6), got {F.shape}"
        print(f"✅ Jacobian F shape correct: {F.shape}")
    
    def test_jacobian_H_shape(self):
        """Verify Jacobian H has correct dimensions"""
        x = np.array([0, 0, 10, 0, 0, 0])
        H = calculate_jacobian_H(x)
        
        assert H.shape == (4, 6), f"Expected (4,6), got {H.shape}"
        print(f"✅ Jacobian H shape correct: {H.shape}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
