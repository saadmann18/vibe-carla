"""
Unit Tests for Week 2 Radar-LiDAR Fusion
Testing individual modules: radar_utils, lidar_utils, tracking, and metrics
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.week2_radar_lidar_fusion.radar_utils import polar_to_cartesian
from src.week2_radar_lidar_fusion.lidar_utils import project_lidar_to_bev
from src.week2_radar_lidar_fusion.tracking import (
    dbscan_cluster, 
    compute_centroids, 
    fuse_radar_lidar,
    KalmanTrack
)
from src.week2_radar_lidar_fusion.metrics import compute_track_stats

class TestRadarUtils:
    def test_polar_to_cartesian_basic(self):
        """Test conversion of a single radar detection"""
        # Mock detection object with required attributes
        class MockDet:
            def __init__(self, depth, azimuth, velocity):
                self.depth = depth
                self.azimuth = azimuth
                self.velocity = velocity
        
        # 10m ahead, 0 rad azimuth, 5m/s velocity
        detections = [MockDet(10.0, 0.0, 5.0)]
        result = polar_to_cartesian(detections)
        
        assert result.shape == (1, 4)
        assert np.allclose(result[0], [10.0, 0.0, 5.0, 0.0])

class TestLidarUtils:
    def test_project_lidar_to_bev_filtering(self):
        """Test height filtering and projection"""
        # Points: [x, y, z]
        points = np.array([
            [10, 0, 1],    # Inside range
            [10, 5, 3],    # Above range
            [10, -5, -2],  # Below range
        ])
        
        bev = project_lidar_to_bev(points, height_range=(-0.5, 2.0))
        
        assert len(bev) == 1
        assert np.allclose(bev[0], [10.0, 0.0])

class TestTrackingModules:
    def test_dbscan_clustering(self):
        """Test grouping of nearby points"""
        points = np.array([
            [10, 0], [10.1, 0.1], [9.9, -0.1], # Cluster 0
            [50, 50], [50.2, 50.1]             # Cluster 1
        ])
        
        labels = dbscan_cluster(points, eps=1.0, min_samples=2)
        
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4]
        assert labels[0] != labels[3]

    def test_fuse_radar_lidar_weighting(self):
        """Test 70/30 fusion weighting"""
        radar_xy = np.array([10.0, 0.0])
        lidar_points = np.array([
            [11.0, 0.0],
            [10.8, 0.2]
        ])
        
        fused = fuse_radar_lidar(radar_xy, lidar_points, search_radius=2.0)
        
        # Lidar mean is [10.9, 0.1]
        # Fusion: 0.7 * [10.9, 0.1] + 0.3 * [10.0, 0.0]
        # X: 0.7*10.9 + 0.3*10.0 = 7.63 + 3.0 = 10.63
        # Y: 0.7*0.1 + 0.3*0.0 = 0.07
        expected = np.array([10.63, 0.07])
        
        assert np.allclose(fused, expected)

    def test_kalman_track_init(self):
        """Test tracker initialization"""
        track = KalmanTrack(1, 10.0, 0.0, 5.0, 1.0)
        assert track.track_id == 1
        assert track.age == 1
        assert np.allclose(track.get_state(), [10.0, 0.0, 5.0, 1.0])

class TestMetrics:
    def test_track_stats(self):
        """Test basic statistics calculation"""
        class MockTrack:
            def __init__(self, age):
                self.age = age
        
        tracks = [MockTrack(10), MockTrack(20)]
        stats = compute_track_stats(tracks)
        
        assert stats['num_tracks'] == 2
        assert stats['mean_age'] == 15.0
        assert stats['max_age'] == 20

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
