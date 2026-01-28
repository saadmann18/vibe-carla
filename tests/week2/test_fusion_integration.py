"""
Integration Tests for Week 2 Radar-LiDAR Fusion
Requires CARLA simulator running
"""

import pytest
import carla
import numpy as np
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.week2_radar_lidar_fusion.radar_lidar_fusion import (
    radar_callback,
    lidar_callback,
    lidar_data_to_array,
    radar_data_to_detections
)

class TestFusionIntegration:
    """End-to-end integration tests with CARLA"""
    
    @classmethod
    def setup_class(cls):
        """Setup CARLA connection"""
        try:
            cls.client = carla.Client('localhost', 2000)
            cls.client.set_timeout(10.0)
            cls.world = cls.client.get_world()
            print("✅ Connected to CARLA")
        except Exception as e:
            pytest.skip(f"CARLA not available: {e}")
            
    def setup_method(self):
        """Setup vehicle and sensors for each test"""
        self.vehicle = None
        self.radar = None
        self.lidar = None
        
        # Spawn vehicle at an empty spot
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        
        for spawn_point in spawn_points:
            try:
                self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                break
            except:
                continue
        
        assert self.vehicle is not None, "Could not spawn vehicle for test"

    def teardown_method(self):
        """Cleanup after each test"""
        if self.radar: self.radar.destroy()
        if self.lidar: self.lidar.destroy()
        if self.vehicle: self.vehicle.destroy()

    def test_sensor_setup_and_data_flow(self):
        """Verify sensors attach and produce data that callbacks capture"""
        blueprint_library = self.world.get_blueprint_library()
        
        # Attach Radar
        radar_bp = blueprint_library.find('sensor.other.radar')
        self.radar = self.world.spawn_actor(radar_bp, carla.Transform(), attach_to=self.vehicle)
        self.radar.listen(radar_callback)
        
        # Attach LiDAR
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        self.lidar = self.world.spawn_actor(lidar_bp, carla.Transform(), attach_to=self.vehicle)
        self.lidar.listen(lidar_callback)
        
        # Wait for data
        data_received = False
        for _ in range(20):
            self.world.tick()
            time.sleep(0.1)
            
            # Check if queues were populated via callbacks
            from src.week2_radar_lidar_fusion.radar_lidar_fusion import radar_queue, lidar_queue
            if radar_queue is not None and lidar_queue is not None:
                data_received = True
                break
                
        assert data_received, "Timed out waiting for sensor data"
        
        # Verify data formats
        from src.week2_radar_lidar_fusion.radar_lidar_fusion import radar_queue, lidar_queue
        
        radar_detections = radar_data_to_detections(radar_queue)
        assert isinstance(radar_detections, list)
        
        lidar_points = lidar_data_to_array(lidar_queue)
        assert isinstance(lidar_points, np.ndarray)
        assert lidar_points.shape[1] == 3 # (x, y, z)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
