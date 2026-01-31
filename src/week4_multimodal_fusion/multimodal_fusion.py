# src/week4_multimodal_fusion/multimodal_fusion.py
"""Bayesian multi-modal sensor fusion."""
import numpy as np

class MultiModalFusion:
    """Fuse Camera + LiDAR + Radar measurements."""
    
    def __init__(self):
        # Sensor confidence weights (tunable)
        self.weights = {
            'camera': 0.7,   # Good for classification
            'lidar': 0.9,    # Best for geometry
            'radar': 0.8     # Best for velocity
        }
    
    def fuse_position(self, camera_pos, lidar_pos, radar_pos):
        """
        Bayesian fusion of position estimates.
        
        Args:
            camera_pos: [x, y] from 2D→3D projection (low confidence)
            lidar_pos: [x, y, z] from clustering (high confidence)
            radar_pos: [x, y] from radar centroid (medium confidence)
        
        Returns:
            fused_pos: [x, y, z] weighted average
        """
        # Pad camera/radar with z=0 if needed
        if len(camera_pos) == 2:
            camera_pos = np.append(camera_pos, 0)
        if len(radar_pos) == 2:
            radar_pos = np.append(radar_pos, 0)
        
        # Weighted fusion
        w_cam = self.weights['camera']
        w_lidar = self.weights['lidar']
        w_radar = self.weights['radar']
        
        total_weight = w_cam + w_lidar + w_radar
        
        fused = (w_cam * camera_pos + w_lidar * lidar_pos + w_radar * radar_pos) / total_weight
        return fused
    
    def compute_track_confidence(self, sensor_flags):
        """
        Compute overall track confidence based on which sensors contributed.
        
        Args:
            sensor_flags: dict {'camera': True, 'lidar': True, 'radar': False}
        
        Returns:
            confidence: [0.0, 1.0]
        """
        active_confidences = [self.weights[s] for s, active in sensor_flags.items() if active]
        
        if not active_confidences:
            return 0.1
        
        # Average of active sensors
        return np.mean(active_confidences)
