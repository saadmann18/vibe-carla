# src/week2_radar_lidar_fusion/lidar_utils.py
import numpy as np

def project_lidar_to_bev(lidar_points, height_range=(-0.5, 2.0)):
    """Project LiDAR 3D points to 2D BEV (Bird's Eye View)."""
    if len(lidar_points) == 0:
        return np.zeros((0, 2))
    
    # Filter by height (ground + vehicle, remove sky/ground clutter)
    mask = (lidar_points[:, 2] >= height_range[0]) & \
           (lidar_points[:, 2] <= height_range[1])
    filtered = lidar_points[mask]
    
    # Project to BEV (drop z coordinate)
    return filtered[:, :2] if len(filtered) > 0 else np.zeros((0, 2))
