# src/week2_radar_lidar_fusion/radar_utils.py
import numpy as np

def polar_to_cartesian(detections):
    """Convert CARLA radar detections from polar to Cartesian BEV."""
    radar_points = []
    for det in detections:
        r = det.depth                    # range [m]
        theta = det.azimuth              # azimuth [rad]
        v_rel = det.velocity             # velocity [m/s]
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        vx = v_rel * np.cos(theta)
        vy = v_rel * np.sin(theta)
        
        radar_points.append([x, y, vx, vy])
    
    return np.array(radar_points) if radar_points else np.zeros((0, 4))
