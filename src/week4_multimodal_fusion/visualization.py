# src/week4_multimodal_fusion/visualization.py
"""Multi-modal BEV visualization."""
import matplotlib.pyplot as plt
import numpy as np
import cv2

def project_lidar_to_camera(image, lidar_points, sensor_proj, max_dist=50.0):
    """
    Overlay LiDAR points onto camera image.
    
    Args:
        image: HxWx3 uint8 array
        lidar_points: Nx3 world coordinates
        sensor_proj: SensorProjection instance
        max_dist: Max distance to plot
    """
    uv, in_front = sensor_proj.project_3d_to_2d(lidar_points)
    
    # Filter points
    valid_mask = in_front & \
                 (uv[:, 0] >= 0) & (uv[:, 0] < image.shape[1]) & \
                 (uv[:, 1] >= 0) & (uv[:, 1] < image.shape[0])
                 
    valid_uv = uv[valid_mask].astype(int)
    
    overlay = image.copy()
    
    # Color based on depth (or simple color)
    # Simple Cyan dots
    for point in valid_uv:
        cv2.circle(overlay, tuple(point), 2, (0, 255, 255), -1)
        
    return overlay

def plot_multimodal_bev(camera_detections, lidar_clusters, radar_centroids, 
                        fused_tracks, filename="multimodal_bev.png"):
    """
    Plot BEV showing all 3 sensor modalities + fused tracks.
    
    Args:
        camera_detections: List of camera 2D bboxes (for legend only)
        lidar_clusters: Nx3 array of LiDAR cluster centroids
        radar_centroids: Mx2 array of radar centroids
        fused_tracks: List of dicts {'pos': [x,y,z], 'confidence': float, 'id': int}
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # LiDAR clusters (blue)
    if len(lidar_clusters) > 0:
        ax.scatter(lidar_clusters[:, 0], lidar_clusters[:, 1], 
                  c='cyan', s=100, alpha=0.6, marker='o', 
                  edgecolors='blue', linewidths=2, label='LiDAR Clusters')
    
    # Radar centroids (red)
    if len(radar_centroids) > 0:
        ax.scatter(radar_centroids[:, 0], radar_centroids[:, 1], 
                  c='red', s=80, alpha=0.7, marker='^', label='Radar Centroids')
    
    # Fused tracks (green with confidence-based size)
    for track in fused_tracks:
        pos = track['pos']
        conf = track['confidence']
        track_id = track['id']
        
        size = 50 + 150 * conf  # Bigger = higher confidence
        ax.scatter(pos[0], pos[1], c='lime', s=size, marker='*', 
                  edgecolors='darkgreen', linewidths=2)
        ax.text(pos[0] + 1, pos[1] + 1, f"T{track_id}\n{conf:.2f}", 
               fontsize=9, color='darkgreen', weight='bold')
    
    # Camera detections indicator (count only)
    if camera_detections:
        ax.text(0.02, 0.98, f"Camera: {len(camera_detections)} detections", 
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('X [m]', fontsize=13)
    ax.set_ylabel('Y [m]', fontsize=13)
    ax.set_title('Multi-Modal Fusion BEV: Camera + LiDAR + Radar', fontsize=15, weight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.set_xlim(-50, 100)
    ax.set_ylim(-50, 50)
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()
