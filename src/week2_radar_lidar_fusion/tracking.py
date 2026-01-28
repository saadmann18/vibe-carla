# src/week2_radar_lidar_fusion/tracking.py
import numpy as np
from sklearn.cluster import DBSCAN
from filterpy.kalman import KalmanFilter

class KalmanTrack:
    """Single object tracking with Kalman filter."""
    def __init__(self, track_id, x, y, vx, vy):
        self.track_id = track_id
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State: [x, y, vx, vy]
        self.kf.x = np.array([[x], [y], [vx], [vy]])
        
        # Process noise (motion uncertainty)
        self.kf.Q = np.eye(4) * 0.1
        
        # Measurement noise
        self.kf.R = np.diag([1.0, 1.0])
        
        # Observation matrix (measure x, y only)
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        
        # Initial covariance
        self.kf.P *= 10
        
        self.age = 1
        self.consecutive_misses = 0
    
    def predict(self, dt):
        """Predict next state (constant velocity model)."""
        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.predict()
    
    def update(self, measurement_xy):
        """Update state with new measurement."""
        z = np.array(measurement_xy).reshape(2, 1)
        self.kf.update(z)
        self.consecutive_misses = 0
        self.age += 1
    
    def get_state(self):
        """Return current state [x, y, vx, vy]."""
        return self.kf.x.flatten()

def dbscan_cluster(points_xy, eps=1.0, min_samples=3):
    """Cluster radar BEV points using DBSCAN."""
    if len(points_xy) < min_samples:
        return np.array([-1] * len(points_xy))
    
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    return clusterer.fit_predict(points_xy)

def compute_centroids(radar_points, cluster_ids):
    """Compute cluster centroid positions and velocities."""
    unique_clusters = set(cluster_ids)
    if -1 in unique_clusters:
        unique_clusters.remove(-1)  # Remove noise
    
    centroids = []
    for cluster_id in sorted(unique_clusters):
        mask = cluster_ids == cluster_id
        cluster_points = radar_points[mask]
        
        x_c, y_c, vx_c, vy_c = np.mean(cluster_points, axis=0)
        centroids.append([x_c, y_c, vx_c, vy_c])
    
    return centroids

def kalman_predict(tracks, dt):
    """Predict all tracks forward by dt."""
    for track in tracks:
        track.predict(dt)
    return tracks

def kalman_update(tracks, centroids, max_distance=3.0):
    """Associate centroids to tracks and update (simple nearest-neighbor)."""
    matched_centroids = set()
    
    for track in tracks:
        state_xy = track.get_state()[:2]
        
        # Find nearest centroid
        if centroids:
            distances = [np.linalg.norm(np.array(c[:2]) - state_xy) for c in centroids]
            nearest_idx = np.argmin(distances)
            
            if distances[nearest_idx] < max_distance:
                track.update(centroids[nearest_idx][:2])
                matched_centroids.add(nearest_idx)
            else:
                track.consecutive_misses += 1
        else:
            track.consecutive_misses += 1
    
    # Create new tracks for unmatched centroids
    next_track_id = max([t.track_id for t in tracks]) + 1 if tracks else 0
    for i, centroid in enumerate(centroids):
        if i not in matched_centroids:
            new_track = KalmanTrack(next_track_id, centroid[0], centroid[1],
                                     centroid[2], centroid[3])
            tracks.append(new_track)
            next_track_id += 1
    
    # Remove stale tracks
    tracks = [t for t in tracks if t.consecutive_misses < 5]
    
    return tracks

def fuse_radar_lidar(radar_xy, lidar_points, search_radius=0.5):
    """Fuse radar centroid with LiDAR support for position refinement."""
    if len(lidar_points) == 0:
        return radar_xy
    
    # Find LiDAR points within search radius
    distances = np.linalg.norm(lidar_points - radar_xy, axis=1)
    nearby_mask = distances < search_radius
    
    if np.sum(nearby_mask) == 0:
        return radar_xy  # No LiDAR support
    
    # Weighted fusion: 70% LiDAR geometry, 30% Radar centroid
    lidar_mean = np.mean(lidar_points[nearby_mask], axis=0)
    fused_xy = 0.7 * lidar_mean + 0.3 * radar_xy
    
    return fused_xy
