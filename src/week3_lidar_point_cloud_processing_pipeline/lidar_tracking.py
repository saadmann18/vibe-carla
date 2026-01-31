# src/week3_lidar_point_cloud_processing_pipeline/lidar_tracking.py
import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanTrack:
    """Individual tracker for a single object in 3D."""
    def __init__(self, track_id, centroid):
        self.track_id = track_id
        # State: [x, y, z, vx, vy, vz]
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        
        self.kf.x = np.array([[centroid[0]], [centroid[1]], [centroid[2]], [0], [0], [0]])
        
        # Process noise
        self.kf.Q = np.eye(6) * 0.1
        
        # Measurement noise (from Lidar)
        self.kf.R = np.eye(3) * 0.5
        
        # Observation matrix (measure x, y, z only)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Initial covariance
        self.kf.P *= 10.0
        
        self.age = 1
        self.hits = 1
        self.consecutive_misses = 0
        
    def predict(self, dt):
        """Predict next state using constant velocity model."""
        self.kf.F = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1]
        ])
        self.kf.predict()
        self.age += 1
        
    def update(self, centroid):
        """Update filter state with new centroid measurement."""
        z = np.array(centroid).reshape(3, 1)
        self.kf.update(z)
        self.hits += 1
        self.consecutive_misses = 0
        
    def get_state(self):
        """Return state array [x, y, z, vx, vy, vz]."""
        return self.kf.x.flatten()

class LiDARTracker:
    """Manager for multiple object tracks."""
    def __init__(self, max_misses=5, min_hits=3, max_dist=2.0):
        self.tracks = []
        self.next_track_id = 0
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.max_dist = max_dist
        
    def update(self, centroids, dt):
        """Update tracker with new detections."""
        # 1. Predict existing tracks
        for track in self.tracks:
            track.predict(dt)
            
        # 2. Match centroids to tracks (Simple Nearest Neighbor)
        matched_centroids = set()
        matched_tracks = set()
        
        if len(self.tracks) > 0 and len(centroids) > 0:
            # Greedy matching
            for i, track in enumerate(self.tracks):
                track_pos = track.get_state()[:3]
                best_dist = self.max_dist
                best_idx = -1
                
                for j, centroid in enumerate(centroids):
                    if j in matched_centroids:
                        continue
                        
                    dist = np.linalg.norm(track_pos - centroid)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = j
                
                if best_idx != -1:
                    track.update(centroids[best_idx])
                    matched_centroids.add(best_idx)
                    matched_tracks.add(i)
                else:
                    track.consecutive_misses += 1
                    
        # 3. Handle unmatched tracks (already incremented misses above)
        # 4. Create new tracks for unmatched centroids
        for j, centroid in enumerate(centroids):
            if j not in matched_centroids:
                new_track = KalmanTrack(self.next_track_id, centroid)
                self.tracks.append(new_track)
                self.next_track_id += 1
                
        # 5. Filter/Cleanup tracks
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.consecutive_misses < self.max_misses]
        
        # Return confirmed tracks (active and enough hits)
        return [t for t in self.tracks if t.hits >= self.min_hits]
