# src/week4_multimodal_fusion/sensor_projection.py
"""2D Camera → 3D world projection and sensor association."""
import numpy as np

class SensorProjection:
    """Geometric projection and association between sensors."""
    
    def __init__(self, camera_intrinsics, camera_transform):
        """
        Args:
            camera_intrinsics: 3x3 camera matrix K
            camera_transform: CARLA Transform (camera pose in world)
        """
        self.K = camera_intrinsics
        self.camera_transform = camera_transform
    
    def project_2d_to_3d_ray(self, pixel_uv):
        """
        Project 2D pixel to 3D ray in world coordinates.
        
        Args:
            pixel_uv: [u, v] pixel coordinates
        
        Returns:
            ray_origin: [x, y, z] ray start in world frame
            ray_direction: [x, y, z] unit vector
        """
        u, v = pixel_uv
        
        # Lift to normalized image plane
        x_norm = (u - self.K[0, 2]) / self.K[0, 0]
        y_norm = (v - self.K[1, 2]) / self.K[1, 1]
        
        # Ray in camera frame (z forward, x right, y down)
        ray_camera = np.array([x_norm, y_norm, 1.0])
        ray_camera = ray_camera / np.linalg.norm(ray_camera)
        
        # Transform to world frame
        rotation = self.camera_transform.get_forward_vector()
        location = self.camera_transform.location
        
        ray_origin = np.array([location.x, location.y, location.z])
        # Simplified: assume camera aligned with vehicle
        ray_direction = ray_camera  
        
        return ray_origin, ray_direction
    
    def associate_camera_lidar(self, camera_detections, lidar_clusters, 
                               max_distance=5.0):
        """
        Associate 2D camera detections with 3D LiDAR clusters.
        
        Strategy: For each camera detection, find nearest LiDAR cluster
        within frustum cone.
        
        Args:
            camera_detections: List from CameraProcessor
            lidar_clusters: List of [x, y, z] cluster centroids
        
        Returns:
            associations: Dict {camera_idx: {'lidar_idx': int, 'confidence': float}}
        """
        associations = {}
        
        for cam_idx, detection in enumerate(camera_detections):
            pixel_center = detection['centroid_pixel']
            ray_origin, ray_dir = self.project_2d_to_3d_ray(pixel_center)
            
            # Find closest LiDAR cluster to this ray
            min_dist = float('inf')
            best_lidar_idx = -1
            
            for lidar_idx, cluster_pos in enumerate(lidar_clusters):
                # Point-to-ray distance
                cluster_vec = np.array(cluster_pos) - ray_origin
                projection = np.dot(cluster_vec, ray_dir)
                
                if projection > 0:  # In front of camera
                    closest_point = ray_origin + projection * ray_dir
                    dist = np.linalg.norm(np.array(cluster_pos) - closest_point)
                    
                    if dist < min_dist and dist < max_distance:
                        min_dist = dist
                        best_lidar_idx = lidar_idx
            
            if best_lidar_idx >= 0:
                confidence = 1.0 - (min_dist / max_distance)
                associations[cam_idx] = {
                    'lidar_idx': best_lidar_idx,
                    'distance': min_dist,
                    'confidence': confidence
                }
        
        return associations

    def project_3d_to_2d(self, points_3d):
        """
        Project 3D world points to 2D pixel coordinates.
        
        Args:
            points_3d: Nx3 array of [x, y, z] world coordinates
            
        Returns:
            points_2d: Nx2 array of [u, v] pixel coordinates
            mask: N-element boolean array (True if point is in front of camera)
        """
        if len(points_3d) == 0:
            return np.zeros((0, 2)), np.zeros(0, dtype=bool)
            
        # Convert to homogeneous coordinates
        points_3d_h = np.hstack((points_3d, np.ones((len(points_3d), 1))))
        
        # World to Camera transform
        # CARLA transform matrix is 4x4
        world_to_camera = np.linalg.inv(self.camera_transform.get_matrix())
        
        # Transform to camera frame
        # Shape: (4, 4) @ (4, N) -> (4, N)
        points_cam = np.dot(world_to_camera, points_3d_h.T).T
        
        # Camera convention: x right, y down, z forward
        # CARLA convention: x forward, y right, z up
        # We need to swap axes to match standard camera frame for projection
        # Standard Camera: X_c = y_carla, Y_c = -z_carla, Z_c = x_carla
        points_cam_standard = np.zeros_like(points_cam)
        points_cam_standard[:, 0] = points_cam[:, 1]
        points_cam_standard[:, 1] = -points_cam[:, 2]
        points_cam_standard[:, 2] = points_cam[:, 0]
        
        # Project to 2D
        # p_uv = K @ p_cam
        # We only need x,y,z components now
        xyz = points_cam_standard[:, :3]
        
        # Check which points are in front of camera (Z > 0)
        in_front = xyz[:, 2] > 0
        
        # Project
        # Shape: (3, 3) @ (3, N) -> (3, N)
        projected = np.dot(self.K, xyz.T).T
        
        # Normalize: u = x/z, v = y/z
        uv = np.zeros((len(points_3d), 2))
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            uv[:, 0] = projected[:, 0] / projected[:, 2]
            uv[:, 1] = projected[:, 1] / projected[:, 2]
            
        return uv, in_front
