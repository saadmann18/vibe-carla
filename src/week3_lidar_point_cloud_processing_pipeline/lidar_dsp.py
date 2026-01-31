# src/week3_lidar_point_cloud_processing_pipeline/lidar_dsp.py
import numpy as np
from sklearn.neighbors import NearestNeighbors

class LiDARDSP:
    """LiDAR digital signal processing module for preprocessing raw point clouds."""
    
    @staticmethod
    def statistical_outlier_removal(point_cloud, k_neighbors=20, std_ratio=2.0):
        """
        Remove noise using statistical properties of point cloud.
        
        Args:
            point_cloud: Nx3 array [[x, y, z], ...]
            k_neighbors: Number of neighbors for mean distance computation
            std_ratio: Threshold = mean + std_ratio * std_dev
        
        Returns:
            filtered_cloud: Points that pass statistical filter
        """
        if len(point_cloud) < k_neighbors:
            return point_cloud
        
        # Compute k-nearest neighbor distances
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(point_cloud)
        distances, _ = nbrs.kneighbors(point_cloud)
        
        # Mean distance to neighbors (excluding self at index 0)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        
        # Compute threshold
        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)
        threshold = global_mean + std_ratio * global_std
        
        # Keep points below threshold
        mask = mean_distances < threshold
        return point_cloud[mask]
    
    @staticmethod
    def voxel_grid_filter(point_cloud, voxel_size=0.2):
        """
        Downsample point cloud using voxel grid filtering.
        
        Args:
            point_cloud: Nx3 array
            voxel_size: Size of each voxel [m]
        
        Returns:
            downsampled_cloud: Mx3 array (M <= N)
        """
        if len(point_cloud) == 0:
            return point_cloud
            
        # Compute voxel indices
        voxel_indices = np.floor(point_cloud / voxel_size).astype(int)
        
        # Get unique voxels and their inverse indices
        unique_voxels, inverse_indices = np.unique(
            voxel_indices, axis=0, return_inverse=True
        )
        
        # Compute centroid of each voxel
        downsampled_cloud = np.zeros((len(unique_voxels), 3))
        for i in range(len(unique_voxels)):
            mask = inverse_indices == i
            downsampled_cloud[i] = np.mean(point_cloud[mask], axis=0)
        
        return downsampled_cloud
    
    @staticmethod
    def ransac_ground_removal(point_cloud, iterations=100, threshold=0.2):
        """
        Remove ground plane using RANSAC.
        
        Args:
            point_cloud: Nx3 array
            iterations: RANSAC iterations
            threshold: Distance threshold for inliers [m]
        
        Returns:
            above_ground: Points above ground plane
            ground_plane: Points belonging to the ground plane
        """
        if len(point_cloud) < 3:
            return point_cloud, np.zeros((0, 3))
            
        best_model = None
        best_inliers_mask = None
        best_inliers_count = 0
        
        N = len(point_cloud)
        
        for _ in range(iterations):
            # Randomly select 3 points
            sample_indices = np.random.choice(N, 3, replace=False)
            p1, p2, p3 = point_cloud[sample_indices]
            
            # Compute plane equation: (p2-p1) x (p3-p1) . (p-p1) = 0
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            
            normal = normal / norm
            d = -np.dot(normal, p1)
            
            # Count inliers
            distances = np.abs(np.dot(point_cloud, normal) + d)
            inliers_mask = distances < threshold
            inliers_count = np.sum(inliers_mask)
            
            if inliers_count > best_inliers_count:
                best_inliers_count = inliers_count
                best_model = (normal, d)
                best_inliers_mask = inliers_mask
        
        if best_model is None:
            return point_cloud, np.zeros((0, 3))
            
        normal, d = best_model
        # Ensure normal points UP (positive z)
        if normal[2] < 0:
            normal = -normal
            d = -d
            
        # Final refinement of mask based on the best model to ensure consistency
        distances = np.dot(point_cloud, normal) + d
        ground_mask = (distances > -threshold) & (distances < threshold)
        above_ground_mask = distances >= threshold
        
        return point_cloud[above_ground_mask], point_cloud[ground_mask]
