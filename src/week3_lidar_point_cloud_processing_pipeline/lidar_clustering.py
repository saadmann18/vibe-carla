# src/week3_lidar_point_cloud_processing_pipeline/lidar_clustering.py
import numpy as np
from sklearn.cluster import DBSCAN

class LiDARClustering:
    """LiDAR clustering module for detecting objects from denoised point clouds."""
    
    @staticmethod
    def euclidean_clustering(points, eps=0.5, min_samples=5):
        """
        Cluster 3D points using DBSCAN.
        
        Args:
            points: Nx3 array
            eps: DBSCAN search radius
            min_samples: Minimum points per cluster
            
        Returns:
            cluster_ids: Array of cluster IDs (-1 for noise)
        """
        if len(points) < min_samples:
            return np.array([-1] * len(points))
            
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        return clusterer.fit_predict(points)
    
    @staticmethod
    def get_bounding_boxes(points, cluster_ids):
        """
        Compute 3D bounding boxes for each cluster.
        
        Args:
            points: Nx3 array
            cluster_ids: Output from euclidean_clustering
            
        Returns:
            bboxes: List of dicts containing 'min_pt', 'max_pt', 'centroid', 'dims'
        """
        unique_clusters = set(cluster_ids)
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
            
        bboxes = []
        for cluster_id in sorted(unique_clusters):
            mask = cluster_ids == cluster_id
            cluster_points = points[mask]
            
            min_pt = np.min(cluster_points, axis=0)
            max_pt = np.max(cluster_points, axis=0)
            centroid = np.mean(cluster_points, axis=0)
            dims = max_pt - min_pt
            
            bboxes.append({
                'cluster_id': cluster_id,
                'min_pt': min_pt,
                'max_pt': max_pt,
                'centroid': centroid,
                'dims': dims,
                'num_points': len(cluster_points)
            })
            
        return bboxes

    @staticmethod
    def filter_clusters_by_size(bboxes, min_dims=(0.2, 0.2, 0.2), max_dims=(20, 10, 5)):
        """
        Filter out clusters that are too small or too large to be vehicles/pedestrians.
        """
        filtered_bboxes = []
        for bbox in bboxes:
            d = bbox['dims']
            if np.all(d >= min_dims) and np.all(d <= max_dims):
                filtered_bboxes.append(bbox)
        return filtered_bboxes
