# src/week3_lidar_point_cloud_processing_pipeline/lidar_visualization.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

class LiDARVisualization:
    """Visualization utilities for Lidar pipeline results."""
    
    @staticmethod
    def plot_bev(point_cloud=None, ground_cloud=None, bboxes=None, tracks=None, 
                 filename="bev_output.png", title="BEV Perception"):
        """
        Generate a Bird's Eye View plot.
        
        Args:
            point_cloud: Nx3 array of filtered/object points
            ground_cloud: Nx3 array of ground points (optional)
            bboxes: List of bounding box dicts
            tracks: List of KalmanTrack objects
            filename: Path to save the image
            title: Title for the plot
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 1. Ground points (Gray)
        if ground_cloud is not None and len(ground_cloud) > 0:
            ax.scatter(ground_cloud[:, 0], ground_cloud[:, 1], c='gray', s=1, alpha=0.1, label='Ground')
            
        # 2. Object points (Cyan/Blue)
        if point_cloud is not None and len(point_cloud) > 0:
            ax.scatter(point_cloud[:, 0], point_cloud[:, 1], c='cyan', s=2, alpha=0.5, label='Objects/Points')
            
        # 3. Bounding Boxes (Red)
        if bboxes is not None:
            for bbox in bboxes:
                min_pt = bbox['min_pt']
                dims = bbox['dims']
                # Create a rectangle patch
                rect = patches.Rectangle(
                    (min_pt[0], min_pt[1]), dims[0], dims[1],
                    linewidth=1, edgecolor='red', facecolor='none', alpha=0.7
                )
                ax.add_patch(rect)
                
        # 4. Tracked Objects (Green + ID)
        if tracks is not None:
            for track in tracks:
                state = track.get_state()
                ax.plot(state[0], state[1], 'go', markersize=8)
                ax.text(state[0] + 0.5, state[1] + 0.5, f'ID:{track.track_id}', 
                        color='green', fontsize=9, fontweight='bold')
                
        # Formatting
        ax.set_aspect('equal')
        ax.set_xlim(-20, 100)
        ax.set_ylim(-40, 40)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
