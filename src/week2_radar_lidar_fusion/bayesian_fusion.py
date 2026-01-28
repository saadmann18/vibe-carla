# src/week2_radar_lidar_fusion/bayesian_fusion.py
import numpy as np
from scipy.stats import multivariate_normal

class BayesianFusion:
    """Bayesian fusion and non-linear filtering methods."""
    
    # =====================================================================
    # PARTICLE FILTER FOR NON-LINEAR TRACKING
    # =====================================================================
    
    class ParticleFilter:
        """Particle filter for non-linear state estimation."""
        
        def __init__(self, num_particles=500, process_noise=0.1, measurement_noise=0.5, initial_state=None):
            """
            Initialize particle filter.
            """
            self.num_particles = num_particles
            self.process_noise = process_noise
            self.measurement_noise = measurement_noise
            
            if initial_state is not None:
                self.particles = np.tile(initial_state, (num_particles, 1))
                # Add some small initial jitter
                self.particles += np.random.normal(0, 0.1, (num_particles, 4))
                self.mean_state = initial_state
            else:
                self.particles = np.zeros((num_particles, 4))
                self.mean_state = np.zeros(4)
                
            self.weights = np.ones(num_particles) / num_particles
        
        def predict(self, dt):
            """
            Predict particles forward using non-linear motion model.
            
            Args:
                dt: Time step
            """
            # Constant velocity model with process noise
            for i in range(self.num_particles):
                x, y, vx, vy = self.particles[i]
                
                # Add process noise
                noise = np.random.normal(0, self.process_noise, 2)
                
                # Update position
                self.particles[i, 0] = x + vx * dt + noise[0]
                self.particles[i, 1] = y + vy * dt + noise[1]
                # Velocity stays same (constant velocity model)
        
        def update(self, measurement):
            """
            Update particle weights based on measurement (sensor observation).
            
            Args:
                measurement: [x_obs, y_obs] measured centroid
            """
            # Measurement likelihood: p(z | x) = N(z; x, R)
            for i in range(self.num_particles):
                x, y = self.particles[i, :2]
                
                # Likelihood: Gaussian centered on particle state
                distance = np.sqrt((x - measurement[0])**2 + (y - measurement[1])**2)
                likelihood = np.exp(-0.5 * (distance / self.measurement_noise)**2)
                
                self.weights[i] *= likelihood
            
            # Normalize weights
            weight_sum = np.sum(self.weights)
            if weight_sum > 1e-12:
                self.weights /= weight_sum
            else:
                self.weights = np.ones(self.num_particles) / self.num_particles
            
            # Compute mean state estimate
            self.mean_state = np.average(self.particles, axis=0, weights=self.weights)
        
        def resample(self):
            """
            Resample particles according to weights (systematic resampling).
            """
            indices = np.zeros(self.num_particles, dtype=int)
            cumsum = np.cumsum(self.weights)
            u = (np.arange(self.num_particles) + np.random.uniform(0, 1)) / self.num_particles
            
            for i in range(self.num_particles):
                indices[i] = np.searchsorted(cumsum, u[i])
            
            # Resample particles
            self.particles = self.particles[indices]
            self.weights = np.ones(self.num_particles) / self.num_particles
        
        def get_state(self):
            """Return estimated state."""
            return self.mean_state
    
    # =====================================================================
    # BAYESIAN OCCUPANCY GRID
    # =====================================================================
    
    @staticmethod
    def occupancy_grid_update(occupancy_grid, lidar_points, cell_size=0.5, 
                               max_range=100, p_occ=0.7, p_free=0.3):
        """
        Update occupancy grid using LiDAR BEV points (Bayesian update).
        
        Args:
            occupancy_grid: HxW grid of occupancy probabilities [0,1]
            lidar_points: Mx2 array of LiDAR BEV points
            cell_size: Size of each grid cell [m]
            max_range: Maximum sensor range [m]
            p_occ: Prior probability of occupied cell hit by ray
            p_free: Prior probability of free cell traversed by ray
        """
        grid_height, grid_width = occupancy_grid.shape
        updated_grid = occupancy_grid.copy()
        
        # Vehicle position (center of grid)
        ego_x_idx, ego_y_idx = grid_width // 2, grid_height // 2
        
        for point in lidar_points:
            px, py = point
            
            # Convert to grid indices
            grid_x_idx = int(ego_x_idx + px / cell_size)
            grid_y_idx = int(ego_y_idx + py / cell_size)
            
            if 0 <= grid_x_idx < grid_width and 0 <= grid_y_idx < grid_height:
                # Mark cell as occupied via Bayesian update
                # p(occ|z) = (p(z|occ) * p(occ)) / (p(z|occ)*p(occ) + p(z|!occ)*p(!occ))
                p_old = occupancy_grid[grid_y_idx, grid_x_idx]
                
                # Using log-odds would be more stable, but following generic plan:
                numerator = p_occ * p_old
                denominator = p_occ * p_old + (1 - p_occ) * (1 - p_old)
                
                if denominator > 1e-6:
                    updated_grid[grid_y_idx, grid_x_idx] = numerator / denominator
        
        return updated_grid
    
    # =====================================================================
    # BAYESIAN SENSOR FUSION (Confidence Weighting)
    # =====================================================================
    
    @staticmethod
    def bayesian_centroid_fusion(radar_centroid, lidar_centroid, 
                                   radar_confidence=0.6, lidar_confidence=0.8):
        """
        Fuse radar and LiDAR centroids using Bayesian weighting.
        """
        total_confidence = radar_confidence + lidar_confidence
        
        if total_confidence == 0:
            return (radar_centroid + lidar_centroid) / 2
        
        fused = (radar_confidence * radar_centroid + 
                 lidar_confidence * lidar_centroid) / total_confidence
        
        return fused
