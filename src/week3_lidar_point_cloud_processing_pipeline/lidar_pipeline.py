# src/week3_lidar_point_cloud_processing_pipeline/lidar_pipeline.py
import carla
import numpy as np
import time
import os
import datetime
from lidar_dsp import LiDARDSP
from lidar_clustering import LiDARClustering
from lidar_tracking import LiDARTracker
from lidar_visualization import LiDARVisualization

# Configuration
SIM_HOST = 'localhost'
SIM_PORT = 2000
STEPS = 50
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

class LiDARPipeline:
    def __init__(self):
        self.client = carla.Client(SIM_HOST, SIM_PORT)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # 1. Modules
        self.dsp = LiDARDSP()
        self.clustering = LiDARClustering()
        self.tracker = LiDARTracker()
        self.viz = LiDARVisualization()
        
        # 2. Sensors/Actors
        self.vehicle = None
        self.lidar_sensor = None
        self.latest_cloud = None
        self.timestamp = None
        
        os.makedirs(RESULTS_DIR, exist_ok=True)

    def _lidar_callback(self, data):
        """Handle raw Lidar data from CARLA."""
        # Convert raw buffer to Nx4 (x, y, z, intensity)
        raw_data = np.frombuffer(data.raw_data, dtype=np.float32)
        self.latest_cloud = raw_data.reshape((-1, 4))[:, :3] # Keep only x, y, z
        self.timestamp = data.timestamp

    def find_hero_vehicle(self):
        """Find the 'hero' vehicle to attach sensor."""
        for actor in self.world.get_actors():
            if actor.attributes.get('role_name') == 'hero':
                self.vehicle = actor
                print(f"✅ Found hero vehicle: {actor.type_id}")
                return True
        print("❌ Could not find hero vehicle. Run manual_control.py first.")
        return False

    def setup_lidar(self):
        """Spawn and attach Lidar sensor."""
        bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        bp.set_attribute('range', '50')
        bp.set_attribute('channels', '32')
        bp.set_attribute('points_per_second', '100000')
        bp.set_attribute('rotation_frequency', '10') # 10Hz
        
        transform = carla.Transform(carla.Location(x=0, z=2.5))
        self.lidar_sensor = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        self.lidar_sensor.listen(self._lidar_callback)

    def cleanup(self):
        if self.lidar_sensor:
            self.lidar_sensor.stop()
            self.lidar_sensor.destroy()
        print("🛑 Pipeline stopped.")

    def run(self):
        if not self.find_hero_vehicle():
            return
            
        self.setup_lidar()
        print("🚀 Starting Pipeline...")
        
        last_time = time.time()
        
        step = 0
        while step < STEPS:
            # Wait for data
            if self.latest_cloud is None:
                time.sleep(0.01)
                continue
                
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # --- START PIPELINE ---
            p_start = time.time()
            
            # 1. Preprocessing (DSP)
            cloud = self.dsp.statistical_outlier_removal(self.latest_cloud)
            cloud = self.dsp.voxel_grid_filter(cloud, voxel_size=0.2)
            above_ground, ground = self.dsp.ransac_ground_removal(cloud)
            
            # 2. Clustering
            cluster_ids = self.clustering.euclidean_clustering(above_ground)
            bboxes = self.clustering.get_bounding_boxes(above_ground, cluster_ids)
            filtered_bboxes = self.clustering.filter_clusters_by_size(bboxes)
            
            # 3. Tracking
            centroids = [b['centroid'] for b in filtered_bboxes]
            active_tracks = self.tracker.update(centroids, dt)
            
            p_end = time.time()
            latency = (p_end - p_start) * 1000 # ms
            
            # --- Visualization ---
            if step % 5 == 0:
                print(f"Step {step:3d} | Latency: {latency:.1f}ms | Objects: {len(active_tracks)}")
                filename = os.path.join(RESULTS_DIR, f"frame_{step:04d}.png")
                self.viz.plot_bev(
                    point_cloud=above_ground,
                    ground_cloud=ground,
                    bboxes=filtered_bboxes,
                    tracks=active_tracks,
                    filename=filename,
                    title=f"Step {step} | Latency: {latency:.1f}ms"
                )
            
            step += 1
            time.sleep(0.1) # 10Hz sync-ish
            
        self.cleanup()

if __name__ == "__main__":
    pipeline = LiDARPipeline()
    try:
        pipeline.run()
    except KeyboardInterrupt:
        pipeline.cleanup()
