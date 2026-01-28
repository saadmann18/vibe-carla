# src/week2_radar_lidar_fusion/radar_lidar_fusion_advanced.py
"""
Advanced Week 2: Radar–LiDAR fusion with DSP and Bayesian methods.
Extends the baseline fusion with signal processing and non-linear tracking.
"""

import carla
import numpy as np
import time
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from radar_utils import polar_to_cartesian
from lidar_utils import project_lidar_to_bev
from radar_dsp import RadarDSP
from lidar_dsp import LiDARDSP
from bayesian_fusion import BayesianFusion
from tracking import dbscan_cluster, compute_centroids, kalman_predict, kalman_update
from visualization import plot_bev, plot_tracks_timeline

# Global sensor queues
radar_queue = None
lidar_queue = None

def radar_callback(radar_data):
    global radar_queue
    radar_queue = radar_data

def lidar_callback(lidar_data):
    global lidar_queue
    lidar_queue = lidar_data

def lidar_data_to_array(lidar_data):
    points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    return points[:, :3]

def radar_data_to_detections(radar_data):
    return list(radar_data)

def run_advanced_week2_simulation(duration_seconds=120, speed_kmh=50, 
                                   use_particle_filter=True, use_cfar=True):
    """
    Advanced Week 2 with DSP + Bayesian methods.
    """
    global radar_queue, lidar_queue
    
    print("=" * 70)
    print("WEEK 2 ADVANCED: RADAR–LIDAR FUSION WITH DSP & BAYESIAN METHODS")
    print("=" * 70)
    print(f"Duration: {duration_seconds}s | Speed: {speed_kmh} km/h")
    print(f"Particle Filter: {use_particle_filter} | CFAR Detector: {use_cfar}")
    print()
    
    # Initialize DSP modules
    radar_dsp = RadarDSP(fft_size=256, sample_rate=10000)
    lidar_dsp = LiDARDSP()
    bayesian = BayesianFusion()
    
    # Connect to CARLA
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        print("✅ Connected to CARLA")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return
    
    blueprint_library = world.get_blueprint_library()
    
    # Spawn vehicle - try multiple spawn points to avoid collisions
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_points = world.get_map().get_spawn_points()
    
    vehicle = None
    for i, spawn_point in enumerate(spawn_points[:10]):
        try:
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            vehicle.set_simulate_physics(True)
            print(f"✅ Spawned vehicle at spawn point {i}: {spawn_point.location}")
            break
        except RuntimeError as e:
            if i < 9: continue
            else:
                print(f"❌ Failed to spawn vehicle after 10 attempts")
                return
    
    if vehicle is None: return
    
    # Attach sensors
    radar_bp = blueprint_library.find('sensor.other.radar')
    radar_bp.set_attribute('range', '100')
    radar = world.spawn_actor(radar_bp, carla.Transform(carla.Location(x=0, z=1.5)), 
                              attach_to=vehicle)
    radar.listen(radar_callback)
    
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '100')
    lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=0, z=2.0)), 
                              attach_to=vehicle)
    lidar.listen(lidar_callback)
    
    # Enable autopilot
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_desired_speed(vehicle, speed_kmh)
    vehicle.set_autopilot(True, 8000)
    
    print("✅ Sensors attached | Autopilot enabled")
    print()
    
    # Initialize tracking
    particle_filters = {}  # track_id -> ParticleFilter
    tracks = []
    frame_count = 0
    tracks_history = defaultdict(list)
    start_time = time.time()
    
    # Occupancy grid
    occupancy_grid = 0.5 * np.ones((200, 200))
    
    print("🚗 Starting advanced simulation...")
    print()
    
    try:
        while time.time() - start_time < duration_seconds:
            world.tick()
            frame_count += 1
            dt = 0.05
            
            if radar_queue is None or lidar_queue is None:
                if frame_count % 100 == 0:
                    print(f"Frame {frame_count:4d}: Waiting for sensor data...")
                continue
            
            # 1. Advanced Radar DSP
            radar_detections = radar_data_to_detections(radar_queue)
            radar_pc = polar_to_cartesian(radar_detections)
            
            if use_cfar and len(radar_detections) > 0:
                rd_map = radar_dsp.compute_range_doppler_map(radar_detections)
                # (Could add CFAR detection filtering logic here if needed)
            
            # 2. Advanced LiDAR DSP
            lidar_points = lidar_data_to_array(lidar_queue)
            
            # Statistical Outlier Removal
            lidar_filtered = lidar_dsp.statistical_outlier_removal(lidar_points)
            
            # RANSAC ground removal
            lidar_above = lidar_dsp.ransac_ground_removal(lidar_filtered)
            
            # BEV Projection
            lidar_bev = project_lidar_to_bev(lidar_above)
            
            # 3. Bayesian Occupancy Grid
            if len(lidar_bev) > 0:
                occupancy_grid = bayesian.occupancy_grid_update(occupancy_grid, lidar_bev)
            
            # 4. Clustering & Control centroids
            if len(radar_pc) > 0:
                cluster_ids = dbscan_cluster(radar_pc[:, :2], eps=1.5, min_samples=2)
                centroids = compute_centroids(radar_pc, cluster_ids)
            else:
                centroids = []
                
            # 5. Advanced Tracking
            if use_particle_filter:
                # Particle filter matching (simplified nearest neighbor)
                matched_indices = set()
                # Store misses to prune
                tracks_to_remove = []
                
                for track_id, info in particle_filters.items():
                    pf = info['pf']
                    pf.predict(dt)
                    state = pf.get_state()
                    
                    found_match = False
                    if centroids:
                        dists = [np.linalg.norm(np.array(c[:2]) - state[:2]) for c in centroids]
                        nearest = np.argmin(dists)
                        if dists[nearest] < 5.0:
                            pf.update(centroids[nearest][:2])
                            pf.resample()
                            matched_indices.add(nearest)
                            tracks_history[track_id].append(pf.get_state()[:2])
                            info['misses'] = 0
                            found_match = True
                    
                    if not found_match:
                        info['misses'] += 1
                        if info['misses'] > 10:
                            tracks_to_remove.append(track_id)
                
                # Prune
                for tid in tracks_to_remove:
                    del particle_filters[tid]
                
                # New particles for new objects
                next_id = max(particle_filters.keys()) + 1 if particle_filters else 0
                for i, cent in enumerate(centroids):
                    if i not in matched_indices:
                        pf = bayesian.ParticleFilter(initial_state=np.array(cent))
                        particle_filters[next_id] = {'pf': pf, 'misses': 0}
                        next_id += 1
            else:
                # Kalman legacy
                tracks = kalman_predict(tracks, dt)
                tracks = kalman_update(tracks, centroids)
                for track in tracks:
                    tracks_history[track.track_id].append(track.get_state()[:2])
            
            # 6. Periodic Output
            if frame_count % 500 == 0:
                num_obj = len(particle_filters) if use_particle_filter else len(tracks)
                print(f"Frame {frame_count:4d} | Radar: {len(radar_pc):2d} pts | "
                      f"LiDAR: {len(lidar_points):4d} pts | Filtered: {len(lidar_above):4d} pts | "
                      f"Objects: {num_obj:2d}")
                
                # Use simple list for visualization
                viz_tracks = []
                if use_particle_filter:
                    class MockTrack:
                        def __init__(self, id, pos): self.track_id = id; self.pos = pos
                        def get_state(self): return np.array([self.pos[0], self.pos[1], 0, 0])
                    for tid, info in particle_filters.items():
                        viz_tracks.append(MockTrack(tid, info['pf'].get_state()[:2]))
                else:
                    viz_tracks = tracks
                    
                plot_bev(radar_pc, lidar_bev, viz_tracks,
                        f"src/week2_radar_lidar_fusion/results/advanced_bev_frame_{frame_count:04d}.png")
                        
    except KeyboardInterrupt:
        print("\n⏹ Simulation interrupted.")
    finally:
        elapsed = time.time() - start_time
        print(f"\nAdvanced Simulation Complete: {elapsed:.1f}s | {frame_count} frames")
        
        if tracks_history:
            plot_tracks_timeline(tracks_history, "src/week2_radar_lidar_fusion/results/advanced_timeline.png")
            
        vehicle.destroy()
        radar.destroy()
        lidar.destroy()
        print("✅ Cleanup complete")

if __name__ == '__main__':
    run_advanced_week2_simulation(duration_seconds=120)
