# src/week2_radar_lidar_fusion/radar_lidar_fusion.py
import carla
import numpy as np
import time
from collections import defaultdict
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from radar_utils import polar_to_cartesian
from lidar_utils import project_lidar_to_bev
from tracking import (dbscan_cluster, compute_centroids, kalman_predict,
                      kalman_update, fuse_radar_lidar)
from metrics import compute_track_stats
from visualization import plot_bev, plot_tracks_timeline

# ============================================================================
# GLOBAL SENSOR QUEUES
# ============================================================================

radar_queue = None
lidar_queue = None

def radar_callback(radar_data):
    """Called whenever radar sensor produces data."""
    global radar_queue
    radar_queue = radar_data

def lidar_callback(lidar_data):
    """Called whenever LiDAR sensor produces data."""
    global lidar_queue
    lidar_queue = lidar_data

def lidar_data_to_array(lidar_data):
    """Convert CARLA LiDAR data to Nx3 numpy array (x, y, z)."""
    points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    return points[:, :3]

def radar_data_to_detections(radar_data):
    """Extract detection list from CARLA radar data."""
    return list(radar_data)

# ============================================================================
# MAIN SIMULATION
# ============================================================================

def run_week2_simulation(duration_seconds=120, speed_kmh=50):
    """
    Main Week 2 simulation: Radar + LiDAR fusion.
    """
    global radar_queue, lidar_queue
    
    print("=" * 70)
    print("WEEK 2: RADAR–LIDAR SENSOR FUSION FOR AV PERCEPTION")
    print("=" * 70)
    print(f"Duration: {duration_seconds}s | Speed: {speed_kmh} km/h")
    print()
    
    # Connect to CARLA
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        print("✅ Connected to CARLA")
    except Exception as e:
        print(f"❌ Failed to connect to CARLA: {e}")
        print("Make sure CARLA is running: ./CarlaUE4.sh -quality-level=Low")
        return
    
    blueprint_library = world.get_blueprint_library()
    
    # Spawn vehicle - try multiple spawn points to avoid collisions
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_points = world.get_map().get_spawn_points()
    
    vehicle = None
    for i, spawn_point in enumerate(spawn_points[:10]):  # Try first 10 spawn points
        try:
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            vehicle.set_simulate_physics(True)
            print(f"✅ Spawned vehicle at spawn point {i}: {spawn_point.location}")
            break
        except RuntimeError as e:
            if i < 9:  # Not the last attempt
                continue
            else:
                print(f"❌ Failed to spawn vehicle after trying 10 spawn points")
                print("Try clearing the CARLA world or restarting the server")
                return
    
    if vehicle is None:
        print("❌ Could not spawn vehicle")
        return
    
    # Attach Radar
    radar_bp = blueprint_library.find('sensor.other.radar')
    radar_bp.set_attribute('horizontal_fov', '35')
    radar_bp.set_attribute('vertical_fov', '20')
    radar_bp.set_attribute('range', '100')
    radar_transform = carla.Transform(carla.Location(x=0, z=1.5))
    radar = world.spawn_actor(radar_bp, radar_transform, attach_to=vehicle)
    radar.listen(radar_callback)
    print("✅ Attached Radar sensor")
    
    # Attach LiDAR
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('range', '100')
    lidar_bp.set_attribute('points_per_second', '100000')
    lidar_transform = carla.Transform(carla.Location(x=0, z=2.0))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    lidar.listen(lidar_callback)
    print("✅ Attached LiDAR sensor")
    
    # Enable autopilot
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_desired_speed(vehicle, speed_kmh)
    vehicle.set_autopilot(True, 8000)  # Use port number directly
    print(f"✅ Autopilot enabled at {speed_kmh} km/h")
    print()
    
    # Initialize tracking
    tracks = []
    frame_count = 0
    tracks_history = defaultdict(list)
    radar_points_history = []
    start_time = time.time()
    
    print("🚗 Starting simulation...")
    print()
    
    try:
        while time.time() - start_time < duration_seconds:
            world.tick()
            frame_count += 1
            dt = 0.05  # 20 Hz CARLA
            
            # Wait for sensor data
            if radar_queue is None or lidar_queue is None:
                if frame_count % 20 == 0:
                    print(f"Frame {frame_count}: Waiting for sensor data...")
                continue
            
            # ============================================================
            # STEP 1: Generate Point Clouds
            # ============================================================
            radar_detections = radar_data_to_detections(radar_queue)
            radar_pc = polar_to_cartesian(radar_detections)
            
            lidar_points = lidar_data_to_array(lidar_queue)
            lidar_bev = project_lidar_to_bev(lidar_points)
            
            # ============================================================
            # STEP 2: Clustering
            # ============================================================
            if len(radar_pc) > 0:
                cluster_ids = dbscan_cluster(radar_pc[:, :2], eps=1.5, min_samples=2)
                centroids = compute_centroids(radar_pc, cluster_ids)
            else:
                centroids = []
            
            # ============================================================
            # STEP 3: Kalman Tracking
            # ============================================================
            tracks = kalman_predict(tracks, dt)
            tracks = kalman_update(tracks, centroids, max_distance=5.0)
            
            # ============================================================
            # STEP 4: LiDAR Fusion Refinement
            # ============================================================
            for track in tracks:
                track_xy = track.get_state()[:2]
                refined_xy = fuse_radar_lidar(track_xy, lidar_bev, search_radius=2.0)
                track.kf.x[:2] = refined_xy.reshape(2, 1)
            
            # ============================================================
            # STEP 5: Logging
            # ============================================================
            for track in tracks:
                state = track.get_state()
                tracks_history[track.track_id].append(state[:2])
            
            radar_points_history.append({
                'frame': frame_count,
                'radar_points': len(radar_pc),
                'lidar_points': len(lidar_bev),
                'num_tracks': len(tracks)
            })
            
            # ============================================================
            # STEP 6: Periodic Visualization
            # ============================================================
            if frame_count % 500 == 0:
                stats = compute_track_stats(tracks)
                print(f"Frame {frame_count:4d} | Radar: {len(radar_pc):3d} pts | "
                      f"LiDAR: {len(lidar_bev):4d} pts | Tracks: {len(tracks):2d}")
                
                plot_bev(radar_pc, lidar_bev, tracks,
                        f"src/week2_radar_lidar_fusion/results/bev_frame_{frame_count:04d}.png")
    
    except KeyboardInterrupt:
        print("\n⏹ Simulation interrupted by user.")
    
    finally:
        elapsed = time.time() - start_time
        print()
        print("=" * 70)
        print("SIMULATION COMPLETE")
        print("=" * 70)
        print(f"Duration: {elapsed:.1f}s | Frames: {frame_count}")
        print(f"Total unique tracks: {len(tracks_history)}")
        
        if tracks_history:
            plot_tracks_timeline(tracks_history,
                               "src/week2_radar_lidar_fusion/results/tracks_timeline.png")
        
        print()
        print("📁 Results saved to: src/week2_radar_lidar_fusion/results/")
        print()
        
        # Cleanup
        vehicle.destroy()
        radar.destroy()
        lidar.destroy()
        print("✅ Cleanup complete")

if __name__ == '__main__':
    run_week2_simulation(duration_seconds=120, speed_kmh=50)
