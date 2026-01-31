# src/week4_multimodal_fusion/multimodal_pipeline.py
"""
Week 4 Main: Camera + LiDAR + Radar Multi-Modal Fusion Pipeline
"""
import carla
import numpy as np
import time
from collections import defaultdict

from camera_utils import CameraProcessor
from sensor_projection import SensorProjection
from multimodal_fusion import MultiModalFusion
from visualization import plot_multimodal_bev, project_lidar_to_camera

# ... (rest of imports)

# ... (rest of imports)

import sys
import os

# Robust path addition
current_dir = os.path.dirname(os.path.abspath(__file__))
week2_path = os.path.join(current_dir, '../week2_radar_lidar_fusion')
week3_path = os.path.join(current_dir, '../week3_lidar_pipeline')

sys.path.append(week2_path)
sys.path.append(week3_path)

try:
    from radar_utils import polar_to_cartesian
except ImportError:
    print(f"Warning: radar_utils not found in {week2_path}")
    pass

try:
    from lidar_dsp import LiDARDSP
except ImportError:
    print(f"Warning: LiDARDSP not found in {week3_path}")
    pass

# Simplified clustering for demo
from sklearn.cluster import DBSCAN

# Global queues
camera_queue = None
lidar_queue = None
radar_queue = None

def camera_callback(image):
    global camera_queue
    camera_queue = image

def lidar_callback(data):
    global lidar_queue
    lidar_queue = data

def radar_callback(data):
    global radar_queue
    radar_queue = data

def simple_euclidean_clustering(points, eps=2.0, min_samples=5):
    """Quick clustering for demo."""
    if len(points) < min_samples:
        return []
    
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clusterer.fit_predict(points)
    
    centroids = []
    for label in set(labels):
        if label == -1:
            continue
        mask = labels == label
        centroid = np.mean(points[mask], axis=0)
        centroids.append(centroid)
    
    return centroids

def run_multimodal_fusion(duration=120):
    """Week 4 complete multi-modal fusion."""
    global camera_queue, lidar_queue, radar_queue
    
    print("=" * 80)
    print("🚀 WEEK 4: MULTI-MODAL SENSOR FUSION")
    print("Camera + LiDAR + Radar → Unified BEV Perception")
    print("=" * 80)
    print()
    
    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    
    # Spawn vehicle
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_points = world.get_map().get_spawn_points()
    
    vehicle = None
    for point in spawn_points:
        try:
            vehicle = world.spawn_actor(vehicle_bp, point)
            break
        except RuntimeError:
            continue
            
    if vehicle is None:
        print("❌ Could not spawn vehicle (all spawn points blocked)")
        return

    vehicle.set_simulate_physics(True)
    print("✅ Vehicle spawned")
    
    # Attach Camera (Semantic Segmentation)
    camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    camera_transform = carla.Transform(carla.Location(x=0, z=2.0))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    camera.listen(camera_callback)
    print("✅ Camera attached (semantic segmentation)")
    
    # Attach LiDAR
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('range', '100')
    lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=0, z=2.5)), 
                              attach_to=vehicle)
    lidar.listen(lidar_callback)
    print("✅ LiDAR attached")
    
    # Attach Radar
    radar_bp = blueprint_library.find('sensor.other.radar')
    radar_bp.set_attribute('range', '100')
    radar = world.spawn_actor(radar_bp, carla.Transform(carla.Location(x=0, z=1.5)), 
                              attach_to=vehicle)
    radar.listen(radar_callback)
    print("✅ Radar attached")
    
    # Enable autopilot
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_desired_speed(vehicle, 50)
    vehicle.set_autopilot(True, traffic_manager.get_port())
    print("✅ Autopilot enabled at 50 km/h")
    print()
    
    # Initialize processors
    cam_processor = CameraProcessor(image_width=800, image_height=600, fov=90)
    # Handle optional import if LiDARDSP is not present
    try:
        lidar_dsp = LiDARDSP()
    except NameError:
        lidar_dsp = None
        
    sensor_proj = SensorProjection(cam_processor.get_intrinsics(), camera_transform)
    fusion = MultiModalFusion()
    
    frame_count = 0
    fused_tracks = []
    next_track_id = 0
    start_time = time.time()
    
    print("🚗 Starting multi-modal fusion...")
    print()
    
    try:
        while time.time() - start_time < duration:
            world.tick()
            frame_count += 1
            
            # Wait for all sensors
            if camera_queue is None or lidar_queue is None or radar_queue is None:
                if frame_count % 20 == 0:
                    print(f"Frame {frame_count}: Waiting for sensors...")
                continue
            
            # =================================================================
            # CAMERA: 2D Object Detection
            # =================================================================
            semantic_array = np.frombuffer(camera_queue.raw_data, dtype=np.dtype("uint8"))
            semantic_array = np.reshape(semantic_array, (camera_queue.height, 
                                                          camera_queue.width, 4))
            camera_detections = cam_processor.process_semantic_segmentation(semantic_array)
            
            # =================================================================
            # LIDAR: 3D Point Cloud → Clustering
            # =================================================================
            lidar_raw = np.frombuffer(lidar_queue.raw_data, dtype=np.dtype('f4'))
            lidar_points = np.reshape(lidar_raw, (int(lidar_raw.shape[0] / 4), 4))[:, :3]
            
            # Preprocessing
            if len(lidar_points) > 50 and lidar_dsp is not None:
                lidar_filtered = lidar_dsp.statistical_outlier_removal(lidar_points, 
                                                                       k_neighbors=20)
                lidar_downsampled = lidar_dsp.voxel_grid_filter(lidar_filtered, voxel_size=0.3)
                lidar_above = lidar_dsp.ransac_ground_removal(lidar_downsampled, 
                                                              iterations=50, threshold=0.3)
            else:
                lidar_above = lidar_points
            
            # Clustering
            lidar_clusters = simple_euclidean_clustering(lidar_above, eps=2.5, min_samples=10)
            
            # =================================================================
            # RADAR: Range-Velocity Centroids
            # =================================================================
            radar_detections = list(radar_queue)
            try:
                radar_pc = polar_to_cartesian(radar_detections)
            except NameError:
                 # Fallback if polar_to_cartesian not imported
                 # Manually convert if needed, or skip
                 radar_pc = np.zeros((0, 4))

            
            # Simple centroid (no clustering for radar in this demo)
            radar_centroids = []
            if len(radar_pc) > 0:
                radar_bev = radar_pc[:, :2]
                radar_centroids = simple_euclidean_clustering(radar_bev, eps=3.0, min_samples=3)
            
            # =================================================================
            # MULTI-MODAL ASSOCIATION & FUSION
            # =================================================================
            associations = sensor_proj.associate_camera_lidar(camera_detections, 
                                                              lidar_clusters, 
                                                              max_distance=5.0)
            
            # Build fused tracks
            new_fused_tracks = []
            for cam_idx, assoc in associations.items():
                lidar_idx = assoc['lidar_idx']
                
                # Get positions
                lidar_pos = lidar_clusters[lidar_idx]
                
                # Find nearest radar (if any)
                radar_pos = None
                if radar_centroids:
                    distances = [np.linalg.norm(np.array(r[:2]) - lidar_pos[:2]) 
                                for r in radar_centroids]
                    if min(distances) < 5.0:
                        radar_pos = radar_centroids[np.argmin(distances)]
                
                # Fusion
                camera_pos_dummy = lidar_pos[:2]  # Simplified
                if radar_pos is not None:
                    fused_pos = fusion.fuse_position(camera_pos_dummy, lidar_pos, radar_pos)
                else:
                    fused_pos = lidar_pos
                
                # Confidence
                sensor_flags = {
                    'camera': True,
                    'lidar': True,
                    'radar': radar_pos is not None
                }
                confidence = fusion.compute_track_confidence(sensor_flags)
                
                new_fused_tracks.append({
                    'id': next_track_id,
                    'pos': fused_pos,
                    'confidence': confidence
                })
                next_track_id += 1
            
            fused_tracks = new_fused_tracks
            
            # =================================================================
            # VISUALIZATION (Every 500 frames)
            # =================================================================
            if frame_count % 500 == 0:
                print(f"Frame {frame_count:4d} | Camera: {len(camera_detections)} | "
                      f"LiDAR: {len(lidar_clusters)} | Radar: {len(radar_centroids)} | "
                      f"Fused: {len(fused_tracks)} tracks")
                
                # 1. BEV Plot
                plot_multimodal_bev(
                    camera_detections,
                    np.array(lidar_clusters) if lidar_clusters else np.zeros((0, 3)),
                    np.array(radar_centroids) if radar_centroids else np.zeros((0, 2)),
                    fused_tracks,
                    f"src/week4_multimodal_fusion/results/multimodal_frame_{frame_count:04d}.png"
                )
                
                # 2. Camera Overlay
                if camera_queue is not None:
                    # Get raw image
                    image_data = np.frombuffer(camera_queue.raw_data, dtype=np.dtype("uint8"))
                    image_data = np.reshape(image_data, (camera_queue.height, camera_queue.width, 4))
                    image_rgb = image_data[:, :, :3]  # Drop alpha
                    
                    # Project LiDAR points
                    # Use full point cloud (not just clusters/filtered) for better visual
                    points_to_project = lidar_above 
                    
                    overlay_img = project_lidar_to_camera(image_rgb, points_to_project, sensor_proj)
                    
                    # Save
                    import cv2
                    cv2.imwrite(f"src/week4_multimodal_fusion/results/camera_overlay_{frame_count:04d}.png", overlay_img)
                    print(f"✅ Saved overlay: camera_overlay_{frame_count:04d}.png")
    
    except KeyboardInterrupt:
        print("\n⏹ Interrupted by user")
    
    finally:
        elapsed = time.time() - start_time
        print()
        print("=" * 80)
        print("WEEK 4 COMPLETE")
        print("=" * 80)
        print(f"Duration: {elapsed:.1f}s | Frames: {frame_count}")
        print(f"Total fused tracks: {next_track_id}")
        print()
        
        vehicle.destroy()
        camera.destroy()
        lidar.destroy()
        radar.destroy()
        print("✅ Cleanup complete")

if __name__ == '__main__':
    run_multimodal_fusion(duration=200) # Increased duration slightly for user to see
