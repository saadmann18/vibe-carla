# src/week2_radar_lidar_fusion/visualization.py
import numpy as np
import matplotlib.pyplot as plt

def plot_bev(radar_points, lidar_bev, tracks, filename="bev_result.png"):
    """Plot BEV with radar, LiDAR, and tracks."""
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Radar points colored by velocity
    if len(radar_points) > 0:
        velocities = np.linalg.norm(radar_points[:, 2:4], axis=1)
        scatter = ax.scatter(radar_points[:, 0], radar_points[:, 1],
                            c=velocities, cmap='Reds', s=30, alpha=0.7,
                            label='Radar BEV')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Velocity [m/s]')
    
    # LiDAR points
    if len(lidar_bev) > 0:
        ax.scatter(lidar_bev[:, 0], lidar_bev[:, 1], c='cyan', s=10, alpha=0.4,
                   label='LiDAR BEV')
    
    # Tracks
    for track in tracks:
        state = track.get_state()
        ax.plot(state[0], state[1], 'go', markersize=12)
        ax.text(state[0] + 0.5, state[1] + 0.5, f'T{track.track_id}', fontsize=10)
    
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_title('BEV: Radar + LiDAR + Fused Tracks', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-50, 150)
    ax.set_ylim(-50, 50)
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()

def plot_tracks_timeline(tracks_history, filename="tracks_timeline.png"):
    """Plot track trajectories over time."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for track_id, positions in tracks_history.items():
        positions = np.array(positions)
        ax.plot(positions[:, 0], positions[:, 1], '-o', 
               label=f'Track {track_id}',
               color=colors[track_id % 10], linewidth=2, markersize=4)
    
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_title('Object Trajectories Over Time', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()
