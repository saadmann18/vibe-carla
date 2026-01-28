# src/week2_radar_lidar_fusion/metrics.py
import numpy as np

def compute_rmse(tracks, ground_truth_positions):
    """Compute RMSE of track positions vs ground truth."""
    errors = []
    for track in tracks:
        if track.track_id in ground_truth_positions:
            state_xy = track.get_state()[:2]
            gt_xy = ground_truth_positions[track.track_id]
            error = np.linalg.norm(state_xy - gt_xy)
            errors.append(error)
    
    if errors:
        return np.sqrt(np.mean(np.array(errors) ** 2))
    return None

def compute_track_stats(tracks):
    """Compute tracking statistics."""
    if not tracks:
        return None
    
    ages = [t.age for t in tracks]
    return {
        'num_tracks': len(tracks),
        'mean_age': np.mean(ages),
        'max_age': np.max(ages),
        'total_age': np.sum(ages)
    }
