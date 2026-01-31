# src/week4_multimodal_fusion/camera_utils.py
"""Camera RGB processing and 2D object detection."""
import numpy as np
import cv2

class CameraProcessor:
    """Process CARLA RGB camera."""
    
    def __init__(self, image_width=800, image_height=600, fov=90):
        self.width = image_width
        self.height = image_height
        self.fov = fov
        
        # Build camera intrinsics matrix
        focal_length = self.width / (2.0 * np.tan(self.fov * np.pi / 360.0))
        self.K = np.array([
            [focal_length, 0, self.width / 2.0],
            [0, focal_length, self.height / 2.0],
            [0, 0, 1.0]
        ])
    
    def process_semantic_segmentation(self, semantic_image):
        """
        Extract vehicle bounding boxes from CARLA semantic segmentation.
        
        Args:
            semantic_image: CARLA semantic image (HWC uint8)
        
        Returns:
            detections: List of dicts with 'bbox', 'class', 'confidence'
        """
        # CARLA semantic labels: 10 = Vehicle
        vehicle_mask = (semantic_image[:, :, 2] == 10).astype(np.uint8) * 255
        
        # Find contours (connected components)
        contours, _ = cv2.findContours(vehicle_mask, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    'bbox': [x, y, x + w, y + h],
                    'class': 'vehicle',
                    'confidence': 0.9,  # Semantic segmentation = high confidence
                    'centroid_pixel': [x + w/2, y + h/2]
                })
        
        return detections
    
    def get_intrinsics(self):
        """Return camera intrinsics matrix."""
        return self.K
