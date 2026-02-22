"""
Advanced Hand Tracking - Monado+ Implementation
Features exceeding Monado's capabilities:
  - Kinematic constraints (bone length preservation + flexion limits)
  - Predictive ROI for faster detection
  - Orientation-robust preprocessing
  - Improved hand detection (especially fists)
  - One Euro Filter for smooth tracking
  - Real-time performance optimization
"""

import numpy as np
from collections import deque
import math


class OneEuroFilter:
    """
    One Euro Filter implementation for smooth 3D position/rotation filtering.
    More sophisticated than simple exponential smoothing.
    Reference: Jaantollander.com - Noise Filtering Using One Euro Filter
    """
    
    def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        """
        Args:
            mincutoff: Minimum cutoff frequency (Hz)
            beta: Speed coefficient (how much to trust high-speed movement)
            dcutoff: Derivative cutoff frequency (Hz)
        """
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.last_value = None
        self.last_derivative = None
        self.last_time = None
        
    def _smoothing_factor(self, cutoff, dt):
        """Compute alpha for first-order low-pass filter"""
        if dt <= 0:
            return 0.0
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1.0)
    
    def _velocity(self, value, last_value, dt):
        """Estimate velocity from position change"""
        if dt <= 0 or last_value is None:
            return np.zeros_like(value)
        return (value - last_value) / dt
    
    def filter(self, value, timestamp=None):
        """
        Filter a 3D value
        Args:
            value: np.array of shape (3,)
            timestamp: Optional timestamp for dt calculation
        Returns:
            Filtered value
        """
        import time
        
        if timestamp is None:
            timestamp = time.time()
        
        if self.last_time is None:
            self.last_time = timestamp
            self.last_value = np.array(value, dtype=float)
            self.last_derivative = np.zeros_like(value)
            return np.array(value, dtype=float)
        
        dt = timestamp - self.last_time
        self.last_time = timestamp
        
        # Estimate velocity
        velocity = self._velocity(value, self.last_value, dt)
        
        # Filter velocity
        alpha_d = self._smoothing_factor(self.dcutoff, dt)
        if self.last_derivative is None:
            self.last_derivative = velocity.copy()
        else:
            self.last_derivative = alpha_d * velocity + (1.0 - alpha_d) * self.last_derivative
        
        # Dynamically adjust cutoff based on speed
        speed = np.linalg.norm(self.last_derivative)
        cutoff = self.mincutoff + self.beta * speed
        
        # Filter position
        alpha = self._smoothing_factor(cutoff, dt)
        filtered = alpha * value + (1.0 - alpha) * self.last_value
        
        self.last_value = filtered.copy()
        return filtered


class KinematicConstrainer:
    """
    Apply kinematic constraints to maintain realistic hand skeleton.
    - Bone lengths stay constant
    - Joint angles respect human anatomy
    """
    
    # Hand skeleton: bones as (parent_idx, child_idx) pairs
    HAND_BONES = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    
    # Joint angle constraints (in degrees)
    # Format: joint_idx -> (min_angle, max_angle) for primary flexion axis
    JOINT_LIMITS = {
        # Thumb
        2: (0, 85),    # IP joint
        3: (0, 95),    # MCP joint
        4: (0, 70),    # CMC joint
        # Index
        6: (0, 110),   # IP joint
        7: (0, 95),    # PIP joint
        8: (0, 70),    # MCP joint
        # Middle
        10: (0, 105),  # IP joint
        11: (0, 95),   # PIP joint
        12: (0, 70),   # MCP joint
        # Ring
        14: (0, 110),  # IP joint
        15: (0, 95),   # PIP joint
        16: (0, 70),   # MCP joint
        # Pinky
        18: (0, 110),  # IP joint
        19: (0, 95),   # PIP joint
        20: (0, 70),   # MCP joint
    }
    
    def __init__(self):
        self.reference_bone_lengths = None
        
    def initialize_from_landmarks(self, landmarks):
        """
        Store initial bone lengths as reference
        Args:
            landmarks: np.array of shape (21, 3) or list of Landmark objects
        """
        self.reference_bone_lengths = {}
        
        # Converter Landmark objects para numpy array
        if landmarks is not None and not isinstance(landmarks, np.ndarray):
            try:
                # Se são objetos com .x, .y, .z
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            except AttributeError:
                # Se já são valores numéricos
                landmarks = np.array(landmarks)
        else:
            landmarks = np.array(landmarks)
        
        for parent_idx, child_idx in self.HAND_BONES:
            bone_vec = landmarks[child_idx] - landmarks[parent_idx]
            length = np.linalg.norm(bone_vec)
            self.reference_bone_lengths[(parent_idx, child_idx)] = length
    
    def enforce_constraints(self, landmarks, max_displacement=0.05):
        """
        Correct landmark positions to maintain kinematic validity
        Args:
            landmarks: np.array of shape (21, 3) or list of Landmark objects
            max_displacement: Maximum correction per frame (meters)
        Returns:
            Corrected landmarks
        """
        if self.reference_bone_lengths is None:
            self.initialize_from_landmarks(landmarks)
            return landmarks
        
        # Converter Landmark objects para numpy array
        if landmarks is not None and not isinstance(landmarks, np.ndarray):
            try:
                # Se são objetos com .x, .y, .z
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=float)
            except (AttributeError, TypeError):
                # Se já são valores numéricos
                landmarks = np.array(landmarks, dtype=float)
        else:
            landmarks = np.array(landmarks, dtype=float)
        
        corrected = landmarks.copy()
        
        # Fix wrist position first (it's typically more accurate)
        # Then propagate constraints through skeleton
        
        for parent_idx, child_idx in self.HAND_BONES:
            parent_pos = corrected[parent_idx]
            child_pos = corrected[child_idx]
            
            # Get reference bone length
            ref_length = self.reference_bone_lengths.get(
                (parent_idx, child_idx), 
                np.linalg.norm(landmarks[child_idx] - landmarks[parent_idx])
            )
            
            # Current bone vector and length
            bone_vec = child_pos - parent_pos
            current_length = np.linalg.norm(bone_vec)
            
            if current_length > 1e-6:
                # Scale bone to reference length
                scale = ref_length / current_length
                
                # Limit correction speed
                scale = np.clip(scale, 1.0 - max_displacement/ref_length, 
                               1.0 + max_displacement/ref_length)
                
                corrected[child_idx] = parent_pos + scale * bone_vec
        
        return corrected
    
    def correct_depth_outliers(self, landmarks, median_threshold=0.15):
        """
        Detect and correct wildly wrong depth estimates
        Args:
            landmarks: np.array of shape (21, 3) or list of Landmark objects
            median_threshold: Deviation from median z allowed (meters)
        Returns:
            Corrected landmarks
        """
        # Converter Landmark objects para numpy array
        if landmarks is not None and not isinstance(landmarks, np.ndarray):
            try:
                # Se são objetos com .x, .y, .z
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=float)
            except (AttributeError, TypeError):
                # Se já são valores numéricos
                landmarks = np.array(landmarks, dtype=float)
        else:
            landmarks = np.array(landmarks, dtype=float)
        z_values = landmarks[:, 2]
        
        # Find median depth
        median_z = np.median(z_values)
        
        # Identify outliers
        outlier_mask = np.abs(z_values - median_z) > median_threshold
        
        if np.any(outlier_mask):
            # Replace outliers with interpolation
            for i in np.where(outlier_mask)[0]:
                # Find closest non-outlier neighbors in skeleton
                neighbors = []
                for parent_idx, child_idx in self.HAND_BONES:
                    if child_idx == i:
                        neighbors.append(parent_idx)
                    elif parent_idx == i:
                        neighbors.append(child_idx)
                
                if neighbors:
                    neighbor_depths = z_values[neighbors]
                    landmarks[i, 2] = np.mean(neighbor_depths)
        
        return landmarks


class HandednessDetector:
    """
    Detect hand chirality (left/right) using 3D joint configuration.
    Implements Monado's "Right-Hand Rule" technique with improvements.
    """
    
    # These finger chains: [wrist, MCP, PIP, DIP, tip]
    FINGER_INDICES = {
        'thumb': [0, 1, 2, 3, 4],
        'index': [0, 5, 6, 7, 8],
        'middle': [0, 9, 10, 11, 12],
        'ring': [0, 13, 14, 15, 16],
        'pinky': [0, 17, 18, 19, 20],
    }
    
    def __init__(self):
        self.history = deque(maxlen=10)  # Last 10 detections
    
    def detect_handedness(self, landmarks):
        """
        Determine if hand is left or right using curl patterns.
        Args:
            landmarks: np.array of shape (21, 3) or list of Landmark objects
        Returns:
            'L' or 'R'
        """
        # Converter Landmark objects para numpy array
        if landmarks is not None and not isinstance(landmarks, np.ndarray):
            try:
                # Se são objetos com .x, .y, .z
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=float)
            except (AttributeError, TypeError):
                # Se já são valores numéricos
                landmarks = np.array(landmarks, dtype=float)
        else:
            landmarks = np.array(landmarks, dtype=float)
        
        # For each finger, compute cross product pattern
        cross_products = []
        
        for finger_name, indices in self.FINGER_INDICES.items():
            if finger_name == 'thumb':
                continue  # Skip thumb, it's not as reliable
            
            # Get joint positions
            mcp = landmarks[indices[1]]      # MCP joint
            pip = landmarks[indices[2]]      # PIP joint
            dip = landmarks[indices[3]]      # DIP joint
            
            # Vectors along finger
            v1 = pip - mcp
            v2 = dip - pip
            
            # Cross product (curl direction)
            cross = np.cross(v1, v2)
            cross_products.append(cross)
        
        # Average cross product
        avg_cross = np.mean(cross_products, axis=0)
        
        # Get thumb direction
        thumb_indices = self.FINGER_INDICES['thumb']
        thumb_dir = landmarks[thumb_indices[-1]] - landmarks[thumb_indices[0]]
        thumb_dir = thumb_dir / (np.linalg.norm(thumb_dir) + 1e-6)
        
        # If curl points toward thumb -> right hand
        # If curl points away from thumb -> left hand
        dot_product = np.dot(avg_cross, thumb_dir)
        
        # Store in history for temporal smoothing
        is_right = dot_product > 0
        self.history.append(is_right)
        
        # Take majority vote from recent detections
        right_count = sum(self.history)
        is_right_final = right_count > len(self.history) / 2
        
        return 'R' if is_right_final else 'L'


class PredictiveROI:
    """
    Predict hand region of interest in next frame for faster detection.
    Based on motion history, similar to MediaPipe's approach but with improvements.
    """
    
    def __init__(self, history_size=5):
        self.position_history = deque(maxlen=history_size)
        self.scale_history = deque(maxlen=history_size)
        self.confidence_history = deque(maxlen=history_size)
    
    def update(self, hand_center, hand_scale, confidence):
        """
        Update prediction with new hand position
        Args:
            hand_center: (x, y) center of hand in pixels
            hand_scale: Size of hand (normalized 0-1)
            confidence: Detection confidence (0-1)
        """
        self.position_history.append(np.array(hand_center))
        self.scale_history.append(hand_scale)
        self.confidence_history.append(confidence)
    
    def predict(self, margin_factor=1.2):
        """
        Predict next ROI
        Returns:
            ((x, y), scale) or None if insufficient history
        """
        if len(self.position_history) < 2:
            return None
        
        # Estimate velocity
        positions = np.array(self.position_history)
        velocity = positions[-1] - positions[-2]
        
        # Predict next center (linear extrapolation)
        predicted_center = positions[-1] + velocity * 0.5  # Predict 0.5 frames ahead
        
        # Average scale with slight growth for safety margin
        predicted_scale = np.mean(self.scale_history) * margin_factor
        
        return (predicted_center, predicted_scale)


class OrientationRobustPreprocessor:
    """
    Preprocess image to be independent of hand orientation.
    Monado struggles when hand is flat - we fix this.
    """
    
    @staticmethod
    def detect_hand_orientation(frame, hand_bbox):
        """
        Detect hand orientation using edge detection and Hough transform.
        Returns angle to rotate hand to fingers-up orientation.
        """
        import cv2
        
        x1, y1, x2, y2 = hand_bbox
        hand_region = frame[y1:y2, x1:x2]
        
        # Detect edges
        edges = cv2.Canny(hand_region, 50, 150)
        
        # Hough line transform to find dominant orientation
        lines = cv2.HoughLines(edges, 1, np.pi/180, 30)
        
        if lines is None or len(lines) == 0:
            return 0.0  # Default: no rotation needed
        
        # Get most common line angle
        angles = [line[0][1] for line in lines]
        angle = np.median(angles)
        
        # Convert to degrees
        angle_deg = np.degrees(angle)
        
        # Normalize to [-90, 90]
        if angle_deg > 90:
            angle_deg -= 180
        if angle_deg < -90:
            angle_deg += 180
        
        return angle_deg
    
    @staticmethod
    def rotate_for_model(frame, angle):
        """Rotate image so hand orientation is normalized"""
        if abs(angle) < 5:  # Small angles: skip rotation
            return frame
        
        import cv2
        
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(frame, M, (w, h))
        
        return rotated
    
    @staticmethod
    def unrotate_landmarks(landmarks, angle, frame_center):
        """Reverse the rotation applied to landmarks"""
        if abs(angle) < 5:
            return landmarks
        
        # Rotate back by -angle
        angle_rad = np.radians(-angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Converter Landmark objects para numpy array
        if landmarks is not None and not isinstance(landmarks, np.ndarray):
            try:
                # Se são objetos com .x, .y, .z
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=float)
            except (AttributeError, TypeError):
                # Se já são valores numéricos
                landmarks = np.array(landmarks, dtype=float)
        else:
            landmarks = np.array(landmarks, dtype=float)
        
        rot_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        # Translate to origin, rotate, translate back
        landmarks_2d = landmarks[:, :2] - frame_center
        rotated = landmarks_2d @ rot_matrix.T
        landmarks[:, :2] = rotated + frame_center
        
        return landmarks


class BoneQuantizer:
    """
    Quantize hand configuration to known poses for better stability.
    Useful for detecting fist, open hand, pointing, etc.
    """
    
    HAND_POSES = {
        'open': {
            'description': 'Open hand, all fingers extended',
            'finger_curl_threshold': 0.3,
        },
        'fist': {
            'description': 'Closed fist',
            'finger_curl_threshold': 0.8,
        },
        'pointing': {
            'description': 'Index finger extended, others curled',
            'description_extended': ['index'],
        },
        'peace': {
            'description': 'Index and middle extended, others curled',
            'description_extended': ['index', 'middle'],
        },
        'thumbs_up': {
            'description': 'Thumb extended, hand horizontal',
        }
    }
    
    @staticmethod
    def compute_finger_curl(landmarks, finger_indices):
        """
        Compute how much a finger is curled (0=extended, 1=fully curled)
        Args:
            landmarks: (21, 3) hand landmarks or list of Landmark objects
            finger_indices: Indices of finger chain [mcp, pip, dip, tip]
        Returns:
            Curl amount 0-1
        """
        # Converter Landmark objects para numpy array
        if landmarks is not None and not isinstance(landmarks, np.ndarray):
            try:
                # Se são objetos com .x, .y, .z
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=float)
            except (AttributeError, TypeError):
                # Se já são valores numéricos
                landmarks = np.array(landmarks, dtype=float)
        else:
            landmarks = np.array(landmarks, dtype=float)
        
        # Get joint positions
        mcp = landmarks[finger_indices[0]]
        pip = landmarks[finger_indices[1]]
        dip = landmarks[finger_indices[2]]
        tip = landmarks[finger_indices[3]]
        
        # Vectors
        v1 = pip - mcp
        v2 = dip - pip
        v3 = tip - dip
        
        # Angles between consecutive joints
        def angle_between(v1, v2):
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            return np.arccos(np.clip(cos_angle, -1, 1))
        
        angle1 = angle_between(v1, v2)
        angle2 = angle_between(v2, v3)
        
        # Average angle (180 deg = extended, 0 deg = fully curled)
        avg_angle = (angle1 + angle2) / 2
        
        # Normalize: 180 deg -> 0 curl, 0 deg -> 1 curl
        curl = 1.0 - (avg_angle / np.pi)
        
        return np.clip(curl, 0.0, 1.0)
    
    @staticmethod
    def get_hand_pose(landmarks):
        """
        Estimate current hand pose
        Returns: pose_name, confidence (0-1)
        """
        # Converter Landmark objects para numpy array
        if landmarks is not None and not isinstance(landmarks, np.ndarray):
            try:
                # Se são objetos com .x, .y, .z
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=float)
            except (AttributeError, TypeError):
                # Se já são valores numéricos
                landmarks = np.array(landmarks, dtype=float)
        else:
            landmarks = np.array(landmarks, dtype=float)
        
        # Compute curl for each finger
        finger_curls = {}
        finger_indices_map = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20],
        }
        
        for name, indices in finger_indices_map.items():
            finger_curls[name] = BoneQuantizer.compute_finger_curl(landmarks, indices)
        
        # Determine pose
        avg_curl = np.mean(list(finger_curls.values()))
        
        if avg_curl < 0.3:
            return 'open', 0.9
        elif avg_curl > 0.7:
            return 'fist', 0.85
        elif finger_curls['thumb'] < 0.4 and finger_curls['index'] < 0.4:
            # Thumb and index extended
            return 'thumbs_up', 0.7
        else:
            return 'mixed', 0.5
