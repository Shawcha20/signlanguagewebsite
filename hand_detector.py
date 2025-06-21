import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional

class HandDetector:
    def __init__(self, static_mode=False, max_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.landmark_indices = {
            'wrist': 0,
            'thumb_tip': 4,
            'index_tip': 8,
            'middle_tip': 12,
            'ring_tip': 16,
            'pinky_tip': 20
        }
    
    def find_hands(self, image: np.ndarray, draw: bool = True) -> Tuple[np.ndarray, List]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(image_rgb)
        
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        all_hands = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                
                all_hands.append(landmarks)
        
        return image, all_hands
    
    def get_hand_roi(self, image: np.ndarray, landmarks: List, padding: int = 50) -> Optional[np.ndarray]:
        if not landmarks:
            return None
        
        h, w, _ = image.shape
        pixel_landmarks = []
        for lm in landmarks:
            x, y = int(lm[0] * w), int(lm[1] * h)
            pixel_landmarks.append([x, y])
        
        x_coords = [lm[0] for lm in pixel_landmarks]
        y_coords = [lm[1] for lm in pixel_landmarks]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        hand_roi = image[y_min:y_max, x_min:x_max]
        
        return hand_roi
    
    def preprocess_hand_image(self, hand_image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        if hand_image is None or hand_image.size == 0:
            return None
        
        resized = cv2.resize(hand_image, target_size)
        
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        normalized = rgb.astype(np.float32) / 255.0
        
        return normalized
    
    def get_hand_gesture_features(self, landmarks: List) -> List[float]:
        if not landmarks:
            return []
        
        features = []
        
        wrist = np.array(landmarks[self.landmark_indices['wrist']])
        thumb_tip = np.array(landmarks[self.landmark_indices['thumb_tip']])
        index_tip = np.array(landmarks[self.landmark_indices['index_tip']])
        middle_tip = np.array(landmarks[self.landmark_indices['middle_tip']])
        ring_tip = np.array(landmarks[self.landmark_indices['ring_tip']])
        pinky_tip = np.array(landmarks[self.landmark_indices['pinky_tip']])
        
        features.extend([
            np.linalg.norm(wrist - thumb_tip),
            np.linalg.norm(wrist - index_tip),
            np.linalg.norm(wrist - middle_tip),
            np.linalg.norm(wrist - ring_tip),
            np.linalg.norm(wrist - pinky_tip)
        ])
        
        features.extend([
            np.linalg.norm(thumb_tip - index_tip),
            np.linalg.norm(index_tip - middle_tip),
            np.linalg.norm(middle_tip - ring_tip),
            np.linalg.norm(ring_tip - pinky_tip)
        ])
        
        return features
    
    def release(self):
        self.hands.close()

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame, hands = detector.find_hands(frame)
        
        cv2.imshow('Hand Detection Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.release() 