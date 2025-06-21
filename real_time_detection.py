import cv2
import numpy as np
import time
from hand_detector import HandDetector
from model import SignLanguageModel
from data_loader import SignLanguageDataLoader

class RealTimeSignLanguageDetector:
    def __init__(self, model_path=None):
        self.hand_detector = HandDetector()
        self.model = SignLanguageModel()
        self.classes = ['hello', 'thanks', 'yes', 'no', 'iloveu', 'sad', 'happy']
        
        if model_path:
            self.model.load_model(model_path)
        else:
            print("Warning: No model path provided. Please train a model first.")
        
        self.prediction_history = []
        self.history_length = 5
        self.confidence_threshold = 0.7
        
    def smooth_predictions(self, prediction):
        self.prediction_history.append(prediction)
        
        if len(self.prediction_history) > self.history_length:
            self.prediction_history.pop(0)
        
        avg_prediction = np.mean(self.prediction_history, axis=0)
        return avg_prediction
    
    def get_prediction_label(self, prediction):
        class_idx = np.argmax(prediction)
        confidence = prediction[class_idx]
        
        if confidence > self.confidence_threshold:
            return self.classes[class_idx], confidence
        else:
            return "Unknown", confidence
    
    def run_detection(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        window_name = 'Sign Language Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        print("Real-time sign language detection started!")
        print("Press 'q' to quit, 's' to save screenshot, 'f' to toggle fullscreen")
        
        fullscreen = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            frame = cv2.flip(frame, 1)
            
            frame_with_landmarks, hands = self.hand_detector.find_hands(frame, draw=True)
            
            if hands and self.model.model is not None:
                for i, hand_landmarks in enumerate(hands):
                    hand_roi = self.hand_detector.get_hand_roi(frame, hand_landmarks, padding=30)
                    
                    if hand_roi is not None and hand_roi.size > 0:
                        processed_hand = self.hand_detector.preprocess_hand_image(hand_roi)
                        
                        if processed_hand is not None:
                            prediction = self.model.predict(processed_hand)
                            
                            smoothed_prediction = self.smooth_predictions(prediction)
                            
                            label, confidence = self.get_prediction_label(smoothed_prediction)
                            
                            self.draw_prediction(frame_with_landmarks, label, confidence, i)
                            
                            self.draw_hand_roi(frame_with_landmarks, hand_landmarks)
            
            self.draw_instructions(frame_with_landmarks)
            
            cv2.imshow(window_name, frame_with_landmarks)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame_with_landmarks)
                print(f"Screenshot saved as {filename}")
            elif key == ord('f'):
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                print(f"Fullscreen: {'ON' if fullscreen else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()
        self.hand_detector.release()
    
    def draw_prediction(self, frame, label, confidence, hand_index=0):
        h, w = frame.shape[:2]
        
        font_scale = max(1.0, w / 800)
        thickness = max(2, int(w / 400))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        if confidence > 0.8:
            color = (0, 255, 0)
        elif confidence > 0.6:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        
        text = f"{label}: {confidence:.2f}"
        
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        x = int(w * 0.02)
        y = int(h * 0.05) + hand_index * int(h * 0.08)
        
        cv2.rectangle(frame, (x - 5, y - text_height - 5), 
                     (x + text_width + 5, y + baseline + 5), (0, 0, 0), -1)
        
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
    
    def draw_hand_roi(self, frame, landmarks):
        if not landmarks:
            return
        
        h, w, _ = frame.shape
        pixel_landmarks = []
        for lm in landmarks:
            x, y = int(lm[0] * w), int(lm[1] * h)
            pixel_landmarks.append([x, y])
        
        x_coords = [lm[0] for lm in pixel_landmarks]
        y_coords = [lm[1] for lm in pixel_landmarks]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        padding = int(max(w, h) * 0.02)
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        thickness = max(2, int(max(w, h) / 400))
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), thickness)
    
    def draw_instructions(self, frame):
        instructions = [
            "Press 'q' to quit",
            "Press 's' to save screenshot",
            "Press 'f' to toggle fullscreen",
            "Show your hand sign to the camera"
        ]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        h, w = frame.shape[:2]
        
        font_scale = max(0.6, w / 1200)
        thickness = max(1, int(w / 800))
        line_spacing = int(h * 0.025)
        
        color = (255, 255, 255)
        
        for i, instruction in enumerate(instructions):
            y = h - int(h * 0.08) + i * line_spacing
            cv2.putText(frame, instruction, (int(w * 0.02), y), font, font_scale, color, thickness)

def train_and_run():
    print("Loading training data...")
    data_loader = SignLanguageDataLoader("train", "test")
    train_images, train_labels, test_images, test_labels = data_loader.load_train_test_data()
    
    if len(train_images) == 0 or len(test_images) == 0:
        print("Error: No training or test data found!")
        return
    
    print("Training the model...")
    model = SignLanguageModel()
    model.build_simple_cnn()
    
    history = model.train(train_images, train_labels, test_images, test_labels, epochs=20)
    
    model.evaluate(test_images, test_labels)
    
    model.save_model("sign_language_model.h5")
    
    print("Model training completed! Starting real-time detection...")
    
    detector = RealTimeSignLanguageDetector("sign_language_model.h5")
    detector.run_detection()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        train_and_run()
    else:
        try:
            detector = RealTimeSignLanguageDetector("sign_language_model.h5")
            detector.run_detection()
        except FileNotFoundError:
            print("No trained model found. Training a new model...")
            train_and_run() 