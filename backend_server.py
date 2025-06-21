import cv2
import numpy as np
import time
import threading
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from hand_detector import HandDetector
from model import SignLanguageModel
from data_loader import SignLanguageDataLoader

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

class WebSocketSignLanguageDetector:
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
        self.is_running = False
        self.camera_thread = None
        
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
            return "No gesture detected", confidence
    
    def camera_loop(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            socketio.emit('error', {'message': 'Could not open camera'})
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Camera started for WebSocket detection")
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            frame = cv2.flip(frame, 1)
            
            frame_with_landmarks, hands = self.hand_detector.find_hands(frame, draw=True)
            
            current_gesture = "No gesture detected"
            current_confidence = 0.0
            
            if hands and self.model.model is not None:
                for hand_landmarks in hands:
                    hand_roi = self.hand_detector.get_hand_roi(frame, hand_landmarks, padding=30)
                    
                    if hand_roi is not None and hand_roi.size > 0:
                        processed_hand = self.hand_detector.preprocess_hand_image(hand_roi)
                        
                        if processed_hand is not None:
                            prediction = self.model.predict(processed_hand)
                            
                            smoothed_prediction = self.smooth_predictions(prediction)
                            
                            label, confidence = self.get_prediction_label(smoothed_prediction)
                            
                            current_gesture = label
                            current_confidence = float(confidence)
            
            # Emit the gesture data to all connected clients
            socketio.emit('gesture_update', {
                'gesture': current_gesture,
                'confidence': current_confidence
            })
            
            time.sleep(0.1)  # 10 FPS
        
        cap.release()
        print("Camera stopped")
    
    def start_detection(self):
        if not self.is_running:
            self.is_running = True
            self.camera_thread = threading.Thread(target=self.camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            print("Detection started")
    
    def stop_detection(self):
        self.is_running = False
        if self.camera_thread:
            self.camera_thread.join(timeout=1)
        print("Detection stopped")

# Global detector instance
detector = None

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to sign language detection server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_detection')
def handle_start_detection():
    global detector
    if detector is None:
        try:
            detector = WebSocketSignLanguageDetector("sign_language_model.h5")
        except FileNotFoundError:
            emit('error', {'message': 'No trained model found. Please train the model first.'})
            return
    
    detector.start_detection()
    emit('status', {'message': 'Detection started'})

@socketio.on('stop_detection')
def handle_stop_detection():
    global detector
    if detector:
        detector.stop_detection()
    emit('status', {'message': 'Detection stopped'})

def train_model_if_needed():
    """Train the model if it doesn't exist"""
    try:
        # Try to load the model
        test_model = SignLanguageModel()
        test_model.load_model("sign_language_model.h5")
        print("Model found, skipping training")
        return True
    except FileNotFoundError:
        print("No trained model found. Training a new model...")
        try:
            data_loader = SignLanguageDataLoader("train", "test")
            train_images, train_labels, test_images, test_labels = data_loader.load_train_test_data()
            
            if len(train_images) == 0 or len(test_images) == 0:
                print("Error: No training or test data found!")
                return False
            
            print("Training the model...")
            model = SignLanguageModel()
            model.build_simple_cnn()
            
            history = model.train(train_images, train_labels, test_images, test_labels, epochs=20)
            model.evaluate(test_images, test_labels)
            model.save_model("sign_language_model.h5")
            
            print("Model training completed!")
            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False

if __name__ == '__main__':
    # Train model if needed
    if not train_model_if_needed():
        print("Failed to train model. Exiting.")
        exit(1)
    
    print("Starting WebSocket server on http://localhost:5000")
    print("The Next.js frontend should connect to this server")
    
    # Start the detection automatically
    detector = WebSocketSignLanguageDetector("sign_language_model.h5")
    detector.start_detection()
    
    # Run the Flask-SocketIO server
    socketio.run(app, host='0.0.0.0', port=5000, debug=False) 