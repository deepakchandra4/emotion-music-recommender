import cv2
import numpy as np
from .model import load_model, EMOTION_DICT

class EmotionDetector:
    def __init__(self, model_path='models/model.h5', cascade_path='core/emotion_detection/haarcascade_frontalface_default.xml'):
        """Initialize the emotion detector with model and face cascade."""
        self.model = load_model(model_path)
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise ValueError("Failed to load face cascade classifier")

    def detect_emotion(self, frame):
        """
        Detect emotion from a single frame.
        Returns: (emotion, processed_frame) or (None, None) if no face detected
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) == 0:
            return None, None
            
        # Process the first face found
        x, y, w, h = faces[0]
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        
        # Extract and process face region
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        
        # Predict emotion
        prediction = self.model.predict(cropped_img)
        max_index = int(np.argmax(prediction))
        emotion = EMOTION_DICT[max_index]
        
        # Add emotion text to frame
        cv2.putText(frame, emotion, (x + 20, y - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        return emotion, frame 