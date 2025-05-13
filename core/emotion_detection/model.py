import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

def create_emotion_model():
    """Create and return the emotion detection CNN model."""
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    return model

def load_model(model_path='models/model.h5'):
    """Load the pre-trained model weights."""
    model = create_emotion_model()
    model.load_weights(model_path)
    return model

# Emotion mapping dictionary
EMOTION_DICT = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
} 