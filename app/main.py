import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import cv2
from PIL import Image
import io
import numpy as np
from dotenv import load_dotenv

from core.emotion_detection.detector import EmotionDetector
from core.recommendation.recommender import MusicRecommender

# Load environment variables
load_dotenv()

# Initialize components
@st.cache_resource
def load_components():
    detector = EmotionDetector()
    recommender = MusicRecommender()
    return detector, recommender

# Page config
st.set_page_config(
    page_title="Emotion-Based Music Recommender",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .song-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .song-title {
        font-size: 1.2em;
        font-weight: bold;
        color: #333;
    }
    .song-artist {
        color: #666;
        font-style: italic;
    }
    .streaming-links {
        margin-top: 10px;
    }
    .streaming-links a {
        margin-right: 15px;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #333;'>🎵 Emotion-Based Music Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Upload a photo or use your webcam to get music recommendations based on your emotions!</p>", unsafe_allow_html=True)

# Initialize components
detector, recommender = load_components()

# Create columns for layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📸 Upload or Capture Photo")
    
    # Option to choose between webcam and upload
    input_method = st.radio("Choose input method:", ["Webcam", "Upload Photo"])
    
    if input_method == "Webcam":
        # Initialize webcam capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam. Please check your camera connection.")
        else:
            # Webcam capture
            if st.button("Take Photo"):
                try:
                    # Read frame from webcam
                    ret, frame = cap.read()
                    if ret:
                        # Convert frame to RGB for display
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, channels="RGB", caption="Captured Photo")
                        
                        # Detect emotion
                        emotion, processed_frame = detector.detect_emotion(frame)
                        
                        if emotion:
                            st.session_state['detected_emotion'] = emotion
                            st.session_state['processed_frame'] = processed_frame
                            st.success(f"Detected emotion: {emotion}")
                        else:
                            st.error("No face detected! Please try again.")
                    else:
                        st.error("Failed to capture image. Please try again.")
                finally:
                    # Always release the webcam
                    cap.release()
    else:
        # File upload
        uploaded_file = st.file_uploader("Upload a photo", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            try:
                # Convert to numpy array
                img = Image.open(uploaded_file)
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Detect emotion
                emotion, processed_frame = detector.detect_emotion(frame)
                
                if emotion:
                    st.session_state['detected_emotion'] = emotion
                    st.session_state['processed_frame'] = processed_frame
                    st.success(f"Detected emotion: {emotion}")
                else:
                    st.error("No face detected! Please try again.")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

with col2:
    st.markdown("### 🎵 Music Recommendations")
    
    if 'detected_emotion' in st.session_state:
        # Get recommendations
        recommendations = recommender.get_recommendations([st.session_state['detected_emotion']])
        
        # Display recommendations
        for song in recommendations:
            st.markdown(f"""
            <div class="song-card">
                <div class="song-title">{song['name']}</div>
                <div class="song-artist">{song['artist']}</div>
                <div class="streaming-links">
                    {f'<a href="{song["spotify_link"]}" target="_blank">🎵 Spotify</a>' if song['spotify_link'] else ''}
                    {f'<a href="{song["youtube_link"]}" target="_blank">▶️ YouTube</a>' if song['youtube_link'] else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Take a photo or upload an image to get music recommendations!")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>Made with ❤️ using Streamlit, TensorFlow, and Spotify API</p>", unsafe_allow_html=True) 