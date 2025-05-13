# Emotion-Based Music Recommendation System - Technical Documentation

## Project Overview
This project implements an emotion-based music recommendation system that uses facial emotion detection to suggest music tracks from Spotify. The system captures facial expressions through a webcam, processes them using a Convolutional Neural Network (CNN), and recommends music based on the detected emotions.

## Project Structure
```
Emotion-based-music-recommendation-Modify/
├── app/                    # Streamlit application
│   ├── main.py            # Main application interface
│   └── __init__.py
├── core/                   # Core functionality
│   ├── emotion/           # Emotion detection module
│   │   ├── detector.py    # Emotion detection logic
│   │   ├── model.py       # CNN model definition
│   │   └── __init__.py
│   └── recommendation/    # Music recommendation module
│       ├── recommender.py # Music recommendation logic
│       ├── music_data.py  # Music data processing
│       └── __init__.py
├── data/                   # Data storage
│   └── muse_v3.csv        # Music dataset
├── models/                 # Model files
│   ├── model.h5           # Trained CNN model
│   └── haarcascade_frontalface_default.xml  # Face detection model
├── .env                   # Environment variables (Spotify API credentials)
├── requirements.txt       # Project dependencies
└── PROJECT_DOCUMENTATION.md  # This documentation file
```

### Directory Descriptions
- **app/**: Contains the Streamlit web application
  - `main.py`: Main application interface with UI components
  - `__init__.py`: Package initialization

- **core/**: Core functionality modules
  - **emotion/**: Emotion detection implementation
    - `detector.py`: Face detection and emotion classification
    - `model.py`: CNN model architecture and loading
    - `__init__.py`: Package initialization
  - **recommendation/**: Music recommendation system
    - `recommender.py`: Spotify integration and recommendation logic
    - `music_data.py`: Music dataset processing
    - `__init__.py`: Package initialization

- **data/**: Data storage directory
  - `muse_v3.csv`: Music dataset with emotion tags

- **models/**: Pre-trained models
  - `model.h5`: Trained CNN model for emotion detection
  - `haarcascade_frontalface_default.xml`: OpenCV face detection model

- **Root Files**:
  - `.env`: Environment variables for API credentials
  - `requirements.txt`: Python package dependencies
  - `PROJECT_DOCUMENTATION.md`: Technical documentation

## Technologies Used

### 1. Core Technologies
- **Python 3.12**: Primary programming language
- **Streamlit**: Web application framework for creating interactive UIs
- **TensorFlow**: Deep learning framework for emotion detection
- **OpenCV**: Computer vision library for face detection and image processing
- **Spotify API**: Music streaming service integration
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Python-dotenv**: Environment variable management

### 2. Machine Learning & AI
- **Convolutional Neural Network (CNN)**: For emotion detection
- **Haar Cascade Classifier**: For face detection
- **Transfer Learning**: Pre-trained model for emotion classification

### 3. APIs and Services
- **Spotify Web API**: For music recommendations and track information
- **Streamlit Cloud**: For web application deployment (optional)

## System Architecture

### 1. Frontend (Streamlit Application)
```
app/
├── main.py           # Main application interface
└── __init__.py
```
- **UI Components**:
  - Webcam capture interface
  - Image upload functionality
  - Real-time emotion display
  - Music recommendation cards
  - Spotify integration links

- **State Management**:
  - Uses Streamlit's session state
  - Caches components for performance
  - Manages webcam resources

### 2. Core Components
```
core/
├── emotion/          # Emotion detection module
│   ├── detector.py   # Emotion detection logic
│   ├── model.py      # CNN model definition
│   └── __init__.py
└── recommendation/   # Music recommendation module
    ├── recommender.py # Music recommendation logic
    ├── music_data.py  # Music data processing
    └── __init__.py
```

## Technical Components

### 1. Emotion Detection Module

#### CNN Model Architecture (`model.py`)
```python
model = Sequential([
    # Input Layer: 48x48 grayscale images
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)),
    
    # Feature Extraction Layers
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Regularization
    Dropout(0.25),
    
    # Classification Layers
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotion classes
])
```

#### Emotion Detection Process
1. **Face Detection**:
   - Uses OpenCV's Haar Cascade Classifier
   - Parameters:
     - Scale factor: 1.3
     - Minimum neighbors: 5
     - Minimum face size: 20x20 pixels
   - Process:
     ```python
     # Face detection pipeline
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
     ```

2. **Emotion Classification**:
   - Preprocessing:
     - Grayscale conversion
     - 48x48 resizing
     - Normalization (0-1)
   - CNN Prediction:
     - 7 emotion classes
     - Softmax activation for probability distribution
   - Emotion Mapping:
     ```python
     EMOTION_DICT = {
         0: "Angry",
         1: "Disgusted",
         2: "Fearful",
         3: "Happy",
         4: "Neutral",
         5: "Sad",
         6: "Surprised"
     }
     ```

### 2. Music Recommendation Module

#### Data Processing (`music_data.py`)
1. **Dataset Structure**:
   - Million Song Dataset (MUSE v3)
   - Columns:
     - track: Song name
     - artist: Artist name
     - lastfm_url: Music link
     - number_of_emotion_tags: Emotional intensity
     - valence_tags: Pleasantness score

2. **Emotion-Based Categorization**:
   ```python
   self.emotion_dfs = {
       'Sad': self.df[:chunk_size],
       'Fear': self.df[chunk_size:2*chunk_size],
       'Angry': self.df[2*chunk_size:3*chunk_size],
       'Neutral': self.df[3*chunk_size:4*chunk_size],
       'Happy': self.df[4*chunk_size:]
   }
   ```

#### Recommendation Algorithm (`recommender.py`)
1. **Emotion Mapping**:
   ```python
   self.emotion_mapping = {
       'Angry': 'Angry',
       'Disgusted': 'Angry',
       'Fearful': 'Fear',
       'Happy': 'Happy',
       'Neutral': 'Neutral',
       'Sad': 'Sad',
       'Surprised': 'Happy'
   }
   ```

2. **Recommendation Process**:
   - Emotion Processing:
     - Counts emotion frequencies
     - Maps to available categories
     - Orders by frequency
   
   - Song Selection:
     - Dynamic distribution based on emotion count:
       ```python
       if n_emotions == 1:
           counts = [10]
       elif n_emotions == 2:
           counts = [6, 4]
       elif n_emotions == 3:
           counts = [5, 3, 2]
       else:
           counts = [3] * n_emotions
       ```

3. **Spotify Integration**:
   - Authentication:
     - Client credentials flow
     - Environment variable management
   - Track Search:
     - Query format: "track:{name} artist:{artist}"
     - Limit: 1 result per track
   - Link Generation:
     - Spotify web player links
     - Direct track access

## Data Flow

1. **Image Capture**:
   ```
   Webcam/Upload → OpenCV Processing → Face Detection → Emotion Classification
   ```

2. **Emotion Processing**:
   ```
   Raw Emotions → Frequency Analysis → Category Mapping → Emotion Distribution
   ```

3. **Music Recommendation**:
   ```
   Emotion Distribution → Song Selection → Spotify Integration → Recommendation Display
   ```

## Performance Considerations

### 1. Real-time Processing
- **Video Capture**:
  - 20 frames per detection
  - Efficient face detection
  - Optimized emotion classification

### 2. Memory Management
- **Data Loading**:
  - Cached components
  - Efficient DataFrame operations
  - Streamlit session state

### 3. API Optimization
- **Spotify API**:
  - Rate limiting handling
  - Error recovery
  - Caching of results

## Security Measures

1. **API Security**:
   - Environment variables for credentials
   - Secure credential storage
   - API key rotation capability

2. **Data Privacy**:
   - Local processing of images
   - No permanent storage of user data
   - Secure session management

## Future Improvements

### 1. Technical Enhancements
- **Model Improvements**:
  - Implement emotion confidence thresholds
  - Add emotion intensity detection
  - Fine-tune CNN architecture
  - Add transfer learning capabilities

- **Recommendation System**:
  - Implement collaborative filtering
  - Add user preference learning
  - Improve recommendation diversity
  - Add genre-based filtering

### 2. Feature Additions
- **User Features**:
  - User authentication
  - Playlist creation
  - Emotion history tracking
  - Custom emotion-music mappings

- **UI/UX Improvements**:
  - Real-time emotion visualization
  - Interactive music player
  - Emotion trend analysis
  - Customizable themes

### 3. Performance Optimizations
- **Model Optimization**:
  - Model quantization
  - Batch processing
  - GPU acceleration
  - Edge computing support

- **System Optimization**:
  - Caching improvements
  - API rate limiting handling
  - Load balancing
  - Distributed processing

## Development Guidelines

### 1. Code Structure
- Modular design
- Clear separation of concerns
- Consistent naming conventions
- Comprehensive documentation

### 2. Testing
- Unit tests for core components
- Integration tests for API interactions
- Performance benchmarking
- User acceptance testing

### 3. Deployment
- Environment setup
- Dependency management
- Configuration management
- Monitoring and logging

## Troubleshooting Guide

### 1. Common Issues
- Webcam access problems
- API rate limiting
- Model loading errors
- Memory management issues

### 2. Debugging Tools
- Logging system
- Error tracking
- Performance monitoring
- User feedback system

## API Documentation

### 1. Spotify API
- Authentication
- Track search
- Playlist management
- Rate limits

### 2. Streamlit Components
- Session management
- State handling
- Component lifecycle
- Event handling 