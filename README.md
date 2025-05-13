# Emotion-Based Music Recommender

A web application that detects emotions from facial expressions and recommends music based on the detected emotions. The application uses deep learning for emotion detection and integrates with Spotify and YouTube for music recommendations.

## Features

- Real-time emotion detection using webcam or photo upload
- Music recommendations based on detected emotions
- Direct links to Spotify and YouTube for recommended songs
- Modern and user-friendly interface
- Support for multiple emotions and weighted recommendations

## Project Structure

```
Emotion-Music-Recommender/
│
├── app/                          # Main application (Streamlit)
│   ├── __init__.py
│   ├── main.py                   # Streamlit UI & core app flow
│   └── assets/                   # CSS/images (if any)
│
├── core/                         # Core logic
│   ├── emotion_detection/        
│   │   ├── model.py              # CNN model definition/loading
│   │   ├── detector.py           # Face detection & emotion prediction
│   │   └── haarcascade_frontalface_default.xml
│   │
│   ├── recommendation/
│   │   ├── recommender.py        # Music recommendation logic
│   │   └── music_data.py         # CSV loading & preprocessing
│   │
│   └── utils.py                  # Helper functions
│
├── data/
│   └── muse_v3.csv              # Music dataset
│
├── models/
│   └── model.h5                 # Pretrained emotion model
│
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/deepakchandra4/emotion-music-recommender.git
cd emotion-music-recommender
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Spotify API credentials:
   - Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
   - Create a new application
   - Copy the Client ID and Client Secret
   - Create a `.env` file in the root directory
   - Add your credentials:
     ```
     SPOTIFY_CLIENT_ID=your_client_id_here
     SPOTIFY_CLIENT_SECRET=your_client_secret_here
     ```

5. Run the application:
```bash
streamlit run app/main.py
```

## Usage

1. Launch the application using the command above
2. Choose between webcam capture or photo upload
3. Take a photo or upload an image
4. View the detected emotion and music recommendations
5. Click on Spotify or YouTube links to listen to recommended songs

## Requirements

- Python 3.8+
- Webcam (for real-time emotion detection)
- Internet connection (for music recommendations)
- Spotify API credentials

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Installation & Run

Create a new project in Pycharm and add the above files. After that open the terminal and run the following command. This will install all the modules needed to run this app. 

```bash
  pip install -r requirements.txt
```

To run the app, type the following command in the terminal. 
```bash
  streamlit run app.py
```

## Libraries

- Streamlit
- Opencv
- Numpy
- Pandas
- Tensorflow
- Keras





## Demo video

 [Emotion-based music recommendation system](



)
 

## Authors

- [DEEPAK CHANDRA MAURYA](https://github.com/deepakchandra4)



## Support

For support,Deepak Chandra Maurya (https://www.linkedin.com/in/deepak-chandra-maurya-a03a21266//)

