from collections import Counter
from typing import List, Dict
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from pytube import Search
import os
from dotenv import load_dotenv
from .music_data import MusicData

load_dotenv()

class MusicRecommender:
    def __init__(self):
        """Initialize the music recommender with Spotify API client."""
        self.music_data = MusicData()
        self.spotify = spotipy.Spotify(
            client_credentials_manager=SpotifyClientCredentials(
                client_id=os.getenv('SPOTIFY_CLIENT_ID'),
                client_secret=os.getenv('SPOTIFY_CLIENT_SECRET')
            )
        )
        # Map detected emotions to available music categories
        self.emotion_mapping = {
            'Angry': 'Angry',
            'Disgusted': 'Angry',  # Map to Angry
            'Fearful': 'Fear',
            'Happy': 'Happy',
            'Neutral': 'Neutral',
            'Sad': 'Sad',
            'Surprised': 'Happy'  # Map to Happy
        }
        
    def _map_emotion(self, emotion: str) -> str:
        """Map detected emotion to available music category."""
        return self.emotion_mapping.get(emotion, 'Neutral')  # Default to Neutral if unknown
        
    def _get_spotify_link(self, track_name: str, artist: str) -> str:
        """Get Spotify link for a track."""
        try:
            query = f"track:{track_name} artist:{artist}"
            results = self.spotify.search(q=query, type='track', limit=1)
            if results['tracks']['items']:
                return results['tracks']['items'][0]['external_urls']['spotify']
        except Exception as e:
            print(f"Error getting Spotify link: {e}")
        return None
        
    def _get_youtube_link(self, track_name: str, artist: str) -> str:
        """Get YouTube link for a track."""
        try:
            search_query = f"{track_name} {artist} audio"
            search = Search(search_query)
            if search.results:
                return search.results[0].watch_url
        except Exception as e:
            print(f"Error getting YouTube link: {e}")
        return None
        
    def _process_emotions(self, emotions: List[str]) -> List[str]:
        """Process emotions to get unique list in order of frequency."""
        # Map emotions to available categories
        mapped_emotions = [self._map_emotion(emotion) for emotion in emotions]
        emotion_counts = Counter(mapped_emotions)
        return [emotion for emotion, _ in emotion_counts.most_common()]
        
    def get_recommendations(self, emotions: List[str]) -> List[Dict]:
        """Get music recommendations based on detected emotions."""
        # Process emotions to get unique list
        unique_emotions = self._process_emotions(emotions)
        
        # Determine number of songs per emotion
        n_emotions = len(unique_emotions)
        if n_emotions == 1:
            counts = [10]
        elif n_emotions == 2:
            counts = [6, 4]
        elif n_emotions == 3:
            counts = [5, 3, 2]
        else:
            counts = [3] * n_emotions
            
        # Get songs for each emotion
        songs_df = self.music_data.get_songs_for_emotions(unique_emotions, counts)
        
        # Get streaming links and format recommendations
        recommendations = []
        for _, row in songs_df.iterrows():
            spotify_link = self._get_spotify_link(row['name'], row['artist'])
            youtube_link = self._get_youtube_link(row['name'], row['artist'])
            
            recommendations.append({
                'name': row['name'],
                'artist': row['artist'],
                'spotify_link': spotify_link,
                'youtube_link': youtube_link
            })
            
        return recommendations 