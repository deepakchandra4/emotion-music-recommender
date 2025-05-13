import pandas as pd
import os
from typing import Dict, List

class MusicData:
    def __init__(self, csv_path='data/muse_v3.csv'):
        """Initialize and load music data."""
        self.df = pd.read_csv(csv_path)
        self._preprocess_data()
        self._split_by_emotion()
        
    def _preprocess_data(self):
        """Preprocess the music dataset."""
        # Rename columns for clarity
        self.df['link'] = self.df['lastfm_url']
        self.df['name'] = self.df['track']
        self.df['emotional'] = self.df['number_of_emotion_tags']
        self.df['pleasant'] = self.df['valence_tags']
        
        # Select relevant columns
        self.df = self.df[['name', 'emotional', 'pleasant', 'link', 'artist']]
        
        # Sort and reset index
        self.df = self.df.sort_values(by=["emotional", "pleasant"]).reset_index(drop=True)
        
    def _split_by_emotion(self):
        """Split dataset into emotion-based categories."""
        total = len(self.df)
        chunk_size = total // 5
        
        self.emotion_dfs = {
            'Sad': self.df[:chunk_size],
            'Fear': self.df[chunk_size:2*chunk_size],
            'Angry': self.df[2*chunk_size:3*chunk_size],
            'Neutral': self.df[3*chunk_size:4*chunk_size],
            'Happy': self.df[4*chunk_size:]
        }
        
    def get_songs_by_emotion(self, emotion: str, n: int = 10) -> pd.DataFrame:
        """Get n songs for a given emotion."""
        if emotion not in self.emotion_dfs:
            raise ValueError(f"Invalid emotion: {emotion}. Must be one of {list(self.emotion_dfs.keys())}")
        return self.emotion_dfs[emotion].sample(n=min(n, len(self.emotion_dfs[emotion])))
        
    def get_songs_for_emotions(self, emotions: List[str], counts: List[int]) -> pd.DataFrame:
        """Get songs for multiple emotions with specified counts."""
        if len(emotions) != len(counts):
            raise ValueError("Number of emotions must match number of counts")
            
        result = pd.DataFrame()
        for emotion, count in zip(emotions, counts):
            songs = self.get_songs_by_emotion(emotion, count)
            result = pd.concat([result, songs], ignore_index=True)
        return result 