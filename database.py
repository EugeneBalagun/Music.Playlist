from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(db.Model, UserMixin):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    spotify_token = db.Column(db.String(255), nullable=True)
    playlists = db.relationship('Playlist', backref='user', lazy=True)

class Playlist(db.Model):
    __tablename__ = 'playlist'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    songs = db.relationship('Song', backref='playlist', lazy=True)

class Song(db.Model):
    __tablename__ = 'song'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    url = db.Column(db.String(255), nullable=False)
    spotify_id = db.Column(db.String(100), nullable=True)
    playlist_id = db.Column(db.Integer, db.ForeignKey('playlist.id'), nullable=False)
    genres = db.Column(db.String(255), nullable=True)
    popularity = db.Column(db.Integer, nullable=True)
    duration_ms = db.Column(db.Integer, nullable=True)
    explicit = db.Column(db.Boolean, nullable=True)
    release_date = db.Column(db.String(50), nullable=True)
    rating = db.Column(db.Integer, nullable=True)
    notes = db.Column(db.Text, nullable=True)
    file_path = db.Column(db.String(255), nullable=True)
    tempo = db.Column(db.Float, nullable=True)
    duration = db.Column(db.Float, nullable=True)
    spectral_centroid = db.Column(db.Float, nullable=True)
    onset_count = db.Column(db.Integer, nullable=True)
    analysis_report_path = db.Column(db.String(255), nullable=True)
    spectrogram_path = db.Column(db.String(255), nullable=True)
    chromagram_path = db.Column(db.String(255), nullable=True)
    pitch_shifted_standard_path = db.Column(db.String(255), nullable=True)
    pitch_shifted_custom_path = db.Column(db.String(255), nullable=True)
    tempo_shifted_standard_path = db.Column(db.String(255), nullable=True)
    tempo_shifted_custom_path = db.Column(db.String(255), nullable=True)

    def __repr__(self):
        return f'<Song {self.name}>'