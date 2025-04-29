from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    spotify_token = db.Column(db.String(255))
    playlists = db.relationship('Playlist', back_populates='user')

class Playlist(db.Model):
    __tablename__ = 'playlists'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', back_populates='playlists')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    description = db.Column(db.String(500), nullable=True)
    songs = db.relationship('Song', backref='playlist', lazy=True)

class Song(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    url = db.Column(db.String(255), nullable=False)
    playlist_id = db.Column(db.Integer, db.ForeignKey('playlists.id'), nullable=False)
    spotify_id = db.Column(db.String(255))
    rating = db.Column(db.Integer, nullable=True)
    notes = db.Column(db.Text, nullable=True)
    genres = db.Column(db.String(255), nullable=True)
    popularity = db.Column(db.Integer, nullable=True)  # Популярность (0–100)
    duration_ms = db.Column(db.Integer, nullable=True)  # Длительность в миллисекундах
    explicit = db.Column(db.Boolean, nullable=True)  # Явный контент
    release_date = db.Column(db.String(50), nullable=True)  # Дата выпуска

    def __repr__(self):
        return f'<Song {self.name}>'