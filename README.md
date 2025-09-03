# 🎵 Music Playlist Manager

A Flask-based web application for managing music playlists.  
Integrates with **Spotify** for importing playlists and albums, downloads songs from **YouTube**, performs **audio analysis**, creates **audio fingerprints**, and allows **tempo shifting** with pitch correction.  

Supports **user authentication**, playlist management, song ratings, notes, and advanced audio processing using libraries like **Librosa** and **SciPy**.

---

## 🚀 Features

- **User Authentication:** Register, login, and logout with secure password hashing (**Bcrypt**)  
- **Playlist Management:** Create, edit, delete playlists; add/remove songs; import from Spotify playlists or albums  
- **Spotify Integration:** Authorize with Spotify, import playlists/albums, fetch song metadata (genres via Last.fm, popularity, duration)  
- **Song Downloading:** Download audio from YouTube based on Spotify track info (**yt-dlp**)  
- **Audio Analysis:** Tempo, spectral features, MFCC, onsets, RMS; generate spectrograms and reports  
- **Audio Fingerprinting:** Create fingerprints (peaks, MFCC, chroma, etc.) and visualize them  
- **Song Comparison:** Compare fingerprints for similarity; cache results and generate visualizations  
- **Tempo Shifting:** Shift tempo by semitones with pitch correction; generate audio and spectrograms  
- **Charts & Visualizations:** Playlist stats (popularity, duration) and interactive player with spectrogram  
- **Database:** Users, playlists, songs, fingerprints, comparisons (**SQLite**)

---

## ⚙️ Requirements

- **Python 3.12+** (tested with 3.12.3)  
- Install dependencies via pip:

```text
flask
flask-login
flask-bcrypt
flask-migrate
sqlalchemy
spotipy
yt-dlp
numpy
matplotlib
librosa
soundfile
scipy
fastdtw
statsmodels
uuid
requests
werkzeug
markupsafe
```
FFmpeg required for audio processing (brew install ffmpeg / apt install ffmpeg)

All packages must be pre-installed; runtime installation not supported


🛠️ Installation

Clone the repository:
git clone <your-repo-url>
cd music-playlist-manager
Set up virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:
pip install -r requirements.txt


🗄️ Database Setup

SQLite is used (users.db by default)

Run the app once to auto-create tables:
from app import db
db.create_all()

Optional (Flask-Migrate):
flask db init
flask db migrate
flask db upgrade

🔧 Configuration

Secret Key: Replace your_secret_key in app.secret_key with a secure random string:
import os
app.secret_key = os.urandom(24).hex()
API Keys (use environment variables!):
import os
app.config['SPOTIPY_CLIENT_ID'] = os.getenv('SPOTIPY_CLIENT_ID')
app.config['SPOTIPY_CLIENT_SECRET'] = os.getenv('SPOTIPY_CLIENT_SECRET')
app.config['LASTFM_API_KEY'] = os.getenv('LASTFM_API_KEY')
app.config['LASTFM_SHARED_SECRET'] = os.getenv('LASTFM_SHARED_SECRET')

Redirect URI: Defaults to http://localhost:5000/callback
Directories (auto-created):
downloads/ – MP3 files
analysis/ – analysis reports & spectrograms
tempo_shifted/ – tempo-shifted audio
fingerprints/ – fingerprint PKL files & visualizations


▶️ Running the App
source venv/bin/activate  # activate venv
python app.py

Runs on http://localhost:5000
 in debug mode

For production, use a WSGI server like Gunicorn:
gunicorn -w 4 app:app

### 📝 Usage

- **Register / Login:** Create an account  
- **Spotify Authorization:** Login with Spotify to enable imports  
- **Create Playlists:** Add manually or import from Spotify  
- **Add Songs:** From Spotify URL or full playlists/albums  
- **Download Songs:** Requires Spotify auth; downloads from YouTube  
- **Analyze Songs:** Generates tempo, spectral features, MFCC, onsets, RMS, spectrograms  
- **Fingerprint Songs:** For similarity comparison  
- **Compare Songs:** Visualize similarity between songs  
- **Tempo Shift:** Adjust tempo with pitch correction  
- **Charts & Player:** View playlist stats and play songs  
- **Logout:** From home page  

⚠️ Note: YouTube search results may vary. Ensure legal usage. Audio processing may be CPU/RAM intensive.

---

### 🗂️ Directory Structure
music-playlist-manager/
├── app.py
├── downloads/ # downloaded MP3s
├── analysis/ # analysis reports, spectrograms
├── tempo_shifted/ # shifted MP3s & spectrograms
├── fingerprints/ # PKL fingerprints & visualizations
├── users.db # SQLite DB
├── migrations/ # Flask-Migrate files
└── templates/ # HTML templates


### 🛠️ Troubleshooting

- **Spotify Errors:** Check API keys & redirect URI  
- **Download Failures:** Verify yt-dlp version and YouTube access  
- **Audio Errors:** Install FFmpeg; check file paths  
- **Database Issues:** Delete `users.db` and restart  
- **Logging:** Check DEBUG console logs

🤝 Contributing
Fork the repo, make changes, and submit a pull request. Focus on security (move keys to env vars) or features (more analysis metrics).
