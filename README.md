Music Playlist Manager
This is a Flask-based web application for managing music playlists. It integrates with Spotify for importing playlists and albums, downloads songs from YouTube, performs audio analysis (e.g., tempo, spectrograms), creates audio fingerprints for similarity comparison, and allows tempo shifting with pitch correction. The app supports user authentication, playlist creation/editing/deletion, song ratings, notes, and advanced audio processing using libraries like Librosa and SciPy.
Features

User Authentication: Register, login, and logout with secure password hashing (Bcrypt).
Playlist Management: Create, edit, delete playlists; add/remove songs; import from Spotify playlists or albums.
Spotify Integration: Authorize with Spotify, import playlists/albums, fetch song metadata (genres via Last.fm, popularity, duration, etc.).
Song Downloading: Download audio from YouTube based on Spotify track info (using yt-dlp).
Audio Analysis: Analyze downloaded songs for tempo, spectral features, MFCC, onsets, RMS; generate spectrograms and reports.
Audio Fingerprinting: Create fingerprints for songs (peaks, MFCC, chroma, etc.); visualize with spectrograms, chromagrams, MFCC plots.
Song Comparison: Compare fingerprints between songs for similarity (peaks, MFCC, chroma, tempo, etc.); cache results; generate visualizations.
Tempo Shifting: Shift song tempo by semitones using standard or custom methods, with pitch correction; generate shifted audio and spectrograms.
Charts and Visualizations: View playlist charts (popularity, duration, etc.); interactive song player with spectrogram.
Database: Stores users, playlists, songs, fingerprints, and comparisons using SQLite.

Requirements

Python 3.12+ (tested with 3.12.3)
Dependencies (listed in the code; install via pip):
Flask
Flask-Login
Flask-Bcrypt
Flask-Migrate
SQLAlchemy
Spotipy
yt-dlp
NumPy
Matplotlib
Librosa
SoundFile
SciPy
Pickle
Hashlib
FastDTW
Statsmodels (for Shapiro test, though minimally used)
Other implicit: tqdm, ecdsa, pandas, sympy, mpmath, PuLP, astropy, qutip, control, biopython, pubchempy, dendropy, rdkit, pyscf, pygame, chess, mido, midiutil, networkx, torch, snappy (though not all are directly used; code assumes a REPL-like environment with these pre-installed).



Note: The app does not support installing additional packages at runtime. All must be pre-installed.
Installation

Clone the Repository (or download the code):
git clone <your-repo-url>
cd music-playlist-manager


Set Up Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:Create a requirements.txt file with the following (based on imported modules):
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

Then install:
pip install -r requirements.txt

Additional Notes:

Some libraries (e.g., Librosa, SciPy) may require system dependencies like FFmpeg for audio processing. Install FFmpeg via your package manager (e.g., brew install ffmpeg on macOS, apt install ffmpeg on Ubuntu).
For full audio support, ensure FFmpeg is in your PATH.


Database Setup:

The app uses SQLite (users.db by default).
Run the app once to auto-create tables (via db.create_all()).
If using migrations:flask db init
flask db migrate
flask db upgrade





Configuration

Secret Key: Replace 'your_secret_key' in app.secret_key with a secure random string (e.g., generate with os.urandom(24).hex()).
API Keys (hardcoded in the code; DO NOT USE IN PRODUCTION – move to environment variables or a config file):
Spotify: Set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET (get from Spotify Developer Dashboard).
Last.fm: Set LASTFM_API_KEY and LASTFM_SHARED_SECRET (get from Last.fm API).
Redirect URI: Defaults to http://localhost:5000/callback – update if running on a different host/port.


Directories: The app creates these automatically:
downloads/: For downloaded MP3 files.
analysis/: For analysis reports and spectrograms.
tempo_shifted/: For tempo-shifted audio.
fingerprints/: For fingerprint PKL files and visualizations.


Database URI: Defaults to sqlite:///users.db. Change in app.config['SQLALCHEMY_DATABASE_URI'] if needed.
Logging: Set to DEBUG level; adjust in logging.basicConfig().

Security Note: Hardcoded API keys are insecure. Use environment variables:
import os
app.config['SPOTIPY_CLIENT_ID'] = os.getenv('SPOTIPY_CLIENT_ID')
# ... similarly for others

Running the App

Activate the virtual environment (if not already).

Run the Flask app:
python app.py


The app runs on http://localhost:5000 in debug mode.
For production, use a WSGI server like Gunicorn: gunicorn -w 4 app:app.


Access the app in your browser: http://localhost:5000.

Register a new user or login.



Usage

Register/Login: Create an account and log in.
Spotify Authorization: Click "Login with Spotify" to enable imports/downloads.
Create Playlists: Add new playlists manually or import from Spotify URLs.
Add Songs: Add via Spotify URLs or import entire playlists/albums.
Download Songs: From song details, download audio (requires Spotify auth).
Analyze Songs: After downloading, analyze for features and visualizations.
Fingerprint Songs: Create fingerprints for comparison.
Compare Songs: Select two fingerprinted songs to compare similarity.
Tempo Shift: Adjust tempo (semitones) using standard/custom methods.
View Charts/Player: See playlist stats or play songs with spectrograms.
Logout: From the home page.

Notes:

Downloads use YouTube search; results may vary – ensure legal usage.
Audio processing can be resource-intensive (CPU/RAM); large files may take time.
Fingerprints and comparisons are cached in the DB for efficiency.
Visualizations are saved as PNGs; reports as TXT.

Directory Structure
music-playlist-manager/
├── app.py              # Main application code
├── downloads/          # Downloaded MP3s (auto-created)
├── analysis/           # Analysis reports, spectrograms, JSON (auto-created)
├── tempo_shifted/      # Shifted MP3s and spectrograms (auto-created)
├── fingerprints/       # PKL fingerprints and PNG visualizations (auto-created)
├── users.db            # SQLite database (auto-created)
├── migrations/         # Flask-Migrate files (if initialized)
└── templates/          # HTML templates (e.g., index.html, playlists.html) – add based on code references

Troubleshooting

Spotify Errors: Ensure valid API keys and redirect URI. Refresh token if expired.
Download Failures: Check yt-dlp version; ensure YouTube access.
Audio Errors: Install FFmpeg; verify file paths.
Database Issues: Delete users.db and restart to reset.
Logging: Check console for DEBUG logs.

Contributing
Fork the repo, make changes, and submit a pull request. Focus on security (e.g., env vars for keys) or features (e.g., more analysis metrics).
License
MIT License (or specify your own). This project is for educational purposes; respect API terms and copyrights for music.
