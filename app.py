import json
import requests
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from urllib.parse import urlparse
import logging
from database import db, User, Playlist, Song, FingerprintComparison
import os
import yt_dlp
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from flask import send_file
from markupsafe import Markup
from os.path import basename
import soundfile as sf
import librosa
import librosa.display
from scipy.signal import find_peaks
import pickle

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SPOTIPY_CLIENT_ID'] = '79357f9118c243568eb3847e9a6baaa9'
app.config['SPOTIPY_CLIENT_SECRET'] = '7a301db0ee9041a7ac8685b5f6613a70'
app.config['SPOTIPY_REDIRECT_URI'] = 'http://localhost:5000/callback'
app.config['LASTFM_API_KEY'] = 'a618107a2b9f5a21c6b6de4818e70c6b'
app.config['LASTFM_SHARED_SECRET'] = '645060ce2167708025f599fb081dedbc'
bcrypt = Bcrypt(app)
db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
migrate = Migrate(app, db)

# Jinja filters
app.jinja_env.filters['basename'] = basename


def format_time(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


app.jinja_env.filters['format_time'] = format_time

with app.app_context():
    db.create_all()

# Spotify OAuth initialization
sp_oauth = SpotifyOAuth(
    client_id=app.config['SPOTIPY_CLIENT_ID'],
    client_secret=app.config['SPOTIPY_CLIENT_SECRET'],
    redirect_uri=app.config['SPOTIPY_REDIRECT_URI'],
    scope='user-library-read user-read-private playlist-read-private playlist-read-collaborative'
)

# Create directories
DOWNLOAD_DIR = os.path.join(os.getcwd(), 'downloads')
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
ANALYSIS_DIR = os.path.join(os.getcwd(), 'analysis')
os.makedirs(ANALYSIS_DIR, exist_ok=True)
TEMPO_SHIFT_DIR = os.path.join(os.getcwd(), 'tempo_shifted')
os.makedirs(TEMPO_SHIFT_DIR, exist_ok=True)
FINGERPRINT_DIR = os.path.join(os.getcwd(), 'fingerprints')
os.makedirs(FINGERPRINT_DIR, exist_ok=True)

# Parameters for methods
K_CUSTOM = 22885686008
N_CUSTOM = 39123338641
K_STANDARD = 7
N_STANDARD = 12


# Get genres from Last.fm
def get_genre_from_lastfm(track_name, artist_name):
    url = f"http://ws.audioscrobbler.com/2.0/?method=track.getInfo&api_key={app.config['LASTFM_API_KEY']}&artist={artist_name}&track={track_name}&format=json"
    try:
        response = requests.get(url)
        data = response.json()
        if 'track' in data and 'toptags' in data['track']:
            tags = [tag['name'] for tag in data['track']['toptags']['tag']]
            return tags[:3]
        return []
    except Exception as e:
        logging.error(f"Last.fm API error: {e}")
        return []


def create_fingerprint(file_path, song):
    # Check if fingerprint exists and is up-to-date
    if song.fingerprint_path and os.path.exists(song.fingerprint_path):
        try:
            audio_mtime = os.path.getmtime(file_path)
            fingerprint_mtime = os.path.getmtime(song.fingerprint_path)
            if fingerprint_mtime >= audio_mtime:
                with open(song.fingerprint_path, 'rb') as f:
                    fingerprint_data = pickle.load(f)
                logging.debug(f"Loaded cached fingerprint for song {song.id}")
                return {
                    'fingerprint_path': song.fingerprint_path,
                    'fingerprint_count': len(fingerprint_data['peaks']),
                    'features': {
                        'tempo': fingerprint_data['tempo'],
                        'spectral_centroid': fingerprint_data['spectral_centroid'],
                        'spectral_rolloff': fingerprint_data['spectral_rolloff'],
                        'spectral_bandwidth': fingerprint_data['spectral_bandwidth'],
                        'rms': fingerprint_data['rms'],
                        'rms_var': fingerprint_data['rms_var'],
                        'onset_count': fingerprint_data['onset_count'],
                        'rhythmic_complexity': fingerprint_data['rhythmic_complexity'],
                        'estimated_instruments': fingerprint_data['estimated_instruments'],
                        'segment_count': fingerprint_data['segment_count'],
                        'zcr': fingerprint_data['zcr']
                    }
                }
        except Exception as e:
            logging.warning(f"Failed to load cached fingerprint for song {song.id}: {e}")

    # Generate new fingerprint
    try:
        y, sr = librosa.load(file_path, sr=44100)
        duration = librosa.get_duration(y=y, sr=sr)

        # Compute STFT for spectral peaks
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        fingerprints = []
        for t in range(S_db.shape[1]):
            peaks, properties = find_peaks(S_db[:, t], height=-20)
            for peak in peaks:
                freq = librosa.fft_frequencies(sr=sr, n_fft=2048)[peak]
                amplitude = S_db[peak, t]
                time = t * 512 / sr
                fingerprints.append((time, freq, amplitude))

        # Additional features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        rms = librosa.feature.rms(y=y).mean()
        rms_var = librosa.feature.rms(y=y).var()
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        onset_count = len(onsets)
        onset_intervals = np.diff(onsets)
        rhythmic_complexity = np.std(onset_intervals) if len(onset_intervals) > 0 else 0
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
        zcr = librosa.feature.zero_crossing_rate(y=y).mean()

        # Estimate number of instruments
        unique_peaks = len(np.unique([f for _, f, _ in fingerprints]))
        mfcc_var = np.var(mfcc)
        # Новая формула: логарифмическое масштабирование и более чувствительные коэффициенты
        estimated_instruments = min(int(np.log1p(unique_peaks) / 2 + mfcc_var * 5), 15)
        logging.debug(
            f"Estimated instruments: unique_peaks={unique_peaks}, mfcc_var={mfcc_var}, result={estimated_instruments}")

        # Song structure
        segments = librosa.segment.recurrence_matrix(librosa.feature.chroma_cqt(y=y, sr=sr), mode='affinity')
        segment_count = len(np.unique(np.argmax(segments, axis=1)))

        # Save fingerprint and features
        fingerprint_data = {
            'peaks': fingerprints,
            'tempo': tempo,
            'spectral_centroid': float(spectral_centroid),
            'spectral_rolloff': float(spectral_rolloff),
            'spectral_bandwidth': float(spectral_bandwidth),
            'mfcc': mfcc.tolist(),
            'rms': float(rms),
            'rms_var': float(rms_var),
            'onset_count': onset_count,
            'rhythmic_complexity': float(rhythmic_complexity),
            'chroma': chroma.tolist(),
            'zcr': float(zcr),
            'estimated_instruments': estimated_instruments,
            'segment_count': segment_count,
            'duration': duration
        }

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fingerprint_path = os.path.join(FINGERPRINT_DIR, f'fingerprint_{song.id}_{timestamp}.pkl')
        with open(fingerprint_path, 'wb') as f:
            pickle.dump(fingerprint_data, f)

        # Update song in database
        song.fingerprint_path = fingerprint_path
        song.tempo = tempo
        song.spectral_centroid = spectral_centroid
        song.duration = duration
        song.onset_count = onset_count
        db.session.commit()

        return {
            'fingerprint_path': fingerprint_path,
            'fingerprint_count': len(fingerprints),
            'features': {
                'tempo': tempo,
                'spectral_centroid': spectral_centroid,
                'spectral_rolloff': spectral_rolloff,
                'spectral_bandwidth': spectral_bandwidth,
                'rms': rms,
                'rms_var': rms_var,
                'onset_count': onset_count,
                'rhythmic_complexity': rhythmic_complexity,
                'estimated_instruments': estimated_instruments,
                'segment_count': segment_count,
                'zcr': zcr
            }
        }
    except Exception as e:
        logging.error(f"Fingerprint creation error: {e}")
        raise

def compare_fingerprints(fingerprint_path1, fingerprint_path2, song_id1, song_id2, force_recompute=False):
    # Check for cached comparison
    if not force_recompute:
        comparison = FingerprintComparison.query.filter(
            ((FingerprintComparison.song_id1 == song_id1) & (FingerprintComparison.song_id2 == song_id2)) |
            ((FingerprintComparison.song_id1 == song_id2) & (FingerprintComparison.song_id2 == song_id1))
        ).first()
        if comparison:
            try:
                # Check if fingerprints or audio files have been updated
                song1_mtime = os.path.getmtime(fingerprint_path1) if os.path.exists(fingerprint_path1) else 0
                song2_mtime = os.path.getmtime(fingerprint_path2) if os.path.exists(fingerprint_path2) else 0
                comparison_mtime = comparison.updated_at.timestamp()
                if comparison_mtime >= song1_mtime and comparison_mtime >= song2_mtime:
                    logging.debug(f"Loaded cached comparison for songs {song_id1} and {song_id2}")
                    return {
                        'overall_similarity': comparison.overall_similarity,
                        'details': json.loads(comparison.details),
                        'cached': True
                    }
            except Exception as e:
                logging.warning(f"Failed to validate cached comparison: {e}")

    # Perform new comparison
    try:
        with open(fingerprint_path1, 'rb') as f:
            data1 = pickle.load(f)
        with open(fingerprint_path2, 'rb') as f:
            data2 = pickle.load(f)

        # Compare spectral peaks
        fingerprint1 = data1['peaks']
        fingerprint2 = data2['peaks']
        matches = 0
        for t1, f1, a1 in fingerprint1:
            for t2, f2, a2 in fingerprint2:
                if abs(t1 - t2) < 0.1 and abs(f1 - f2) < 50:
                    amplitude_diff = abs(a1 - a2) / max(abs(a1), abs(a2), 1e-6)
                    if amplitude_diff < 0.5:
                        matches += 1
                        break
        peak_similarity = (matches / min(len(fingerprint1), len(fingerprint2)) * 100) if min(len(fingerprint1), len(fingerprint2)) > 0 else 0

        # Compare additional features
        tempo_diff = abs(data1['tempo'] - data2['tempo']) / max(data1['tempo'], data2['tempo'], 1e-6)
        tempo_similarity = max(0, 1 - tempo_diff) * 100

        centroid_diff = abs(data1['spectral_centroid'] - data2['spectral_centroid']) / max(data1['spectral_centroid'], data2['spectral_centroid'], 1e-6)
        centroid_similarity = max(0, 1 - centroid_diff) * 100

        rolloff_diff = abs(data1['spectral_rolloff'] - data2['spectral_rolloff']) / max(data1['spectral_rolloff'], data2['spectral_rolloff'], 1e-6)
        rolloff_similarity = max(0, 1 - rolloff_diff) * 100

        bandwidth_diff = abs(data1['spectral_bandwidth'] - data2['spectral_bandwidth']) / max(data1['spectral_bandwidth'], data2['spectral_bandwidth'], 1e-6)
        bandwidth_similarity = max(0, 1 - bandwidth_diff) * 100

        mfcc1 = np.array(data1['mfcc'])
        mfcc2 = np.array(data2['mfcc'])
        mfcc_correlation = np.corrcoef(mfcc1, mfcc2)[0, 1]
        mfcc_similarity = max(0, mfcc_correlation) * 100

        rms_diff = abs(data1['rms'] - data2['rms']) / max(data1['rms'], data2['rms'], 1e-6)
        rms_similarity = max(0, 1 - rms_diff) * 100

        rms_var_diff = abs(data1['rms_var'] - data2['rms_var']) / max(data1['rms_var'], data2['rms_var'], 1e-6)
        rms_var_similarity = max(0, 1 - rms_var_diff) * 100

        onset_diff = abs(data1['onset_count'] - data2['onset_count']) / max(data1['onset_count'], data2['onset_count'], 1e-6)
        onset_similarity = max(0, 1 - onset_diff) * 100

        rhythmic_diff = abs(data1['rhythmic_complexity'] - data2['rhythmic_complexity']) / max(data1['rhythmic_complexity'], data2['rhythmic_complexity'], 1e-6)
        rhythmic_similarity = max(0, 1 - rhythmic_diff) * 100

        chroma1 = np.array(data1['chroma'])
        chroma2 = np.array(data2['chroma'])
        chroma_correlation = np.corrcoef(chroma1, chroma2)[0, 1]
        chroma_similarity = max(0, chroma_correlation) * 100

        zcr_diff = abs(data1['zcr'] - data2['zcr']) / max(data1['zcr'], data2['zcr'], 1e-6)
        zcr_similarity = max(0, 1 - zcr_diff) * 100

        instruments_diff = abs(data1['estimated_instruments'] - data2['estimated_instruments']) / max(data1['estimated_instruments'], data2['estimated_instruments'], 1e-6)
        instruments_similarity = max(0, 1 - instruments_diff) * 100

        segment_diff = abs(data1['segment_count'] - data2['segment_count']) / max(data1['segment_count'], data2['segment_count'], 1e-6)
        segment_similarity = max(0, 1 - segment_diff) * 100

        # Weighted average similarity
        weights = {
            'peak': 0.3,
            'tempo': 0.1,
            'centroid': 0.1,
            'rolloff': 0.05,
            'bandwidth': 0.05,
            'mfcc': 0.15,
            'rms': 0.05,
            'rms_var': 0.05,
            'onset': 0.05,
            'rhythmic': 0.05,
            'chroma': 0.05,
            'zcr': 0.03,
            'instruments': 0.02,
            'segment': 0.05
        }
        overall_similarity = (
            weights['peak'] * peak_similarity +
            weights['tempo'] * tempo_similarity +
            weights['centroid'] * centroid_similarity +
            weights['rolloff'] * rolloff_similarity +
            weights['bandwidth'] * bandwidth_similarity +
            weights['mfcc'] * mfcc_similarity +
            weights['rms'] * rms_similarity +
            weights['rms_var'] * rms_var_similarity +
            weights['onset'] * onset_similarity +
            weights['rhythmic'] * rhythmic_similarity +
            weights['chroma'] * chroma_similarity +
            weights['zcr'] * zcr_similarity +
            weights['instruments'] * instruments_similarity +
            weights['segment'] * segment_similarity
        )

        result = {
            'overall_similarity': overall_similarity,
            'details': {
                'Spectral Peaks': peak_similarity,
                'Tempo': tempo_similarity,
                'Spectral Centroid': centroid_similarity,
                'Spectral Rolloff': rolloff_similarity,
                'Spectral Bandwidth': bandwidth_similarity,
                'MFCC Correlation': mfcc_similarity,
                'RMS': rms_similarity,
                'RMS Variance': rms_var_similarity,
                'Onset Count': onset_similarity,
                'Rhythmic Complexity': rhythmic_similarity,
                'Chroma Correlation': chroma_similarity,
                'Zero Crossing Rate': zcr_similarity,
                'Estimated Instruments': instruments_similarity,
                'Segment Count': segment_similarity
            },
            'cached': False
        }

        # Save to cache
        try:
            comparison = FingerprintComparison.query.filter(
                ((FingerprintComparison.song_id1 == song_id1) & (FingerprintComparison.song_id2 == song_id2)) |
                ((FingerprintComparison.song_id1 == song_id2) & (FingerprintComparison.song_id2 == song_id1))
            ).first()
            if comparison:
                comparison.overall_similarity = overall_similarity
                comparison.details = json.dumps(result['details'])
                comparison.updated_at = db.func.current_timestamp()
            else:
                comparison = FingerprintComparison(
                    song_id1=song_id1,
                    song_id2=song_id2,
                    overall_similarity=overall_similarity,
                    details=json.dumps(result['details'])
                )
                db.session.add(comparison)
            db.session.commit()
            logging.debug(f"Cached comparison for songs {song_id1} and {song_id2}")
        except Exception as e:
            logging.error(f"Failed to cache comparison: {e}")

        return result
    except Exception as e:
        logging.error(f"Fingerprint comparison error: {e}")
        raise

@app.route('/login_spotify')
@login_required
def login_spotify():
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)


@app.route('/logout_spotify')
@login_required
def logout_spotify():
    current_user.spotify_token = None
    db.session.commit()
    flash('Successfully logged out from Spotify.')
    return redirect(url_for('home'))


@app.route('/callback')
def callback():
    code = request.args.get('code')
    logging.debug(f'Received code: {code}')
    try:
        token_info = sp_oauth.get_access_token(code)
        sp = spotipy.Spotify(auth=token_info['access_token'])
        logging.debug(f'Token received: {token_info}')
        current_user.spotify_token = token_info['refresh_token']
        db.session.commit()
        flash("Successfully authorized with Spotify!")
        if 'pending_playlist_url' in session:
            playlist_url = session.pop('pending_playlist_url')
            return redirect(url_for('import_spotify_playlist', playlist_url=playlist_url))
    except Exception as e:
        logging.error(f'Error in callback: {e}')
        flash('Error during Spotify authorization.')
    return redirect(url_for('home'))


def get_spotify_client():
    if not current_user.is_authenticated:
        logging.debug("User not authenticated")
        return None
    token_info = sp_oauth.get_cached_token()
    if token_info and not sp_oauth.is_token_expired(token_info):
        logging.debug("Using cached token")
        return spotipy.Spotify(auth=token_info['access_token'])
    if current_user.spotify_token:
        logging.debug(f"Refresh token from DB: {current_user.spotify_token}")
        try:
            token_info = sp_oauth.refresh_access_token(current_user.spotify_token)
            current_user.spotify_token = token_info['refresh_token']
            db.session.commit()
            logging.debug("Token successfully refreshed")
            return spotipy.Spotify(auth=token_info['access_token'])
        except Exception as e:
            logging.error(f"Token refresh error: {e}")
            current_user.spotify_token = None
            db.session.commit()
            flash("Your Spotify token is invalid. Please authorize again.")
            return None
    logging.debug("No token, Spotify login required")
    flash("Please authorize with Spotify to continue.")
    return None


@login_manager.user_loader
def user_loader(user_id):
    return User.query.get(int(user_id))


@app.route('/add_song_spotify/<int:playlist_id>', methods=['POST'])
@login_required
def add_song_spotify(playlist_id):
    song_url = request.form.get('song_url')
    if not song_url:
        flash("Please enter a Spotify song URL.")
        return redirect(url_for('playlists_page'))
    sp = get_spotify_client()
    if not sp:
        return redirect(url_for('login_spotify'))
    try:
        parsed_url = urlparse(song_url)
        track_id = parsed_url.path.split('/')[-1].split('?')[0]
        track = sp.track(track_id)
        artist_name = track['artists'][0]['name']
        genres = get_genre_from_lastfm(track['name'], artist_name)
        new_song = Song(
            name=track['name'],
            url=track['external_urls']['spotify'],
            spotify_id=track_id,
            playlist_id=playlist_id,
            genres=','.join(genres) if genres else None,
            popularity=track.get('popularity'),
            duration_ms=track.get('duration_ms'),
            explicit=track.get('explicit'),
            release_date=track['album'].get('release_date')
        )
        db.session.add(new_song)
        db.session.commit()
        flash(f'Song "{track["name"]}" successfully added!')
    except Exception as e:
        logging.error(f"Error: {e}")
        flash("Failed to add song.")
    return redirect(url_for('playlists_page'))


@app.route('/playlists')
@login_required
def playlists_page():
    playlists = Playlist.query.filter_by(user_id=current_user.id).all()
    return render_template('playlists.html', playlists=playlists)


@app.route('/add_note/<int:song_id>', methods=['POST'])
@login_required
def add_note(song_id):
    note = request.form.get('note')
    song = Song.query.get_or_404(song_id)
    if note:
        song.notes = note
        db.session.commit()
        flash(f'Note for song "{song.name}" successfully updated!')
    return redirect(url_for('playlists_page'))


def get_song_name_from_url(url):
    sp = get_spotify_client()
    if not sp:
        logging.error("Spotify client unavailable!")
        return None
    try:
        track_id = url.split("/")[-1].split("?")[0]
        track = sp.track(track_id)
        return track["name"]
    except Exception as e:
        logging.error(f"Error fetching Spotify data: {e}")
        return None


@app.route('/add_playlist', methods=['GET', 'POST'])
@login_required
def add_playlist():
    if request.method == 'POST':
        playlist_name = request.form['playlist_name']
        description = request.form.get('description')
        if playlist_name:
            new_playlist = Playlist(name=playlist_name, user_id=current_user.id, description=description)
            db.session.add(new_playlist)
            db.session.commit()
            flash('Playlist successfully added!')
        else:
            flash('Please enter a playlist name.')
        return redirect(url_for('playlists_page'))
    return render_template('add_playlist.html')


@app.route('/')
@login_required
def home():
    playlists = Playlist.query.filter_by(user_id=current_user.id).all()
    spotify_logged_in = bool(current_user.spotify_token)
    return render_template(
        'index.html',
        playlists=playlists,
        spotify_logged_in=spotify_logged_in
    )


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('You have successfully registered!')
        login_user(new_user)
        return redirect(url_for('home'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('home'))
        else:
            flash('Invalid username or password!')
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/edit_playlist/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_playlist(id):
    playlist = Playlist.query.get(id)
    if playlist and playlist.user_id == current_user.id:
        if request.method == 'POST':
            new_name = request.form['playlist_name']
            description = request.form.get('description')
            if new_name:
                playlist.name = new_name
                playlist.description = description
                db.session.commit()
                flash(f"Playlist '{new_name}' successfully updated!")
            else:
                flash("Please enter a playlist name.")
            return redirect(url_for('playlists_page'))
    else:
        flash('You cannot edit this playlist.')
        return redirect(url_for('playlists_page'))
    return render_template('edit_playlist.html', playlist=playlist)


@app.route('/delete_song/<int:song_id>', methods=['POST'])
@login_required
def delete_song(song_id):
    song = Song.query.get(song_id)
    if song:
        if song.playlist.user_id == current_user.id:
            db.session.delete(song)
            db.session.commit()
            flash('Song successfully deleted from playlist!')
        else:
            flash('You cannot delete this song.')
    else:
        flash('Song not found.')
    if song:
        return redirect(url_for('playlists_page', playlist_id=song.playlist.id))
    else:
        return redirect(url_for('playlists_page', playlist_id=request.args.get('playlist_id')))


@app.route('/delete_playlist/<int:playlist_id>', methods=['POST'])
@login_required
def delete_playlist(playlist_id):
    playlist = Playlist.query.get(playlist_id)
    if playlist and playlist.user_id == current_user.id:
        Song.query.filter_by(playlist_id=playlist_id).delete()
        db.session.delete(playlist)
        db.session.commit()
        flash('Playlist successfully deleted!')
    else:
        flash('You cannot delete this playlist.')
    return redirect(url_for('playlists_page'))


@app.route('/rate_song/<int:song_id>', methods=['POST'])
@login_required
def rate_song(song_id):
    song = Song.query.get(song_id)
    if song:
        rating = request.form['rating']
        try:
            song.rating = int(rating)
            db.session.commit()
            flash(f'Rating for song "{song.name}" successfully updated!')
        except ValueError:
            flash('Error: Rating must be a number from 1 to 10.')
    else:
        flash('Error: Song not found.')
    return redirect(url_for('playlists_page'))


@app.route('/import_spotify_playlist', methods=['POST'])
@login_required
def import_spotify_playlist():
    playlist_url = request.form.get('playlist_url')
    logging.debug(f"Received URL: {playlist_url}")
    if not playlist_url:
        flash("Please enter a Spotify playlist or album URL.")
        return redirect(url_for('playlists_page'))
    sp = get_spotify_client()
    if not sp:
        session['pending_playlist_url'] = playlist_url
        return redirect(url_for('login_spotify'))
    try:
        parsed_url = urlparse(playlist_url)
        spotify_id = parsed_url.path.split('/')[-1].split('?')[0]
        logging.debug(f"Extracted Spotify ID: {spotify_id}")
        if '/playlist/' in parsed_url.path:
            logging.debug("Processing playlist")
            data = sp.playlist(spotify_id)
            tracks = data['tracks']['items']
            source_type = "playlist"
            description = data.get('description', None)
        elif '/album/' in parsed_url.path:
            logging.debug("Processing album")
            data = sp.album(spotify_id)
            tracks = data['tracks']['items']
            source_type = "album"
            description = None
        else:
            flash("Invalid URL. Use a Spotify playlist or album URL.")
            return redirect(url_for('playlists_page'))
        logging.debug(f"Retrieved {source_type}: {data['name']}")
        new_playlist = Playlist(
            name=data['name'],
            user_id=current_user.id,
            description=description
        )
        db.session.add(new_playlist)
        db.session.commit()
        logging.debug(f"Created playlist: {new_playlist.name} (ID: {new_playlist.id})")
        added_count = 0
        for item in tracks:
            track = item['track'] if source_type == "playlist" else item
            if not track or 'id' not in track:
                logging.debug("Track skipped: no ID")
                continue
            artist_name = track['artists'][0]['name']
            genres = get_genre_from_lastfm(track['name'], artist_name)
            song = Song(
                name=track['name'],
                url=track['external_urls']['spotify'],
                spotify_id=track['id'],
                playlist_id=new_playlist.id,
                genres=','.join(genres) if genres else None,
                popularity=track.get('popularity'),
                duration_ms=track.get('duration_ms'),
                explicit=track.get('explicit'),
                release_date=track['album'].get('release_date')
            )
            db.session.add(song)
            added_count += 1
            logging.debug(f"Added track: {track['name']}")
        db.session.commit()
        flash(f'Playlist "{data["name"]}" successfully imported with {added_count} songs!')
    except spotipy.SpotifyException as e:
        logging.error(f"Spotify API error: {e}")
        flash(f"Spotify error: {str(e)}")
    except Exception as e:
        logging.error(f"Import error: {e}")
        flash("Failed to import playlist.")
    return redirect(url_for('playlists_page'))


@app.route('/api/songs/<int:playlist_id>', methods=['GET'])
@login_required
def get_songs_data(playlist_id):
    playlist = Playlist.query.get_or_404(playlist_id)
    if playlist.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    songs = Song.query.filter_by(playlist_id=playlist_id).all()
    songs_data = [{
        'name': song.name,
        'popularity': song.popularity,
        'duration_ms': song.duration_ms,
        'explicit': song.explicit,
        'release_date': song.release_date,
        'genres': song.genres,
        'rating': song.rating
    } for song in songs]
    return jsonify(songs_data)


@app.route('/playlist/<int:playlist_id>/charts')
@login_required
def playlist_charts(playlist_id):
    playlist = Playlist.query.get_or_404(playlist_id)
    if playlist.user_id != current_user.id:
        flash('You cannot view charts for this playlist.')
        return redirect(url_for('playlists_page'))
    return render_template('charts.html', playlist=playlist)


@app.route('/download_song/<int:song_id>', methods=['POST'])
@login_required
def download_song(song_id):
    song = Song.query.get_or_404(song_id)
    if song.playlist.user_id != current_user.id:
        flash('You cannot download this song.')
        return redirect(url_for('playlists_page'))
    sp = get_spotify_client()
    if not sp:
        return redirect(url_for('login_spotify'))
    try:
        track = sp.track(song.spotify_id)
        track_name = track['name']
        artist_name = track['artists'][0]['name']
        query = f"{track_name} {artist_name}"
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(DOWNLOAD_DIR, f"{track_name} - {artist_name}.%(ext)s"),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"ytsearch:{query}"])
        file_path = os.path.join(DOWNLOAD_DIR, f"{track_name} - {artist_name}.mp3")
        if os.path.exists(file_path):
            song.file_path = file_path
            db.session.commit()
            flash(f'Song "{track_name}" successfully downloaded and saved!')
        else:
            flash(f'Failed to download song "{track_name}".')
            logging.error(f"File not found after download: {file_path}")
    except Exception as e:
        logging.error(f"Download error: {e}")
        flash(f'Failed to download song: {str(e)}')
    return redirect(url_for('playlists_page'))


def analyze_song(file_path, song):
    try:
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = mfcc.mean(axis=1).tolist()
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        onset_count = len(onsets)
        onset_times = onsets.tolist()
        rms = librosa.feature.rms(y=y).mean()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Generate spectrogram
        spectrogram_path = os.path.join(ANALYSIS_DIR, f'spectrogram_{song.id}_{timestamp}.png')
        window_size = 1024
        step_size = 512
        chunk_duration_sec = 4
        spectrogram, time, freq = process_full_audio(y, sr, window_size, step_size, chunk_duration_sec)
        width_pixels = max(5000, int(500 * duration / 10))
        fig_width = width_pixels / 100
        fig = plt.figure(figsize=(fig_width, 6), dpi=100)
        ax = fig.add_subplot(111)
        im = ax.imshow(
            20 * np.log10(spectrogram + 1e-6),
            aspect='auto',
            origin='lower',
            extent=[time[0], time[-1], freq[0], freq[-1]],
            cmap='magma'
        )
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_title(f'Spectrogram: {song.name}')
        fig.colorbar(im, ax=ax, label='Amplitude [dB]')
        fig.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.05)
        ax_pos = ax.get_position()
        data_area = {
            'x0': 0.0,
            'x1': 1.0,
            'width': 1.0,
            'pixel_x0': 0,
            'pixel_width': width_pixels
        }
        logging.debug(
            f"Generated data_area: x0={data_area['x0']}, x1={data_area['x1']}, width={data_area['width']}, pixel_x0={data_area['pixel_x0']}, pixel_width={data_area['pixel_width']}")
        data_area_path = os.path.join(ANALYSIS_DIR, f'data_area_{song.id}_{timestamp}.json')
        with open(data_area_path, 'w') as f:
            json.dump(data_area, f)
        plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Generate report
        report_path = os.path.join(ANALYSIS_DIR, f'report_{song.id}_{timestamp}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Audio Analysis: {os.path.basename(file_path)}\n")
            f.write(f"Duration: {duration:.2f} sec\n")
            f.write(f"Tempo: {tempo:.2f} BPM\n")
            f.write(f"Number of Beats: {len(beat_times)}\n")
            f.write(f"Beat Timestamps: {', '.join([f'{x:.2f}' for x in beat_times[:10]])}...\n")
            f.write(f"Spectral Centroid: {spectral_centroid:.2f} Hz\n")
            f.write(f"Spectral Rolloff: {spectral_rolloff:.2f} Hz\n")
            f.write(f"Spectral Bandwidth: {spectral_bandwidth:.2f} Hz\n")
            f.write(f"Mean MFCC: {', '.join([f'{x:.2f}' for x in mfcc_mean])}\n")
            f.write(f"Number of Onsets: {onset_count}\n")
            f.write(f"Onset Timestamps: {', '.join([f'{x:.2f}' for x in onset_times[:10]])}...\n")
            f.write(f"Mean RMS: {rms:.4f}\n")
            f.write(f"Spectrogram Path: {spectrogram_path}\n")
            f.write(f"Data Area Path: {data_area_path}\n")

        song.tempo = tempo
        song.duration = duration
        song.spectral_centroid = spectral_centroid
        song.onset_count = onset_count
        song.analysis_report_path = report_path
        song.spectrogram_path = spectrogram_path
        db.session.commit()

        summary = (
            f"Tempo: {tempo:.2f} BPM, "
            f"Duration: {duration:.2f} sec, "
            f"Spectral Centroid: {spectral_centroid:.2f} Hz, "
            f"Onsets: {onset_count}"
        )
        return {
            'summary': summary,
            'spectrogram_path': spectrogram_path,
            'report_path': report_path,
            'report_content': open(report_path, 'r', encoding='utf-8').read(),
            'mfcc': mfcc_mean,
            'spectral_centroid': spectral_centroid,
            'rms': rms,
            'data_area_path': data_area_path,
            'data_area': data_area
        }
    except Exception as e:
        logging.error(f"Analysis error: {e}")
        raise


def tempo_shift_song(file_path, song, semitones, method='standard'):
    try:
        y, sr = librosa.load(file_path, sr=44100)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if method == 'standard':
            K = K_STANDARD
            N = N_STANDARD
            output_prefix = f'standard_tempo_{song.id}_{semitones}_{timestamp}'
            rate = 2 ** (semitones / N)
            pitch_steps = -semitones  # Compensate pitch change
        elif method == 'custom':
            K = K_CUSTOM
            N = N_CUSTOM
            output_prefix = f'custom_tempo_{song.id}_{semitones}_{timestamp}'
            rate = 2 ** (semitones * K / 7 / N)
            pitch_steps = -semitones * K / 7 / N  # Compensate pitch change
        else:
            raise ValueError(f"Unknown method: {method}")

        # Apply tempo shift
        y_shifted = librosa.effects.time_stretch(y, rate=rate)
        # Apply pitch correction
        y_shifted = librosa.effects.pitch_shift(y_shifted, sr=sr, n_steps=pitch_steps)
        output_path = os.path.join(TEMPO_SHIFT_DIR, f'{output_prefix}.mp3')
        sf.write(output_path, y_shifted, sr)
        logging.debug(f"Saved tempo-shifted file: {output_path}, exists: {os.path.exists(output_path)}")

        # Generate spectrogram
        spectrogram_path = os.path.join(TEMPO_SHIFT_DIR, f'spectrogram_{output_prefix}.png')
        plt.figure(figsize=(20, 4), dpi=150)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y_shifted)), ref=np.max),
                                 sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram: {song.name} ({method}, {semitones} semitones)')
        plt.savefig(spectrogram_path, bbox_inches='tight')
        plt.close()

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Audio file not saved: {output_path}")
        if not os.path.exists(spectrogram_path):
            raise FileNotFoundError(f"Spectrogram not saved: {spectrogram_path}")

        mfcc = librosa.feature.mfcc(y=y_shifted, sr=sr, n_mfcc=13).mean(axis=1).tolist()
        spectral_centroid = librosa.feature.spectral_centroid(y=y_shifted, sr=sr).mean()
        rms = librosa.feature.rms(y=y_shifted).mean()
        return {
            'output_path': output_path,
            'spectrogram_path': spectrogram_path,
            'mfcc': mfcc,
            'spectral_centroid': spectral_centroid,
            'rms': rms
        }
    except Exception as e:
        logging.error(f"Tempo shift error ({method}): {e}")
        raise


@app.route('/song_processing/<int:song_id>', methods=['GET', 'POST'])
@login_required
def song_processing(song_id):
    song = Song.query.get_or_404(song_id)
    if song.playlist.user_id != current_user.id:
        flash('You cannot process this song.')
        return redirect(url_for('playlists_page'))

    analysis_result = None
    tempo_shift_results = []
    metrics = {}
    fingerprint_result = None

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'download' and not song.file_path:
            sp = get_spotify_client()
            if not sp:
                return redirect(url_for('login_spotify'))
            try:
                track = sp.track(song.spotify_id)
                track_name = track['name']
                artist_name = track['artists'][0]['name']
                query = f"{track_name} {artist_name}"
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': os.path.join(DOWNLOAD_DIR, f"{track_name} - {artist_name}.%(ext)s"),
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '192',
                    }],
                    'quiet': True,
                    'no_warnings': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([f"ytsearch:{query}"])
                file_path = os.path.join(DOWNLOAD_DIR, f"{track_name} - {artist_name}.mp3")
                if os.path.exists(file_path):
                    song.file_path = file_path
                    db.session.commit()
                    flash(f'Song "{track_name}" successfully downloaded!')
                else:
                    flash(f'Failed to download song "{track_name}".')
                    return redirect(url_for('song_processing', song_id=song.id))
            except Exception as e:
                logging.error(f"Download error: {e}")
                flash(f'Failed to download song: {str(e)}')
                return redirect(url_for('song_processing', song_id=song.id))

        if action == 'analyze' and song.file_path:
            if not (song.analysis_report_path and os.path.exists(song.analysis_report_path)):
                try:
                    analysis_result = analyze_song(song.file_path, song)
                    flash(f'Analysis of song "{song.name}" successfully completed.')
                except Exception as e:
                    logging.error(f"Analysis error: {e}")
                    flash(f'Failed to analyze song: {str(e)}')
                    return redirect(url_for('song_processing', song_id=song.id))
            else:
                analysis_result = {
                    'summary': f"Tempo: {song.tempo:.2f} BPM, Duration: {song.duration:.2f} sec, "
                               f"Spectral Centroid: {song.spectral_centroid:.2f} Hz, Onsets: {song.onset_count}",
                    'spectrogram_path': song.spectrogram_path,
                    'report_path': song.analysis_report_path,
                    'report_content': open(song.analysis_report_path, 'r',
                                           encoding='utf-8').read() if song.analysis_report_path else '',
                    'mfcc': librosa.feature.mfcc(y=librosa.load(song.file_path, sr=44100)[0], sr=44100, n_mfcc=13).mean(
                        axis=1).tolist(),
                    'spectral_centroid': song.spectral_centroid,
                    'rms': librosa.feature.rms(y=librosa.load(song.file_path, sr=44100)[0]).mean(),
                    'data_area_path': song.analysis_report_path.replace('.txt',
                                                                        '.json') if song.analysis_report_path else None,
                    'data_area': json.load(open(song.analysis_report_path.replace('.txt', '.json'),
                                                'r')) if song.analysis_report_path and os.path.exists(
                        song.analysis_report_path.replace('.txt', '.json')) else None
                }

        if action == 'fingerprint' and song.file_path:
            if not (song.fingerprint_path and os.path.exists(song.fingerprint_path)):
                try:
                    fingerprint_result = create_fingerprint(song.file_path, song)
                    flash(f'Fingerprint for song "{song.name}" successfully created.')
                except Exception as e:
                    logging.error(f"Fingerprint error: {e}")
                    flash(f'Failed to create fingerprint: {str(e)}')
                    return redirect(url_for('song_processing', song_id=song.id))
            else:
                with open(song.fingerprint_path, 'rb') as f:
                    fingerprints = pickle.load(f)
                fingerprint_result = {
                    'fingerprint_path': song.fingerprint_path,
                    'fingerprint_count': len(fingerprints)
                }

        if action == 'process' and song.file_path:
            tempo_semitones = float(request.form.get('tempo_semitones', 0))
            tempo_methods = request.form.getlist('tempo_methods')

            try:
                # Tempo shifting with pitch correction
                for method in tempo_methods:
                    if method == 'standard' and not (
                            song.tempo_shifted_standard_path and os.path.exists(song.tempo_shifted_standard_path)):
                        result = tempo_shift_song(song.file_path, song, tempo_semitones, 'standard')
                        song.tempo_shifted_standard_path = result['output_path']
                        tempo_shift_results.append({
                            'method': 'Standard',
                            'audio_path': result['output_path'],
                            'spectrogram_path': result['spectrogram_path'],
                            'mfcc': result['mfcc'],
                            'spectral_centroid': result['spectral_centroid'],
                            'rms': result['rms']
                        })
                    elif method == 'custom' and not (
                            song.tempo_shifted_custom_path and os.path.exists(song.tempo_shifted_custom_path)):
                        result = tempo_shift_song(song.file_path, song, tempo_semitones, 'custom')
                        song.tempo_shifted_custom_path = result['output_path']
                        tempo_shift_results.append({
                            'method': 'Custom',
                            'audio_path': result['output_path'],
                            'spectrogram_path': result['spectrogram_path'],
                            'mfcc': result['mfcc'],
                            'spectral_centroid': result['spectral_centroid'],
                            'rms': result['rms']
                        })

                db.session.commit()
                flash('Song processing successfully completed.')

                # Compute metrics
                if analysis_result or song.file_path:
                    y_orig, sr = librosa.load(song.file_path, sr=44100)
                    mfcc_orig = librosa.feature.mfcc(y=y_orig, sr=sr, n_mfcc=13).mean(axis=1)
                    spectral_centroid_orig = librosa.feature.spectral_centroid(y=y_orig, sr=sr).mean()
                    rms_orig = librosa.feature.rms(y=y_orig).mean()
                    metrics['Original'] = {
                        'mfcc_correlation': 1.0,
                        'spectral_centroid': spectral_centroid_orig,
                        'rms': rms_orig
                    }

                    for result in tempo_shift_results:
                        try:
                            correlation = np.corrcoef(mfcc_orig, result['mfcc'])[0, 1]
                            metrics[result['method']] = {
                                'mfcc_correlation': correlation,
                                'spectral_centroid_diff': abs(spectral_centroid_orig - result['spectral_centroid']),
                                'rms_diff': abs(rms_orig - result['rms'])
                            }
                        except Exception as e:
                            logging.error(f"Metrics computation error for {result['method']}: {e}")
                            metrics[result['method']] = {
                                'mfcc_correlation': 'Error',
                                'spectral_centroid_diff': 'Error',
                                'rms_diff': 'Error'
                            }

            except Exception as e:
                logging.error(f"Processing error: {e}")
                flash(f'Failed to process song: {str(e)}')
                return redirect(url_for('song_processing', song_id=song.id))

    # Add original track for comparison
    if song.file_path and not tempo_shift_results:
        if song.spectrogram_path and os.path.exists(song.spectrogram_path):
            tempo_shift_results.append({
                'method': 'Original',
                'audio_path': song.file_path,
                'spectrogram_path': song.spectrogram_path,
                'mfcc': librosa.feature.mfcc(y=librosa.load(song.file_path, sr=44100)[0], sr=44100, n_mfcc=13).mean(
                    axis=1).tolist(),
                'spectral_centroid': song.spectral_centroid or librosa.feature.spectral_centroid(
                    y=librosa.load(song.file_path, sr=44100)[0], sr=44100).mean(),
                'rms': librosa.feature.rms(y=librosa.load(song.file_path, sr=44100)[0]).mean()
            })

    return render_template('song_processing.html',
                           song=song,
                           analysis_result=analysis_result,
                           tempo_shift_results=tempo_shift_results,
                           metrics=metrics,
                           fingerprint_result=fingerprint_result)


@app.route('/compare_fingerprints')
@login_required
def compare_fingerprints_route():
    song_id1 = request.args.get('song_id1', type=int)
    song_id2 = request.args.get('song_id2', type=int)
    force_recompute = request.args.get('force_recompute', default='false').lower() == 'true'
    if not (song_id1 and song_id2):
        flash('Please select two songs to compare.')
        return redirect(url_for('playlists_page'))
    song1 = Song.query.get_or_404(song_id1)
    song2 = Song.query.get_or_404(song_id2)
    if song1.playlist.user_id != current_user.id or song2.playlist.user_id != current_user.id:
        flash('You cannot compare these songs.')
        return redirect(url_for('playlists_page'))
    if not (song1.fingerprint_path and song2.fingerprint_path and os.path.exists(song1.fingerprint_path) and os.path.exists(song2.fingerprint_path)):
        flash('One or both songs do not have fingerprints. Please create fingerprints first.')
        return redirect(url_for('song_processing', song_id=song_id1))
    try:
        comparison_result = compare_fingerprints(song1.fingerprint_path, song2.fingerprint_path, song1.id, song2.id, force_recompute)
        return render_template('compare_fingerprints.html',
                              song1=song1,
                              song2=song2,
                              similarity=comparison_result['overall_similarity'],
                              details=comparison_result['details'],
                              cached=comparison_result['cached'])
    except Exception as e:
        logging.error(f"Comparison error: {e}")
        flash(f'Failed to compare fingerprints: {str(e)}')
        return redirect(url_for('playlists_page'))


@app.route('/song_processing_file/<path:filename>')
@login_required
def serve_song_processing_file(filename):
    for directory in [DOWNLOAD_DIR, ANALYSIS_DIR, TEMPO_SHIFT_DIR, FINGERPRINT_DIR]:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            return send_file(file_path)
    flash('File not found.')
    return redirect(url_for('playlists_page'))


@app.route('/song_player/<int:song_id>')
@login_required
def song_player(song_id):
    song = Song.query.get_or_404(song_id)
    if song.playlist.user_id != current_user.id:
        flash('You cannot view this song.')
        return redirect(url_for('playlists_page'))
    if not song.file_path or not os.path.exists(song.file_path):
        flash('Audio file not found. Please download the song first.')
        return redirect(url_for('song_processing', song_id=song.id))
    if not song.spectrogram_path or not os.path.exists(song.spectrogram_path):
        try:
            analysis_result = analyze_song(song.file_path, song)
            song.spectrogram_path = analysis_result['spectrogram_path']
            db.session.commit()
        except Exception as e:
            logging.error(f"Spectrogram generation error: {e}")
            flash('Failed to generate spectrogram.')
            return redirect(url_for('song_processing', song_id=song.id))
    data_area = None
    data_area_path = song.analysis_report_path.replace('.txt', '.json') if song.analysis_report_path else None
    if data_area_path and os.path.exists(data_area_path):
        with open(data_area_path, 'r') as f:
            data_area = json.load(f)
    return render_template('song_player.html', song=song, data_area=data_area)


def standard_fft_spectrogram(signal, sample_rate, window_size, step_size):
    spectrogram = []
    window = np.hanning(window_size)
    for start in range(0, len(signal) - window_size, step_size):
        segment = signal[start:start + window_size] * window
        fft_result = np.fft.fft(segment)
        magnitude = np.abs(fft_result[:window_size // 2])
        spectrogram.append(magnitude)
    spectrogram = np.array(spectrogram).T
    time = np.arange(spectrogram.shape[1]) * (step_size / sample_rate)
    freq = np.fft.fftfreq(window_size, d=1 / sample_rate)[:window_size // 2]
    return spectrogram, time, freq


def process_full_audio(signal, sample_rate, window_size, step_size, chunk_duration_sec):
    chunk_size = int(chunk_duration_sec * sample_rate)
    full_spectrogram = []
    full_time = []
    for i in range(0, len(signal), chunk_size):
        chunk = signal[i:i + chunk_size]
        if len(chunk) < window_size:
            break
        spectrogram, time, freq = standard_fft_spectrogram(chunk, sample_rate, window_size, step_size)
        if len(full_spectrogram) == 0:
            full_spectrogram = spectrogram
            full_time = time + (i / sample_rate)
        else:
            full_spectrogram = np.hstack((full_spectrogram, spectrogram))
            full_time = np.concatenate((full_time, time + (i / sample_rate)))
    total_duration = len(signal) / sample_rate
    if len(full_time) > 0 and full_time[-1] > total_duration:
        full_time = full_time[:int(total_duration / (step_size / sample_rate)) + 1]
        full_spectrogram = full_spectrogram[:, :len(full_time)]
    return full_spectrogram, full_time, freq


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)