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
import hashlib
import json
import librosa.sequence
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats import shapiro
from uuid import uuid4
from scipy.spatial.distance import euclidean, cosine


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
for directory in [DOWNLOAD_DIR, ANALYSIS_DIR, TEMPO_SHIFT_DIR, FINGERPRINT_DIR]:
    os.makedirs(directory, exist_ok=True)

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



def create_fingerprint(file_path, song, sr=22050, n_fft=4096, hop_length=256, peak_height=-40,
                      peak_distance=10, peak_prominence=3, chunk_duration=10):
    """
    Creates audio fingerprints for a song by processing it in chunks and saves visualizations.

    Args:
        file_path (str): Path to the audio file.
        song: Song object from the database.
        sr (int): Sampling rate.
        n_fft (int): FFT window size (increased to 4096 for better frequency resolution).
        hop_length (int): Hop length for STFT (decreased to 256 for better time resolution).
        peak_height (float): Minimum peak height in dB.
        peak_distance (int): Minimum distance between peaks.
        peak_prominence (float): Minimum peak prominence.
        chunk_duration (int): Duration of each chunk in seconds (default: 10s).

    Returns:
        dict: Single fingerprint with all peaks and visualization paths.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        fingerprint_dir = os.path.join(os.getcwd(), 'fingerprints')
        os.makedirs(fingerprint_dir, exist_ok=True)

        # Load audio
        y, sr = librosa.load(file_path, sr=sr)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < 1:
            raise ValueError("Audio file is too short")
        logging.info(f"Loaded {song.name}: duration={duration:.2f}s, samples={len(y)}")

        # Split audio into chunks
        chunk_size = int(sr * chunk_duration)
        chunks = [y[i:i + chunk_size] for i in range(0, len(y), chunk_size)]
        logging.info(f"Split into {len(chunks)} chunks of {chunk_duration}s each")

        # Initialize storage for fingerprints
        all_peaks = []
        all_stft_db = []
        all_chroma = []
        all_mfcc = []

        # Process each chunk
        for i, chunk in enumerate(chunks):
            if len(chunk) == 0:
                continue

            # Compute STFT
            stft = np.abs(librosa.stft(chunk, n_fft=n_fft, hop_length=hop_length, window='hann'))
            stft_db = librosa.amplitude_to_db(stft, ref=np.max)
            all_stft_db.append(stft_db)

            # Peak detection
            peaks = []
            for t in range(stft_db.shape[1]):
                frame_peaks, properties = find_peaks(stft_db[:, t], height=peak_height,
                                                    distance=peak_distance, prominence=peak_prominence)
                for f_idx, peak_idx in enumerate(frame_peaks):
                    freq = peak_idx * sr / n_fft
                    if 200 < freq < 20000:
                        amplitude = properties['peak_heights'][f_idx]
                        # Adjust time frame to global time
                        global_t = i * chunk_duration + (t * hop_length / sr)
                        peaks.append((peak_idx, global_t, amplitude))
            peaks = np.array(peaks, dtype=[('freq_bin', int), ('time_frame', float), ('amplitude', float)])
            all_peaks.extend(peaks)

            # Compute features
            mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13, hop_length=hop_length)
            chroma = librosa.feature.chroma_stft(y=chunk, sr=sr, n_fft=n_fft, hop_length=hop_length)
            all_mfcc.append(mfcc)
            all_chroma.append(chroma)

        all_peaks = np.array(all_peaks, dtype=[('freq_bin', int), ('time_frame', float), ('amplitude', float)])
        logging.info(f"Generated {len(all_peaks)} peaks across all chunks")

        # Create a single fingerprint with all peaks
        single_fingerprint = {
            'peaks': all_peaks,
            'mfcc_mean': np.mean([librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13, hop_length=hop_length).mean(axis=1)
                                for chunk in chunks], axis=0),
            'chroma_mean': np.mean([librosa.feature.chroma_stft(y=chunk, sr=sr, n_fft=n_fft, hop_length=hop_length).mean(axis=1)
                                  for chunk in chunks], axis=0),
            'tempo': np.mean([float(librosa.beat.beat_track(y=chunk, sr=sr, hop_length=hop_length)[0][0])
                            if isinstance(librosa.beat.beat_track(y=chunk, sr=sr, hop_length=hop_length)[0], np.ndarray)
                            else float(librosa.beat.beat_track(y=chunk, sr=sr, hop_length=hop_length)[0])
                            for chunk in chunks]),
            'spectral_centroid': np.mean([librosa.feature.spectral_centroid(y=chunk, sr=sr, n_fft=n_fft, hop_length=hop_length)[0].mean()
                                       for chunk in chunks]),
            'spectral_rolloff': np.mean([librosa.feature.spectral_rolloff(y=chunk, sr=sr, n_fft=n_fft, hop_length=hop_length)[0].mean()
                                      for chunk in chunks]),
            'zcr': np.mean([librosa.feature.zero_crossing_rate(y=chunk, hop_length=hop_length)[0].mean()
                          for chunk in chunks]),
            'duration': duration,
            'file_path': file_path
        }

        # Validate fingerprint
        is_valid, validation_message = validate_fingerprint(single_fingerprint)
        if not is_valid:
            raise ValueError(f"Invalid fingerprint: {validation_message}")

        # Save fingerprint
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fingerprint_id = str(uuid4())
        fingerprint_path = os.path.join(fingerprint_dir, f"fingerprint_{song.id}_{timestamp}_{fingerprint_id}.pkl")
        with open(fingerprint_path, 'wb') as f:
            pickle.dump(single_fingerprint, f)  # Save single fingerprint

        # Generate and save visualizations
        plt.style.use('seaborn-v0_8')

        # Combined Spectrogram
        spectrogram_path = os.path.join(fingerprint_dir, f"combined_spectrogram_{song.id}_{timestamp}_{fingerprint_id}.png")
        try:
            plt.figure(figsize=(20, 10))  # Increased width for better detail
            # Concatenate STFT along time axis
            combined_stft_db = np.hstack(all_stft_db)
            if combined_stft_db.size == 0:
                raise ValueError("Combined STFT is empty")
            librosa.display.specshow(combined_stft_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz',
                                    cmap='magma')  # Changed colormap for better contrast
            plt.scatter(all_peaks['time_frame'], all_peaks['freq_bin'] * sr / n_fft, c='red', s=10, label='Peaks')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Combined Spectrogram: {song.name}')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(spectrogram_path, bbox_inches='tight', dpi=300)  # Increased DPI for detail
            plt.close()
        except Exception as e:
            logging.error(f"Failed to generate spectrogram: {e}")
            spectrogram_path = None

        # Combined Chromagram
        chromagram_path = os.path.join(fingerprint_dir, f"combined_chromagram_{song.id}_{timestamp}_{fingerprint_id}.png")
        try:
            plt.figure(figsize=(20, 10))
            combined_chroma = np.hstack(all_chroma)
            if combined_chroma.size == 0:
                raise ValueError("Combined chroma is empty")
            librosa.display.specshow(combined_chroma, y_axis='chroma', x_axis='time', hop_length=hop_length, cmap='magma')
            plt.colorbar()
            plt.title(f'Combined Chromagram: {song.name}')
            plt.xlabel('Time (s)')
            plt.ylabel('Pitch Class')
            plt.tight_layout()
            plt.savefig(chromagram_path, bbox_inches='tight', dpi=300)
            plt.close()
        except Exception as e:
            logging.error(f"Failed to generate chromagram: {e}")
            chromagram_path = None

        # Combined MFCC
        mfcc_path = os.path.join(fingerprint_dir, f"combined_mfcc_{song.id}_{timestamp}_{fingerprint_id}.png")
        try:
            plt.figure(figsize=(20, 10))
            combined_mfcc = np.hstack(all_mfcc)
            if combined_mfcc.size == 0:
                raise ValueError("Combined MFCC is empty")
            librosa.display.specshow(combined_mfcc, x_axis='time', hop_length=hop_length, cmap='magma')
            plt.colorbar()
            plt.title(f'Combined MFCC: {song.name}')
            plt.xlabel('Time (s)')
            plt.ylabel('MFCC Coefficients')
            plt.tight_layout()
            plt.savefig(mfcc_path, bbox_inches='tight', dpi=300)
            plt.close()
        except Exception as e:
            logging.error(f"Failed to generate MFCC: {e}")
            mfcc_path = None

        # Update song in database with available paths
        song.fingerprint_path = fingerprint_path
        song.tempo = single_fingerprint['tempo']
        song.spectral_centroid = single_fingerprint['spectral_centroid'].mean()
        song.duration = duration
        song.scatter_plot_path = spectrogram_path
        song.chromagram_path = chromagram_path
        song.mfcc_path = mfcc_path
        db.session.commit()

        logging.info(f"Fingerprint created for {song.name}: {fingerprint_path}")
        logging.debug(f"Returning fingerprint: {list(single_fingerprint.keys())}")  # Debug log
        return {
            'fingerprint_path': fingerprint_path,
            'fingerprint_count': len(all_peaks),
            'scatter_plot_path': spectrogram_path,
            'chromagram_path': chromagram_path,
            'mfcc_path': mfcc_path,
            'features': {
                'tempo': single_fingerprint['tempo'],
                'spectral_centroid': single_fingerprint['spectral_centroid'],
                'spectral_rolloff': single_fingerprint['spectral_rolloff'],
                'zcr': single_fingerprint['zcr'],
                'duration': single_fingerprint['duration']
            }
        }

    except Exception as e:
        logging.error(f"{song.name}: Fingerprint creation failed - {str(e)}")
        raise

def compare_fingerprints(fingerprint_path1, fingerprint_path2, song_id1, song_id2, force_recompute=False, sr=22050,hop_length=512, n_fft=2048):
    """
    Compares two audio fingerprints with normalization and caching.

    Args:
        fingerprint_path1 (str): Path to the first fingerprint file.
        fingerprint_path2 (str): Path to the second fingerprint file.
        song_id1 (int): ID of the first song.
        song_id2 (int): ID of the second song.
        force_recompute (bool): Whether to force recomputation.
        sr (int): Sampling rate.
        hop_length (int): Hop length for normalization.
        n_fft (int): FFT window size for normalization.

    Returns:
        dict: Comparison results with similarity metrics.
    """
    try:
        # Check cache
        if not force_recompute:
            comparison = FingerprintComparison.query.filter(
                ((FingerprintComparison.song_id1 == song_id1) & (FingerprintComparison.song_id2 == song_id2)) |
                ((FingerprintComparison.song_id1 == song_id2) & (FingerprintComparison.song_id2 == song_id1))
            ).first()
            if comparison:
                try:
                    with open(fingerprint_path1, 'rb') as f1, open(fingerprint_path2, 'rb') as f2:
                        hash1 = hashlib.md5(f1.read()).hexdigest()
                        hash2 = hashlib.md5(f2.read()).hexdigest()
                    if comparison.fingerprint_hash1 == hash1 and comparison.fingerprint_hash2 == hash2:
                        logging.info(f"Loaded cached comparison for songs {song_id1} and {song_id2}")
                        return {
                            'overall_similarity': comparison.overall_similarity,
                            'details': json.loads(comparison.details),
                            'cached': True
                        }
                except Exception as e:
                    logging.warning(f"Failed to validate cached comparison: {e}, forcing recompute")

        # Load fingerprints
        with open(fingerprint_path1, 'rb') as f1, open(fingerprint_path2, 'rb') as f2:
            fp1 = pickle.load(f1)
            fp2 = pickle.load(f2)

        # Normalize peaks
        peaks1 = np.array([(p['time_frame'] * hop_length / sr / fp1['duration'],
                            p['freq_bin'] * sr / n_fft / 5000) for p in fp1['peaks']])
        peaks2 = np.array([(p['time_frame'] * hop_length / sr / fp2['duration'],
                            p['freq_bin'] * sr / n_fft / 5000) for p in fp2['peaks']])

        # Compute peak similarity
        peak_similarity = 0
        if len(peaks1) > 0 and len(peaks2) > 0:
            peaks1 = peaks1[:min(10000, len(peaks1))]
            peaks2 = peaks2[:min(10000, len(peaks2))]
            if peaks1.size > 0 and peaks2.size > 0:
                distance, _ = fastdtw(peaks1, peaks2, dist=euclidean)
                time_std1 = np.std(peaks1[:, 0]) if len(peaks1) > 1 else 1.0
                time_std2 = np.std(peaks2[:, 0]) if len(peaks2) > 1 else 1.0
                freq_std1 = np.std(peaks1[:, 1]) if len(peaks1) > 1 else 1.0
                freq_std2 = np.std(peaks2[:, 1]) if len(peaks2) > 1 else 1.0
                scale_factor = max(len(peaks1), len(peaks2)) / 2
                max_distance = np.sqrt((time_std1 + time_std2) ** 2 + (freq_std1 + freq_std2) ** 2) * scale_factor * 2
                peak_similarity = max(0, min(100, (1 - distance / max_distance) * 100 * 1.2))
                logging.info(
                    f"Peak similarity: {peak_similarity:.2f}%, distance={distance:.2f}, max_distance={max_distance:.2f}")

        # Compute MFCC similarity
        mfcc1, mfcc2 = fp1['mfcc_mean'].flatten(), fp2['mfcc_mean'].flatten()
        mfcc_distance = euclidean(mfcc1, mfcc2)
        mfcc_similarity = max(0, min(100, (1 - mfcc_distance / (np.std(mfcc1) + np.std(mfcc2) + 1e-6)) * 100))
        logging.info(f"MFCC similarity: {mfcc_similarity:.2f}%")

        # Compute Chroma similarity
        chroma1, chroma2 = fp1['chroma_mean'].flatten(), fp2['chroma_mean'].flatten()
        chroma1 = (chroma1 - np.min(chroma1)) / (np.max(chroma1) - np.min(chroma1) + 1e-6)
        chroma2 = (chroma2 - np.min(chroma2)) / (np.max(chroma2) - np.min(chroma2) + 1e-6)
        chroma_cosine = cosine(chroma1, chroma2)
        chroma_similarity = max(0, min(100, (1 - chroma_cosine / 2) * 100))
        logging.info(f"Chroma similarity: {chroma_similarity:.2f}%")

        # Compute other similarities
        tempo_diff = abs(fp1['tempo'] - fp2['tempo'])
        tempo_similarity = max(0, 1 - tempo_diff / max(fp1['tempo'], fp2['tempo'])) * 100
        centroid_diff = abs(fp1['spectral_centroid'].mean() - fp2['spectral_centroid'].mean())
        centroid_similarity = max(0, 1 - centroid_diff / max(fp1['spectral_centroid'].mean(),
                                                             fp2['spectral_centroid'].mean())) * 100
        rolloff_diff = abs(fp1['spectral_rolloff'].mean() - fp2['spectral_rolloff'].mean())
        rolloff_similarity = max(0, 1 - rolloff_diff / max(fp1['spectral_rolloff'].mean(),
                                                           fp2['spectral_rolloff'].mean())) * 100
        zcr_diff = abs(fp1['zcr'].mean() - fp2['zcr'].mean())
        zcr_similarity = max(0, 1 - zcr_diff / max(fp1['zcr'].mean(), fp2['zcr'].mean())) * 100

        # Compute overall similarity
        weights = [0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]
        overall_similarity = (
                weights[0] * peak_similarity + weights[1] * mfcc_similarity +
                weights[2] * chroma_similarity + weights[3] * tempo_similarity +
                weights[4] * centroid_similarity + weights[5] * rolloff_similarity +
                weights[6] * zcr_similarity
        )
        overall_similarity = max(0, min(100, overall_similarity))
        logging.info(f"Overall similarity: {overall_similarity:.2f}%")

        # Prepare details
        details = {
            'Spectral Peaks': peak_similarity,
            'MFCC': mfcc_similarity,
            'Chroma': chroma_similarity,
            'Tempo': tempo_similarity,
            'Spectral Centroid': centroid_similarity,
            'Spectral Rolloff': rolloff_similarity,
            'Zero Crossing Rate': zcr_similarity
        }

        # Save to cache
        try:
            with open(fingerprint_path1, 'rb') as f1, open(fingerprint_path2, 'rb') as f2:
                hash1 = hashlib.md5(f1.read()).hexdigest()
                hash2 = hashlib.md5(f2.read()).hexdigest()
            comparison = FingerprintComparison.query.filter(
                ((FingerprintComparison.song_id1 == song_id1) & (FingerprintComparison.song_id2 == song_id2)) |
                ((FingerprintComparison.song_id1 == song_id2) & (FingerprintComparison.song_id2 == song_id1))
            ).first()
            if comparison:
                comparison.overall_similarity = overall_similarity
                comparison.details = json.dumps(details)
                comparison.fingerprint_hash1 = hash1
                comparison.fingerprint_hash2 = hash2
                comparison.updated_at = db.func.current_timestamp()
            else:
                comparison = FingerprintComparison(
                    song_id1=song_id1,
                    song_id2=song_id2,
                    overall_similarity=overall_similarity,
                    details=json.dumps(details),
                    fingerprint_hash1=hash1,
                    fingerprint_hash2=hash2
                )
                db.session.add(comparison)
            db.session.commit()
            logging.info(f"Cached comparison for songs {song_id1} and {song_id2}")
        except Exception as e:
            logging.error(f"Failed to cache comparison: {e}")

        # Generate visualizations
        plt.style.use('seaborn-v0_8')
        fingerprint_dir = os.path.join(os.getcwd(), 'fingerprints')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        comparison_id = str(uuid4())

        # Similarity bar plot
        similarity_plot = os.path.join(fingerprint_dir,
                                       f"similarity_{song_id1}_{song_id2}_{timestamp}_{comparison_id}.png")
        metrics = ['Overall', 'Peaks', 'MFCC', 'Chroma', 'Tempo', 'Centroid', 'Rolloff', 'ZCR']
        values = [overall_similarity, peak_similarity, mfcc_similarity, chroma_similarity,
                  tempo_similarity, centroid_similarity, rolloff_similarity, zcr_similarity]
        plt.figure(figsize=(12, 6))
        bars = plt.bar(metrics, values, color='skyblue', edgecolor='black')
        plt.ylim(0, 100)
        plt.title(f'Similarity: {os.path.basename(fp1["file_path"])} vs {os.path.basename(fp2["file_path"])}')
        plt.ylabel('Similarity (%)')
        plt.xlabel('Metrics')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(similarity_plot, bbox_inches='tight', dpi=150)
        plt.close()

        # MFCC and Chroma comparison plot
        feature_plot = os.path.join(fingerprint_dir,
                                    f"feature_comparison_{song_id1}_{song_id2}_{timestamp}_{comparison_id}.png")
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(mfcc1, label=os.path.basename(fp1['file_path']), color='blue')
        plt.plot(mfcc2, label=os.path.basename(fp2['file_path']), color='orange')
        plt.title('MFCC Mean Comparison')
        plt.xlabel('Coefficient Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(chroma1, label=os.path.basename(fp1['file_path']), color='blue')
        plt.plot(chroma2, label=os.path.basename(fp2['file_path']), color='orange')
        plt.title('Chroma Mean Comparison')
        plt.xlabel('Pitch Class Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(feature_plot, bbox_inches='tight', dpi=150)
        plt.close()

        return {
            'overall_similarity': overall_similarity,
            'details': details,
            'cached': False,
            'similarity_plot': similarity_plot,
            'feature_plot': feature_plot
        }

    except Exception as e:
        logging.error(f"Fingerprint comparison error: {str(e)}")
        raise


def validate_fingerprint(fingerprint, min_peaks=100, max_duration=600):
    """
    Validates the audio fingerprint.

    Args:
        fingerprint (dict): Fingerprint dictionary.
        min_peaks (int): Minimum number of peaks.
        max_duration (int): Maximum duration in seconds.

    Returns:
        tuple: (is_valid, validation_message)
    """
    try:
        if len(fingerprint['peaks']) < min_peaks:
            return False, f"Too few peaks ({len(fingerprint['peaks'])} < {min_peaks})"
        if fingerprint['tempo'] <= 0 or fingerprint['tempo'] > 300:
            return False, f"Invalid tempo: {fingerprint['tempo']} BPM"
        if fingerprint['duration'] <= 0 or fingerprint['duration'] > max_duration:
            return False, f"Invalid duration: {fingerprint['duration']} seconds"
        if fingerprint['mfcc_mean'].shape != (13,):
            return False, f"Invalid MFCC mean shape: {fingerprint['mfcc_mean'].shape}"
        if fingerprint['chroma_mean'].shape != (12,):
            return False, f"Invalid chroma mean shape: {fingerprint['chroma_mean'].shape}"
        if np.any(np.isnan(fingerprint['mfcc_mean'])) or np.any(np.isnan(fingerprint['chroma_mean'])):
            return False, "NaN values found in MFCC or chroma mean"
        time_frames = fingerprint['peaks']['time_frame']
        if len(time_frames) > 0:
            time_span = max(time_frames) - min(time_frames)
            if time_span < fingerprint['duration'] * 0.5:
                return False, "Peaks are too concentrated in time"
        return True, "Fingerprint is valid"
    except KeyError as e:
        return False, f"Missing key in fingerprint: {str(e)}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

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
            pitch_steps = -semitones
        elif method == 'custom':
            K = K_CUSTOM
            N = N_CUSTOM
            output_prefix = f'custom_tempo_{song.id}_{semitones}_{timestamp}'
            rate = 2 ** (semitones * K / 7 / N)
            pitch_steps = -semitones * K / 7 / N
        else:
            raise ValueError(f"Unknown method: {method}")

        # Apply tempo shift
        y_shifted = librosa.effects.time_stretch(y, rate=rate)
        if len(y_shifted) == 0:
            raise ValueError("Time stretch resulted in empty array")
        # Apply pitch correction
        y_shifted = librosa.effects.pitch_shift(y_shifted, sr=sr, n_steps=pitch_steps)
        output_path = os.path.join(TEMPO_SHIFT_DIR, f'{output_prefix}.mp3')
        sf.write(output_path, y_shifted, sr)
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Failed to save audio file: {output_path}")

        # Generate spectrogram
        spectrogram_path = os.path.join(TEMPO_SHIFT_DIR, f'spectrogram_{output_prefix}.png')
        plt.figure(figsize=(20, 4), dpi=150)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y_shifted)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram: {song.name} ({method}, {semitones} semitones)')
        plt.savefig(spectrogram_path, bbox_inches='tight')
        plt.close()
        if not os.path.exists(spectrogram_path):
            raise FileNotFoundError(f"Failed to save spectrogram: {spectrogram_path}")

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
    except (librosa.ParameterError, ValueError) as e:
        logging.error(f"Audio processing error ({method}): {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in tempo shift ({method}): {e}")
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
                    'report_content': open(song.analysis_report_path, 'r', encoding='utf-8').read() if song.analysis_report_path else '',
                    'mfcc': librosa.feature.mfcc(y=librosa.load(song.file_path, sr=44100)[0], sr=44100, n_mfcc=13).mean(
                        axis=1).tolist(),
                    'spectral_centroid': song.spectral_centroid,
                    'rms': librosa.feature.rms(y=librosa.load(song.file_path, sr=44100)[0]).mean(),
                    'data_area_path': song.analysis_report_path.replace('.txt', '.json') if song.analysis_report_path else None,
                    'data_area': json.load(open(song.analysis_report_path.replace('.txt', '.json'), 'r')) if song.analysis_report_path and os.path.exists(song.analysis_report_path.replace('.txt', '.json')) else None
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
                logging.debug(f"Loaded fingerprints keys: {list(fingerprints.keys())}")
                if 'peaks' not in fingerprints:
                    logging.warning(f"Old fingerprint format detected for song {song.id}. Recreating fingerprint.")
                    try:
                        fingerprint_result = create_fingerprint(song.file_path, song)
                        flash(f'Fingerprint for song "{song.name}" successfully recreated.')
                    except Exception as e:
                        logging.error(f"Recreation error: {e}")
                        flash(f'Failed to recreate fingerprint: {str(e)}')
                        return redirect(url_for('song_processing', song_id=song.id))
                else:
                    fingerprint_result = {
                        'fingerprint_path': song.fingerprint_path,
                        'fingerprint_count': len(fingerprints['peaks']),
                        'scatter_plot_path': song.scatter_plot_path,
                        'chromagram_path': song.chromagram_path,
                        'mfcc_path': song.mfcc_path,
                        'features': {
                            'tempo': fingerprints.get('tempo', 0.0),
                            'spectral_centroid': fingerprints.get('spectral_centroid', 0.0).mean() if 'spectral_centroid' in fingerprints else 0.0,
                            'spectral_rolloff': fingerprints.get('spectral_rolloff', 0.0).mean() if 'spectral_rolloff' in fingerprints else 0.0,
                            'zcr': fingerprints.get('zcr', 0.0),
                            'duration': fingerprints.get('duration', 0.0)
                        }
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
                              cached=comparison_result['cached'],
                              similarity_plot=comparison_result.get('similarity_plot'),
                              feature_plot=comparison_result.get('feature_plot'))
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
    for i in range(0, len(signal) - window_size + 1, chunk_size):
        chunk = signal[i:i + chunk_size]
        if len(chunk) < window_size:
            chunk = np.pad(chunk, (0, window_size - len(chunk)), mode='constant')
        spectrogram, time, freq = standard_fft_spectrogram(chunk, sample_rate, window_size, step_size)
        if len(full_spectrogram) == 0:
            full_spectrogram = spectrogram
            full_time = time + (i / sample_rate)
        else:
            full_spectrogram = np.hstack((full_spectrogram, spectrogram))
            full_time = np.concatenate((full_time, time + (i / sample_rate) + full_time[-1] - time[0]))
    total_duration = len(signal) / sample_rate
    if len(full_time) > total_duration:
        idx = np.searchsorted(full_time, total_duration, side='right')
        full_time = full_time[:idx]
        full_spectrogram = full_spectrogram[:, :idx]
    return full_spectrogram, full_time, freq


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)