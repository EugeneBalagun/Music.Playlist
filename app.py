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


def create_fingerprint(file_path, song, sr=44100):
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=sr, duration=None)
        duration = librosa.get_duration(y=y, sr=sr)
        logging.info(f"Loaded {song.name}: duration={duration:.2f}s, samples={len(y)}")

        # STFT parameters
        n_fft = 512
        hop_length = n_fft // 8
        window = np.hanning(n_fft)
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window, center=False))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        S_db = S_db[(freqs >= 300) & (freqs <= 4000), :]
        logging.info(f"STFT shape: {S_db.shape}, expected frames: {len(y) / hop_length:.0f}")

        # RMS for noise filtering
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms_threshold = np.percentile(rms, 1)
        rms_var = np.var(rms)  # Добавляем дисперсию RMS
        logging.info(
            f"RMS threshold: {rms_threshold:.2f}, RMS min: {np.min(rms):.4f}, max: {np.max(rms):.4f}, var: {rms_var:.4f}")

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length).mean()
        logging.info(f"Zero Crossing Rate: {zcr:.4f}")

        # Rhythmic Complexity (упрощенная оценка через частоту атак)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        rhythmic_complexity = len(onsets) / (duration + 1e-6)  # Нормализованная сложность
        logging.info(f"Rhythmic Complexity: {rhythmic_complexity:.4f}")

        # Peak detection
        fingerprints = []
        time_peaks = {}
        energy_threshold = np.percentile(S_db.flatten(), 50)  # Adaptive threshold
        logging.info(f"Energy threshold: {energy_threshold:.2f} dB")

        for t in range(S_db.shape[1]):
            time = t * hop_length / sr
            if time > duration + (hop_length / sr):
                logging.debug(f"Frame {t} skipped: time {time:.2f}s exceeds duration {duration:.2f}s")
                break
            if rms[t] < rms_threshold:
                logging.debug(f"Frame {t} skipped: RMS {rms[t]:.4f} < threshold {rms_threshold:.4f}, time={time:.2f}s")
                continue

            peaks, properties = find_peaks(S_db[:, t], height=energy_threshold, distance=50, prominence=10)
            if len(peaks) == 0:
                continue

            for i, peak_idx in enumerate(peaks):
                freq = freqs[(freqs >= 300) & (freqs <= 4000)][peak_idx]
                amplitude = properties['peak_heights'][i]
                if amplitude < -30:
                    continue

                key = round(time, 2)
                if key not in time_peaks:
                    time_peaks[key] = []
                too_close = False
                for existing_time, existing_freq in time_peaks[key]:
                    if abs(freq - existing_freq) < 50:
                        too_close = True
                        break
                if not too_close:
                    fingerprints.append((time, freq, amplitude))
                    time_peaks[key].append((time, freq))

        logging.info(
            f"Generated {len(fingerprints)} peaks, time range: {min([p[0] for p in fingerprints]):.2f} to {max([p[0] for p in fingerprints]):.2f}s")



        # Adaptive segmentation
        top_db = max(5, min(30, 20 + 10 * (np.mean(rms) / (np.max(rms) + 1e-6))))
        segments = librosa.effects.split(y, top_db=top_db)
        if len(segments) == 0:
            segments = [(0, len(y))]

        # Segment-wise tempo analysis
        tempos = []
        for start, end in segments:
            if end - start >= hop_length:
                seg_y = y[start:end]
                tempo, _ = librosa.beat.beat_track(y=seg_y, sr=sr, hop_length=hop_length)
                tempos.append(float(tempo))
        tempo = np.mean(tempos) if tempos else float(librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)[0])
        tempo_std = np.std(tempos) if tempos else 0

        # Normalized features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length).mean() / (sr / 100)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length).mean() / (sr / 100)
        spectral_bandwidth = (np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)) +
                              np.std(librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length))) / (sr / 20)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length).mean()
        spectral_flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length).mean()

        # Segment-wise MFCC and Chroma
        segment_mfccs = []
        segment_chromas = []
        for start, end in segments[:100]:
            if end - start >= hop_length:
                seg_y = y[start:end]
                seg_mfcc = librosa.feature.mfcc(y=seg_y, sr=sr, n_mfcc=13, hop_length=hop_length)
                seg_chroma = librosa.feature.chroma_cqt(y=seg_y, sr=sr, hop_length=hop_length)
                if seg_mfcc.shape[1] > 0 and seg_chroma.shape[1] > 0:
                    segment_mfccs.append(seg_mfcc.mean(axis=1) / (np.std(seg_mfcc, axis=1) + 1e-6))
                    segment_chromas.append(seg_chroma.mean(axis=1) / (np.std(seg_chroma, axis=1) + 1e-6))
        mfcc_mean = np.mean(segment_mfccs, axis=0) if segment_mfccs else np.zeros(13)
        chroma_mean = np.mean(segment_chromas, axis=0) if segment_chromas else np.zeros(12)

        # Additional features
        rms_mean = librosa.feature.rms(y=y, hop_length=hop_length).mean()
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time', hop_length=hop_length,
                                            backtrack=True, delta=0.5)
        onset_count = len(onsets)

        # Spectral entropy
        S_power = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)) ** 2
        S_power = S_power / (np.sum(S_power, axis=0, keepdims=True) + 1e-6)
        spectral_entropy = -np.sum(S_power * np.log2(S_power + 1e-10), axis=0).mean()

        # Segment count
        segment_count = max(1, len(onsets) // 10)

        # Visualizations
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fingerprint_dir = os.path.join(os.getcwd(), 'fingerprints')
        os.makedirs(fingerprint_dir, exist_ok=True)

        scatter_plot_path = os.path.join(fingerprint_dir, f'scatter_peaks_{song.id}_{timestamp}.svg')
        width_pixels = 1200
        fig_width = width_pixels / 100
        plt.figure(figsize=(fig_width, 8), dpi=100)
        times = [peak[0] for peak in fingerprints]
        freqs = [peak[1] for peak in fingerprints]
        amplitudes = [peak[2] for peak in fingerprints]
        logging.info(f"Time range of peaks: {min(times):.2f} to {max(times):.2f} seconds")
        vmin = np.min(amplitudes) if amplitudes else 0
        vmax = np.max(amplitudes) if amplitudes else 1
        plt.scatter(times, freqs, c=amplitudes, cmap='viridis', s=10, alpha=0.6, vmin=vmin, vmax=vmax)
        plt.colorbar(label='Amplitude (dB)')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title(f'Spectral Peaks Fingerprint: {song.name}')
        plt.xlim(0, duration)
        plt.ylim(300, 4000)
        plt.savefig(scatter_plot_path, format='svg', bbox_inches='tight', pad_inches=0)
        plt.close()

        chromagram_path = os.path.join(fingerprint_dir, f'chromagram_{song.id}_{timestamp}.png')
        plt.figure(figsize=(10, 6))
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        librosa.display.specshow(chroma_cqt, y_axis='chroma', x_axis='time', cmap='coolwarm')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Chromagram: {song.name}')
        plt.savefig(chromagram_path, bbox_inches='tight', dpi=100)
        plt.close()

        mfcc_path = os.path.join(fingerprint_dir, f'mfcc_{song.id}_{timestamp}.png')
        plt.figure(figsize=(10, 6))
        mfcc_full = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        librosa.display.specshow(mfcc_full, x_axis='time', cmap='coolwarm')
        plt.colorbar()
        plt.close()

        # Save fingerprint data
        fingerprint_data = {
            'peaks': fingerprints,
            'tempo': tempo,
            'tempo_std': tempo_std,
            'spectral_centroid': float(spectral_centroid),
            'spectral_rolloff': float(spectral_rolloff),
            'spectral_bandwidth': float(spectral_bandwidth),
            'spectral_contrast': float(spectral_contrast),
            'spectral_flatness': float(spectral_flatness),
            'mfcc': mfcc_mean.tolist(),
            'segment_mfccs': [seg.tolist() for seg in segment_mfccs],
            'segment_chromas': [seg.tolist() for seg in segment_chromas],
            'chroma': chroma_mean.tolist(),
            'rms': float(rms_mean),
            'rms_var': float(rms_var),  # Добавляем
            'onset_count': onset_count,
            'rhythmic_complexity': float(rhythmic_complexity),  # Добавляем
            'zcr': float(zcr),  # Добавляем
            'spectral_entropy': float(spectral_entropy),
            'segment_count': float(segment_count),
            'duration': duration
        }

        fingerprint_path = os.path.join(fingerprint_dir, f'fingerprint_{song.id}_{timestamp}.pkl')
        with open(fingerprint_path, 'wb') as f:
            pickle.dump(fingerprint_data, f)

        song.fingerprint_path = fingerprint_path
        song.tempo = tempo
        song.spectral_centroid = spectral_centroid
        song.duration = duration
        song.onset_count = onset_count
        song.scatter_plot_path = scatter_plot_path
        song.chromagram_path = chromagram_path
        song.mfcc_path = mfcc_path
        db.session.commit()

        return {
            'fingerprint_path': fingerprint_path,
            'fingerprint_count': len(fingerprints),
            'features': {
                'tempo': tempo,
                'tempo_std': tempo_std,
                'spectral_centroid': sr,
                'spectral_rolloff': spectral_rolloff,
                'spectral_bandwidth': spectral_bandwidth,
                'spectral_contrast': spectral_contrast,
                'spectral_flatness': spectral_flatness,
                'rms': rms_mean,
                'onset_count': onset_count,
                'spectral_entropy': spectral_entropy
            },
            'scatter_plot_path': scatter_plot_path,
            'chromagram_path': chromagram_path,
            'mfcc_path': mfcc_path
        }
    except Exception as e:
        logging.error(f"{song.name}: Fingerprint creation failed - {str(e)}")
        raise




def compare_fingerprints(fingerprint_path1, fingerprint_path2, song_id1, song_id2, force_recompute=False):
    """
    Compares two audio fingerprints with enhanced normalization and segment handling.

    Args:
        fingerprint_path1 (str): Path to the first fingerprint file.
        fingerprint_path2 (str): Path to the second fingerprint file.
        song_id1 (int): ID of the first song.
        song_id2 (int): ID of the second song.
        force_recompute (bool): Whether to force recomputation instead of using cache.

    Returns:
        dict: Comparison results with overall similarity and detailed metrics.
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
                    else:
                        logging.warning("Cached hashes do not match, forcing recompute")
                except Exception as e:
                    logging.warning(f"Failed to validate cached comparison: {e}, forcing recompute")

        # Load fingerprint data
        with open(fingerprint_path1, 'rb') as f:
            data1 = pickle.load(f)
        with open(fingerprint_path2, 'rb') as f:
            data2 = pickle.load(f)

        # Log and validate data
        required_keys = ['peaks', 'tempo', 'spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth',
                         'spectral_contrast', 'spectral_flatness', 'rms', 'rms_var', 'onset_count',
                         'rhythmic_complexity', 'zcr', 'segment_count', 'spectral_entropy', 'duration']
        for key in required_keys:
            if key not in data1 or data1[key] is None:
                logging.warning(f"Song {song_id1} missing or invalid key: {key}, using default 0")
                data1[key] = 0
            if key not in data2 or data2[key] is None:
                logging.warning(f"Song {song_id2} missing or invalid key: {key}, using default 0")
                data2[key] = 0
        logging.debug(f"Song {song_id1} data: peaks={len(data1['peaks'])}, segments={len(data1.get('segment_mfccs', []))}")
        logging.debug(f"Song {song_id2} data: peaks={len(data2['peaks'])}, segments={len(data2.get('segment_mfccs', []))}")

        # Compare spectral peaks with dynamic limit
        max_peaks = min(20000, max(len(data1['peaks']), len(data2['peaks'])))
        peaks1 = np.array([(t, f) for t, f, _ in data1['peaks']])[:max_peaks]
        peaks2 = np.array([(t, f) for t, f, _ in data2['peaks']])[:max_peaks]
        peak_similarity = 0
        if len(peaks1) > 0 and len(peaks2) > 0:
            try:
                distance, _ = fastdtw(peaks1, peaks2, dist=euclidean)
                max_distance = np.sqrt((max(data1['duration'], data2['duration'])**2 +
                                      (max(peaks1[:, 1].max(), peaks2[:, 1].max())**2) + 1e-6))
                peak_similarity = max(0, 1 - (distance.mean() / max_distance)) * 100
            except Exception as e:
                logging.error(f"Peak FastDTW error: {e}")
        logging.info(f"Peak similarity: {peak_similarity:.2f}%")

        # Enhanced normalization with dynamic ranges
        def normalize_diff(val1, val2, min_range=1e-6, max_range=None):
            if val1 == val2 == 0:
                return 100.0
            if max_range is None:
                max_range = max(abs(val1), abs(val2), min_range)
            diff = abs(val1 - val2)
            return max(0, 1 - (diff / max(max_range, min_range))) * 100

        # Define dynamic ranges based on typical values
        ranges = {
            'tempo': 200.0,  # BPM range
            'spectral_centroid': 5000.0,  # Hz
            'spectral_rolloff': 5000.0,  # Hz
            'spectral_bandwidth': 5000.0,  # Hz
            'spectral_contrast': 50.0,  # dB
            'spectral_flatness': 1.0,  # 0 to 1
            'rms': 1.0,  # Normalized amplitude
            'rms_var': 0.1,  # Variance of RMS
            'onset_count': max(data1['onset_count'], data2['onset_count'], 100),  # Dynamic
            'rhythmic_complexity': 10.0,  # Per second
            'zcr': 0.5,  # 0 to 0.5 typically
            'segment_count': 100.0,  # Max segments
            'spectral_entropy': 10.0  # Bits
        }

        # Compare features with dynamic ranges
        tempo_similarity = normalize_diff(data1['tempo'], data2['tempo'], max_range=ranges['tempo'])
        centroid_similarity = normalize_diff(data1['spectral_centroid'], data2['spectral_centroid'], max_range=ranges['spectral_centroid'])
        rolloff_similarity = normalize_diff(data1['spectral_rolloff'], data2['spectral_rolloff'], max_range=ranges['spectral_rolloff'])
        bandwidth_similarity = normalize_diff(data1['spectral_bandwidth'], data2['spectral_bandwidth'], max_range=ranges['spectral_bandwidth'])
        contrast_similarity = normalize_diff(data1['spectral_contrast'], data2['spectral_contrast'], max_range=ranges['spectral_contrast'])
        flatness_similarity = normalize_diff(data1['spectral_flatness'], data2['spectral_flatness'], max_range=ranges['spectral_flatness'])
        rms_similarity = normalize_diff(data1['rms'], data2['rms'], max_range=ranges['rms'])
        rms_var_similarity = normalize_diff(data1['rms_var'], data2['rms_var'], max_range=ranges['rms_var'])
        onset_similarity = normalize_diff(data1['onset_count'], data2['onset_count'], max_range=ranges['onset_count'])
        rhythmic_similarity = normalize_diff(data1['rhythmic_complexity'], data2['rhythmic_complexity'], max_range=ranges['rhythmic_complexity'])
        zcr_similarity = normalize_diff(data1['zcr'], data2['zcr'], max_range=ranges['zcr'])
        segment_similarity = normalize_diff(data1['segment_count'], data2['segment_count'], max_range=ranges['segment_count'])
        entropy_similarity = normalize_diff(data1['spectral_entropy'], data2['spectral_entropy'], max_range=ranges['spectral_entropy'])

        # MFCC comparison with improved handling
        segment_mfccs1 = data1.get('segment_mfccs', [data1['mfcc']])[:50]
        segment_mfccs2 = data2.get('segment_mfccs', [data2['mfcc']])[:50]
        segment_mfcc_similarity = 0
        if segment_mfccs1 and segment_mfccs2 and len(segment_mfccs1) > 0 and len(segment_mfccs2) > 0:
            try:
                distances = []
                for i in range(min(len(segment_mfccs1), len(segment_mfccs2))):
                    mfcc1 = np.array(segment_mfccs1[i])
                    mfcc2 = np.array(segment_mfccs2[i])
                    if len(mfcc1) > 0 and len(mfcc2) > 0:
                        if len(mfcc1) != len(mfcc2):
                            min_len = min(len(mfcc1), len(mfcc2))
                            mfcc1 = mfcc1[:min_len]
                            mfcc2 = mfcc2[:min_len]
                        distance, _ = fastdtw(mfcc1, mfcc2, dist=euclidean)
                        norm_factor = np.max([np.std(mfcc1), np.std(mfcc2), 1e-6])
                        distances.append(distance / norm_factor)
                segment_mfcc_similarity = max(0, 1 - np.mean(distances)) * 100 if distances else 0
            except Exception as e:
                logging.warning(f"MFCC FastDTW error: {e}")
        logging.info(f"MFCC similarity: {segment_mfcc_similarity:.2f}%")

        # Chroma comparison with improved handling
        segment_chromas1 = data1.get('segment_chromas', [data1['chroma']])[:50]
        segment_chromas2 = data2.get('segment_chromas', [data2['chroma']])[:50]
        segment_chroma_similarity = 0
        if segment_chromas1 and segment_chromas2 and len(segment_chromas1) > 0 and len(segment_chromas2) > 0:
            try:
                distances = []
                for i in range(min(len(segment_chromas1), len(segment_chromas2))):
                    chroma1 = np.array(segment_chromas1[i])
                    chroma2 = np.array(segment_chromas2[i])
                    if len(chroma1) > 0 and len(chroma2) > 0:
                        if len(chroma1) != len(chroma2):
                            min_len = min(len(chroma1), len(chroma2))
                            chroma1 = chroma1[:min_len]
                            chroma2 = chroma2[:min_len]
                        distance, _ = fastdtw(chroma1, chroma2, dist=euclidean)
                        norm_factor = np.max([np.std(chroma1), np.std(chroma2), 1e-6])
                        distances.append(distance / norm_factor)
                segment_chroma_similarity = max(0, 1 - np.mean(distances)) * 100 if distances else 0
            except Exception as e:
                logging.warning(f"Chroma FastDTW error: {e}")
        logging.info(f"Chroma similarity: {segment_chroma_similarity:.2f}%")

        # Adjusted weights with emphasis on segmental features
        weights = {
            'peak': 0.15,
            'tempo': 0.10,
            'centroid': 0.05,
            'rolloff': 0.05,
            'bandwidth': 0.05,
            'contrast': 0.05,
            'flatness': 0.05,
            'segment_mfcc': 0.20,
            'segment_chroma': 0.15,
            'rms': 0.05,
            'rms_var': 0.03,
            'onset': 0.03,
            'rhythmic': 0.02,
            'zcr': 0.02,
            'segment': 0.01,
            'entropy': 0.05
        }

        overall_similarity = (
            weights['peak'] * peak_similarity +
            weights['tempo'] * tempo_similarity +
            weights['centroid'] * centroid_similarity +
            weights['rolloff'] * rolloff_similarity +
            weights['bandwidth'] * bandwidth_similarity +
            weights['contrast'] * contrast_similarity +
            weights['flatness'] * flatness_similarity +
            weights['segment_mfcc'] * segment_mfcc_similarity +
            weights['segment_chroma'] * segment_chroma_similarity +
            weights['rms'] * rms_similarity +
            weights['rms_var'] * rms_var_similarity +
            weights['onset'] * onset_similarity +
            weights['rhythmic'] * rhythmic_similarity +
            weights['zcr'] * zcr_similarity +
            weights['segment'] * segment_similarity +
            weights['entropy'] * entropy_similarity
        )
        logging.info(f"Overall similarity: {overall_similarity:.2f}%")

        # Detailed results
        details = {
            'Spectral Peaks': peak_similarity,
            'Tempo': tempo_similarity,
            'Spectral Centroid': centroid_similarity,
            'Spectral Rolloff': rolloff_similarity,
            'Spectral Bandwidth': bandwidth_similarity,
            'Spectral Contrast': contrast_similarity,
            'Spectral Flatness': flatness_similarity,
            'Segment MFCC': segment_mfcc_similarity,
            'Segment Chroma': segment_chroma_similarity,
            'RMS': rms_similarity,
            'RMS Variance': rms_var_similarity,
            'Onset Count': onset_similarity,
            'Rhythmic Complexity': rhythmic_similarity,
            'Zero Crossing Rate': zcr_similarity,
            'Segment Count': segment_similarity,
            'Spectral Entropy': entropy_similarity
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

        return {
            'overall_similarity': overall_similarity,
            'details': details,
            'cached': False
        }
    except Exception as e:
        logging.error(f"Fingerprint comparison error: {str(e)}")
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
                    'fingerprint_count': len(fingerprints['peaks']),
                    'scatter_plot_path': song.scatter_plot_path if hasattr(song, 'scatter_plot_path') and song.scatter_plot_path else None
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