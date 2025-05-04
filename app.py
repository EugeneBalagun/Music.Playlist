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
from database import db, User, Playlist, Song
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

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Инициализация приложения
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

# Фильтры для Jinja
app.jinja_env.filters['basename'] = basename


def format_time(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


app.jinja_env.filters['format_time'] = format_time

with app.app_context():
    db.create_all()  # Создаём таблицы при запуске приложения

# Инициализация Spotify OAuth
sp_oauth = SpotifyOAuth(
    client_id=app.config['SPOTIPY_CLIENT_ID'],
    client_secret=app.config['SPOTIPY_CLIENT_SECRET'],
    redirect_uri=app.config['SPOTIPY_REDIRECT_URI'],
    scope='user-library-read user-read-private playlist-read-private playlist-read-collaborative'
)

# Создаём директории
DOWNLOAD_DIR = os.path.join(os.getcwd(), 'downloads')
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
ANALYSIS_DIR = os.path.join(os.getcwd(), 'analysis')
os.makedirs(ANALYSIS_DIR, exist_ok=True)
PITCH_SHIFT_DIR = os.path.join(os.getcwd(), 'pitch_shifted')
os.makedirs(PITCH_SHIFT_DIR, exist_ok=True)
TEMPO_SHIFT_DIR = os.path.join(os.getcwd(), 'tempo_shifted')
os.makedirs(TEMPO_SHIFT_DIR, exist_ok=True)

# Параметры для методов
K_CUSTOM = 22885686008
N_CUSTOM = 39123338641
K_STANDARD = 7
N_STANDARD = 12


# Функция для получения жанров из Last.fm
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
        logging.error(f"Ошибка Last.fm API: {e}")
        return []


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
    flash('Вы успешно вышли из Spotify.')
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
        flash("Вы успешно авторизованы через Spotify!")
        if 'pending_playlist_url' in session:
            playlist_url = session.pop('pending_playlist_url')
            return redirect(url_for('import_spotify_playlist', playlist_url=playlist_url))
    except Exception as e:
        logging.error(f'Error in callback: {e}')
        flash('Ошибка при авторизации через Spotify.')
    return redirect(url_for('home'))


def get_spotify_client():
    if not current_user.is_authenticated:
        logging.debug("Пользователь не авторизован")
        return None
    token_info = sp_oauth.get_cached_token()
    if token_info and not sp_oauth.is_token_expired(token_info):
        logging.debug("Используем токен из кэша")
        return spotipy.Spotify(auth=token_info['access_token'])
    if current_user.spotify_token:
        logging.debug(f"Refresh token из базы: {current_user.spotify_token}")
        try:
            token_info = sp_oauth.refresh_access_token(current_user.spotify_token)
            current_user.spotify_token = token_info['refresh_token']
            db.session.commit()
            logging.debug("Токен успешно обновлён")
            return spotipy.Spotify(auth=token_info['access_token'])
        except Exception as e:
            logging.error(f"Ошибка обновления токена: {e}")
            current_user.spotify_token = None
            db.session.commit()
            flash("Ваш Spotify токен недействителен. Пожалуйста, авторизуйтесь снова.")
            return None
    logging.debug("Токена нет, нужен логин в Spotify")
    flash("Пожалуйста, авторизуйтесь в Spotify для продолжения.")
    return None


@login_manager.user_loader
def user_loader(user_id):
    return User.query.get(int(user_id))


@app.route('/add_song_spotify/<int:playlist_id>', methods=['POST'])
@login_required
def add_song_spotify(playlist_id):
    song_url = request.form.get('song_url')
    if not song_url:
        flash("Введите ссылку на песню Spotify.")
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
        flash(f'Песня "{track["name"]}" успешно добавлена!')
    except Exception as e:
        logging.error(f"Ошибка: {e}")
        flash("Не удалось добавить песню.")
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
        flash(f'Заметка для песни "{song.name}" успешно обновлена!')
    return redirect(url_for('playlists_page'))


def get_song_name_from_url(url):
    sp = get_spotify_client()
    if not sp:
        logging.error("Spotify client не доступен!")
        return None
    try:
        track_id = url.split("/")[-1].split("?")[0]
        track = sp.track(track_id)
        return track["name"]
    except Exception as e:
        logging.error(f"Ошибка при получении данных с Spotify: {e}")
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
            flash('Плейлист успешно добавлен!')
        else:
            flash('Пожалуйста, введите название плейлиста.')
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
        flash('Вы успешно зарегистрировались!')
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
            flash('Неверное имя пользователя или пароль!')
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
                flash(f"Плейлист '{new_name}' успешно обновлён!")
            else:
                flash("Пожалуйста, введите название плейлиста.")
            return redirect(url_for('playlists_page'))
    else:
        flash('Вы не можете редактировать этот плейлист.')
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
            flash('Песня успешно удалена из плейлиста!')
        else:
            flash('Вы не можете удалить эту песню.')
    else:
        flash('Песня не найдена.')
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
        flash('Плейлист успешно удалён!')
    else:
        flash('Вы не можете удалить этот плейлист.')
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
            flash(f'Оценка для песни "{song.name}" успешно обновлена!')
        except ValueError:
            flash('Ошибка: оценка должна быть числом от 1 до 10.')
    else:
        flash('Ошибка: песня не найдена.')
    return redirect(url_for('playlists_page'))


@app.route('/import_spotify_playlist', methods=['POST'])
@login_required
def import_spotify_playlist():
    playlist_url = request.form.get('playlist_url')
    logging.debug(f"Получена ссылка: {playlist_url}")
    if not playlist_url:
        flash("Будь ласка, введіть посилання на плейлист или альбом Spotify.")
        return redirect(url_for('playlists_page'))
    sp = get_spotify_client()
    if not sp:
        session['pending_playlist_url'] = playlist_url
        return redirect(url_for('login_spotify'))
    try:
        parsed_url = urlparse(playlist_url)
        spotify_id = parsed_url.path.split('/')[-1].split('?')[0]
        logging.debug(f"Извлечённый Spotify ID: {spotify_id}")
        if '/playlist/' in parsed_url.path:
            logging.debug("Обрабатываем плейлист")
            data = sp.playlist(spotify_id)
            tracks = data['tracks']['items']
            source_type = "плейлиста"
            description = data.get('description', None)
        elif '/album/' in parsed_url.path:
            logging.debug("Обрабатываем альбом")
            data = sp.album(spotify_id)
            tracks = data['tracks']['items']
            source_type = "альбома"
            description = None
        else:
            flash("Невірне посилання. Використовуйте плейлист або альбом Spotify.")
            return redirect(url_for('playlists_page'))
        logging.debug(f"Отримали {source_type}: {data['name']}")
        new_playlist = Playlist(
            name=data['name'],
            user_id=current_user.id,
            description=description
        )
        db.session.add(new_playlist)
        db.session.commit()
        logging.debug(f"Створено плейлист: {new_playlist.name} (ID: {new_playlist.id})")
        added_count = 0
        for item in tracks:
            track = item['track'] if source_type == "плейлиста" else item
            if not track or 'id' not in track:
                logging.debug("Трек пропущено: немає ID")
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
            logging.debug(f"Додано трек: {track['name']}")
        db.session.commit()
        flash(f'Плейлист "{data["name"]}" успішно імпортовано з {added_count} піснями!')
    except spotipy.SpotifyException as e:
        logging.error(f"Помилка Spotify API: {e}")
        flash(f"Помилка Spotify: {str(e)}")
    except Exception as e:
        logging.error(f"Помилка імпорту: {e}")
        flash("Не вдалося імпортувати плейлист.")
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
        flash('Вы не можете просматривать графики этого плейлиста.')
        return redirect(url_for('playlists_page'))
    return render_template('charts.html', playlist=playlist)


@app.route('/download_song/<int:song_id>', methods=['POST'])
@login_required
def download_song(song_id):
    song = Song.query.get_or_404(song_id)
    if song.playlist.user_id != current_user.id:
        flash('Вы не можете скачать эту песню.')
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
            flash(f'Песня "{track_name}" успешно скачана и сохранена!')
        else:
            flash(f'Не удалось скачать песню "{track_name}".')
            logging.error(f"Файл не найден после скачивания: {file_path}")
    except Exception as e:
        logging.error(f"Ошибка при скачивании: {e}")
        flash(f'Не удалось скачать песню: {str(e)}')
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
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1).tolist()
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        onset_count = len(onsets)
        onset_times = onsets.tolist()
        rms = librosa.feature.rms(y=y).mean()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Генерация спектрограммы, как в PyQt5, с увеличенной шириной
        spectrogram_path = os.path.join(ANALYSIS_DIR, f'spectrogram_{song.id}_{timestamp}.png')
        window_size = 1024
        step_size = 512
        chunk_duration_sec = 4
        spectrogram, time, freq = process_full_audio(y, sr, window_size, step_size, chunk_duration_sec)
        width_pixels = max(5000, int(500 * duration / 10))  # Аналогично PyQt5: 500 пикселей на 10 секунд
        fig_width = width_pixels / 100  # Масштабируем для figsize
        fig = plt.figure(figsize=(fig_width, 6), dpi=100)
        ax = fig.add_subplot(111)
        im = ax.imshow(
            20 * np.log10(spectrogram + 1e-6),
            aspect='auto',
            origin='lower',
            extent=[time[0], time[-1], freq[0], freq[-1]],
            cmap='magma'
        )
        ax.set_xlabel('Время [с]')
        ax.set_ylabel('Частота [Гц]')
        ax.set_title(f'FFT Спектрограмма: {song.name}')
        fig.colorbar(im, ax=ax, label='Амплитуда [dB]')
        fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)

        # Сохранение границ области данных
        ax_pos = ax.get_position()
        data_area = {
            'x0': ax_pos.x0,
            'x1': ax_pos.x1,
            'width': ax_pos.width,
            'pixel_x0': ax_pos.x0 * fig.dpi * fig_width,
            'pixel_width': ax_pos.width * fig.dpi * fig_width
        }
        data_area_path = os.path.join(ANALYSIS_DIR, f'data_area_{song.id}_{timestamp}.json')
        with open(data_area_path, 'w') as f:
            json.dump(data_area, f)

        plt.savefig(spectrogram_path, bbox_inches='tight')
        plt.close(fig)

        # Генерация хромаграммы
        chromagram_path = os.path.join(ANALYSIS_DIR, f'chromagram_{song.id}_{timestamp}.png')
        plt.figure(figsize=(20, 4), dpi=150)
        librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr)
        plt.colorbar()
        plt.title(f'Chromagram: {song.name}')
        plt.savefig(chromagram_path, bbox_inches='tight')
        plt.close()

        report_path = os.path.join(ANALYSIS_DIR, f'report_{song.id}_{timestamp}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Анализ аудиофайла: {os.path.basename(file_path)}\n")
            f.write(f"Длительность: {duration:.2f} сек\n")
            f.write(f"Темп: {tempo:.2f} BPM\n")
            f.write(f"Количество ударов: {len(beat_times)}\n")
            f.write(f"Временные метки ударов: {', '.join([f'{x:.2f}' for x in beat_times[:10]])}...\n")
            f.write(f"Спектральный центроид: {spectral_centroid:.2f} Гц\n")
            f.write(f"Спектральный спад: {spectral_rolloff:.2f} Гц\n")
            f.write(f"Спектральная ширина: {spectral_bandwidth:.2f} Гц\n")
            f.write(f"Средние MFCC: {', '.join([f'{x:.2f}' for x in mfcc_mean])}\n")
            f.write(f"Средняя хромаграмма: {', '.join([f'{x:.2f}' for x in chroma_mean])}\n")
            f.write(f"Количество звуковых событий (onsets): {onset_count}\n")
            f.write(f"Временные метки событий: {', '.join([f'{x:.2f}' for x in onset_times[:10]])}...\n")
            f.write(f"Средняя энергия (RMS): {rms:.4f}\n")
            f.write(f"Путь к спектрограмме: {spectrogram_path}\n")
            f.write(f"Путь к хромаграмме: {chromagram_path}\n")
            f.write(f"Путь к данным области: {data_area_path}\n")

        song.tempo = tempo
        song.duration = duration
        song.spectral_centroid = spectral_centroid
        song.onset_count = onset_count
        song.analysis_report_path = report_path
        song.spectrogram_path = spectrogram_path
        song.chromagram_path = chromagram_path
        db.session.commit()

        summary = (
            f"Темп: {tempo:.2f} BPM, "
            f"Длительность: {duration:.2f} сек, "
            f"Спектральный центроид: {spectral_centroid:.2f} Гц, "
            f"Звуковые события: {onset_count}"
        )
        return {
            'summary': summary,
            'spectrogram_path': spectrogram_path,
            'chromagram_path': chromagram_path,
            'report_path': report_path,
            'report_content': open(report_path, 'r', encoding='utf-8').read(),
            'mfcc': mfcc_mean,
            'spectral_centroid': spectral_centroid,
            'rms': rms,
            'data_area_path': data_area_path,
            'data_area': data_area
        }
    except Exception as e:
        logging.error(f"Ошибка анализа: {e}")
        raise


def pitch_shift_song(file_path, song, semitones, method='standard'):
    try:
        y, sr = librosa.load(file_path, sr=44100)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if method == 'standard':
            K = K_STANDARD
            N = N_STANDARD
            output_prefix = f'standard_{song.id}_{semitones}_{timestamp}'
            pitch_steps = semitones
        else:
            K = K_CUSTOM
            N = N_CUSTOM
            output_prefix = f'custom_{song.id}_{semitones}_{timestamp}'
            pitch_steps = semitones * K / 7 / N
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_steps)
        output_path = os.path.join(PITCH_SHIFT_DIR, f'{output_prefix}.mp3')
        sf.write(output_path, y_shifted, sr)
        logging.debug(f"Saved pitch-shifted file: {output_path}, exists: {os.path.exists(output_path)}")
        spectrogram_path = os.path.join(PITCH_SHIFT_DIR, f'spectrogram_{output_prefix}.png')
        plt.figure(figsize=(20, 4), dpi=150)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y_shifted)), ref=np.max),
                                 sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram: {song.name} ({method}, {semitones} semitones)')
        plt.savefig(spectrogram_path, bbox_inches='tight')
        plt.close()
        chromagram_path = os.path.join(PITCH_SHIFT_DIR, f'chromagram_{output_prefix}.png')
        chroma = librosa.feature.chroma_stft(y=y_shifted, sr=sr)
        plt.figure(figsize=(20, 4), dpi=150)
        librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr)
        plt.colorbar()
        plt.title(f'Chromagram: {song.name} ({method}, {semitones} semitones)')
        plt.savefig(chromagram_path, bbox_inches='tight')
        plt.close()
        logging.debug(f"Saved spectrogram: {spectrogram_path}, exists: {os.path.exists(spectrogram_path)}")
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Аудиофайл не сохранён: {output_path}")
        if not os.path.exists(spectrogram_path):
            raise FileNotFoundError(f"Спектрограмма не сохранена: {spectrogram_path}")
        mfcc = librosa.feature.mfcc(y=y_shifted, sr=sr, n_mfcc=13).mean(axis=1).tolist()
        spectral_centroid = librosa.feature.spectral_centroid(y=y_shifted, sr=sr).mean()
        rms = librosa.feature.rms(y=y_shifted).mean()
        return {
            'output_path': output_path,
            'spectrogram_path': spectrogram_path,
            'chromagram_path': chromagram_path,
            'mfcc': mfcc,
            'spectral_centroid': spectral_centroid,
            'rms': rms
        }
    except Exception as e:
        logging.error(f"Ошибка питч-шифтинга ({method}): {e}")
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
        elif method == 'custom':
            K = K_CUSTOM
            N = N_CUSTOM
            output_prefix = f'custom_tempo_{song.id}_{semitones}_{timestamp}'
            rate = 2 ** (semitones * K / 7 / N)
        else:
            raise ValueError(f"Неизвестный метод: {method}")
        y_shifted = librosa.effects.time_stretch(y, rate=rate)
        output_path = os.path.join(TEMPO_SHIFT_DIR, f'{output_prefix}.mp3')
        sf.write(output_path, y_shifted, sr)
        logging.debug(f"Saved tempo-shifted file: {output_path}, exists: {os.path.exists(output_path)}")
        spectrogram_path = os.path.join(TEMPO_SHIFT_DIR, f'spectrogram_{output_prefix}.png')
        plt.figure(figsize=(20, 4), dpi=150)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y_shifted)), ref=np.max),
                                 sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram: {song.name} ({method}, {semitones} semitones)')
        plt.savefig(spectrogram_path, bbox_inches='tight')
        plt.close()
        chromagram_path = os.path.join(TEMPO_SHIFT_DIR, f'chromagram_{output_prefix}.png')
        chroma = librosa.feature.chroma_stft(y=y_shifted, sr=sr)
        plt.figure(figsize=(20, 4), dpi=150)
        librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr)
        plt.colorbar()
        plt.title(f'Chromagram: {song.name} ({method}, {semitones} semitones)')
        plt.savefig(chromagram_path, bbox_inches='tight')
        plt.close()
        logging.debug(f"Saved spectrogram: {spectrogram_path}, exists: {os.path.exists(spectrogram_path)}")
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Аудиофайл не сохранён: {output_path}")
        if not os.path.exists(spectrogram_path):
            raise FileNotFoundError(f"Спектрограмма не сохранена: {spectrogram_path}")
        mfcc = librosa.feature.mfcc(y=y_shifted, sr=sr, n_mfcc=13).mean(axis=1).tolist()
        spectral_centroid = librosa.feature.spectral_centroid(y=y_shifted, sr=sr).mean()
        rms = librosa.feature.rms(y=y_shifted).mean()
        return {
            'output_path': output_path,
            'spectrogram_path': spectrogram_path,
            'chromagram_path': chromagram_path,
            'mfcc': mfcc,
            'spectral_centroid': spectral_centroid,
            'rms': rms
        }
    except Exception as e:
        logging.error(f"Ошибка изменения темпа ({method}): {e}")
        raise


@app.route('/song_processing/<int:song_id>', methods=['GET', 'POST'])
@login_required
def song_processing(song_id):
    song = Song.query.get_or_404(song_id)
    if song.playlist.user_id != current_user.id:
        flash('Вы не можете обрабатывать эту песню.')
        return redirect(url_for('playlists_page'))

    analysis_result = None
    pitch_shift_results = []
    tempo_shift_results = []
    metrics = {}

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
                    flash(f'Песня "{track_name}" успешно скачана!')
                else:
                    flash(f'Не удалось скачать песню "{track_name}".')
                    return redirect(url_for('song_processing', song_id=song.id))
            except Exception as e:
                logging.error(f"Ошибка при скачивании: {e}")
                flash(f'Не удалось скачать песню: {str(e)}')
                return redirect(url_for('song_processing', song_id=song.id))

        if action == 'analyze' and song.file_path:
            if not (song.analysis_report_path and os.path.exists(song.analysis_report_path)):
                try:
                    analysis_result = analyze_song(song.file_path, song)
                    flash(f'Анализ песни "{song.name}" успешно выполнен.')
                except Exception as e:
                    logging.error(f"Ошибка анализа: {e}")
                    flash(f'Не удалось проанализировать песню: {str(e)}')
                    return redirect(url_for('song_processing', song_id=song.id))
            else:
                analysis_result = {
                    'summary': f"Темп: {song.tempo:.2f} BPM, Длительность: {song.duration:.2f} сек, "
                               f"Спектральный центроид: {song.spectral_centroid:.2f} Гц, Звуковые события: {song.onset_count}",
                    'spectrogram_path': song.spectrogram_path,
                    'chromagram_path': song.chromagram_path,
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

        if action == 'process' and song.file_path:
            pitch_semitones = float(request.form.get('pitch_semitones', 0))
            tempo_semitones = float(request.form.get('tempo_semitones', 0))
            pitch_methods = request.form.getlist('pitch_methods')
            tempo_methods = request.form.getlist('tempo_methods')

            try:
                # Питч-шифтинг
                for method in pitch_methods:
                    if method == 'standard' and not (
                            song.pitch_shifted_standard_path and os.path.exists(song.pitch_shifted_standard_path)):
                        result = pitch_shift_song(song.file_path, song, pitch_semitones, 'standard')
                        song.pitch_shifted_standard_path = result['output_path']
                        pitch_shift_results.append({
                            'method': 'Standard',
                            'audio_path': result['output_path'],
                            'spectrogram_path': result['spectrogram_path'],
                            'chromagram_path': result['chromagram_path'],
                            'mfcc': result['mfcc'],
                            'spectral_centroid': result['spectral_centroid'],
                            'rms': result['rms']
                        })
                    elif method == 'custom' and not (
                            song.pitch_shifted_custom_path and os.path.exists(song.pitch_shifted_custom_path)):
                        result = pitch_shift_song(song.file_path, song, pitch_semitones, 'custom')
                        song.pitch_shifted_custom_path = result['output_path']
                        pitch_shift_results.append({
                            'method': 'Custom',
                            'audio_path': result['output_path'],
                            'spectrogram_path': result['spectrogram_path'],
                            'chromagram_path': result['chromagram_path'],
                            'mfcc': result['mfcc'],
                            'spectral_centroid': result['spectral_centroid'],
                            'rms': result['rms']
                        })

                # Изменение темпа
                for method in tempo_methods:
                    if method == 'standard' and not (
                            song.tempo_shifted_standard_path and os.path.exists(song.tempo_shifted_standard_path)):
                        result = tempo_shift_song(song.file_path, song, tempo_semitones, 'standard')
                        song.tempo_shifted_standard_path = result['output_path']
                        tempo_shift_results.append({
                            'method': 'Standard',
                            'audio_path': result['output_path'],
                            'spectrogram_path': result['spectrogram_path'],
                            'chromagram_path': result['chromagram_path'],
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
                            'chromagram_path': result['chromagram_path'],
                            'mfcc': result['mfcc'],
                            'spectral_centroid': result['spectral_centroid'],
                            'rms': result['rms']
                        })

                db.session.commit()
                flash('Обработка песни успешно выполнена.')

                # Вычисление метрик
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

                    for result in pitch_shift_results + tempo_shift_results:
                        try:
                            correlation = np.corrcoef(mfcc_orig, result['mfcc'])[0, 1]
                            metrics[result['method']] = {
                                'mfcc_correlation': correlation,
                                'spectral_centroid_diff': abs(spectral_centroid_orig - result['spectral_centroid']),
                                'rms_diff': abs(rms_orig - result['rms'])
                            }
                        except Exception as e:
                            logging.error(f"Ошибка вычисления метрик для {result['method']}: {e}")
                            metrics[result['method']] = {
                                'mfcc_correlation': 'Ошибка',
                                'spectral_centroid_diff': 'Ошибка',
                                'rms_diff': 'Ошибка'
                            }

            except Exception as e:
                logging.error(f"Ошибка обработки: {e}")
                flash(f'Не удалось выполнить обработку: {str(e)}')
                return redirect(url_for('song_processing', song_id=song.id))

    # Добавляем оригинальный трек для сравнения
    if song.file_path and (not pitch_shift_results or not tempo_shift_results):
        if song.spectrogram_path and song.chromagram_path and os.path.exists(song.spectrogram_path) and os.path.exists(
                song.chromagram_path):
            pitch_shift_results.append({
                'method': 'Original',
                'audio_path': song.file_path,
                'spectrogram_path': song.spectrogram_path,
                'chromagram_path': song.chromagram_path,
                'mfcc': librosa.feature.mfcc(y=librosa.load(song.file_path, sr=44100)[0], sr=44100, n_mfcc=13).mean(
                    axis=1).tolist(),
                'spectral_centroid': song.spectral_centroid or librosa.feature.spectral_centroid(
                    y=librosa.load(song.file_path, sr=44100)[0], sr=44100).mean(),
                'rms': librosa.feature.rms(y=librosa.load(song.file_path, sr=44100)[0]).mean()
            })
            tempo_shift_results.append({
                'method': 'Original',
                'audio_path': song.file_path,
                'spectrogram_path': song.spectrogram_path,
                'chromagram_path': song.chromagram_path,
                'mfcc': librosa.feature.mfcc(y=librosa.load(song.file_path, sr=44100)[0], sr=44100, n_mfcc=13).mean(
                    axis=1).tolist(),
                'spectral_centroid': song.spectral_centroid or librosa.feature.spectral_centroid(
                    y=librosa.load(song.file_path, sr=44100)[0], sr=44100).mean(),
                'rms': librosa.feature.rms(y=librosa.load(song.file_path, sr=44100)[0]).mean()
            })

    return render_template('song_processing.html',
                           song=song,
                           analysis_result=analysis_result,
                           pitch_shift_results=pitch_shift_results,
                           tempo_shift_results=tempo_shift_results,
                           metrics=metrics)


@app.route('/song_processing_file/<path:filename>')
@login_required
def serve_song_processing_file(filename):
    for directory in [DOWNLOAD_DIR, ANALYSIS_DIR, PITCH_SHIFT_DIR, TEMPO_SHIFT_DIR]:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            return send_file(file_path)
    flash('Файл не найден.')
    return redirect(url_for('playlists_page'))


@app.route('/song_player/<int:song_id>')
@login_required
def song_player(song_id):
    song = Song.query.get_or_404(song_id)
    if song.playlist.user_id != current_user.id:
        flash('Вы не можете просматривать эту песню.')
        return redirect(url_for('playlists_page'))
    if not song.file_path or not os.path.exists(song.file_path):
        flash('Аудиофайл не найден. Пожалуйста, сначала скачайте песню.')
        return redirect(url_for('song_processing', song_id=song.id))
    if not song.spectrogram_path or not os.path.exists(song.spectrogram_path):
        try:
            analysis_result = analyze_song(song.file_path, song)
            song.spectrogram_path = analysis_result['spectrogram_path']
            db.session.commit()
        except Exception as e:
            logging.error(f"Ошибка генерации спектрограммы: {e}")
            flash('Не удалось сгенерировать спектрограмму.')
            return redirect(url_for('song_processing', song_id=song.id))
    # Загружаем границы области данных
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
    freq = np.fft.fftfreq(window_size, d=1/sample_rate)[:window_size // 2]
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
        else:
            full_spectrogram = np.hstack((full_spectrogram, spectrogram))
        if len(full_time) == 0:
            full_time = time + i / sample_rate
        else:
            full_time = np.concatenate((full_time, time + i / sample_rate))
    return full_spectrogram, full_time, freq


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)