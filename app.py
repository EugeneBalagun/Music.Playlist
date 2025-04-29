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
from spotdl import Spotdl
import librosa
import librosa.display
import yt_dlp
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from flask import send_file
from markupsafe import Markup
from os.path import basename
import soundfile as sf
import scipy.signal

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

# Фильтр для Jinja
app.jinja_env.filters['basename'] = basename

with app.app_context():
    db.create_all()  # Создаём таблицы при запуске приложения

# Инициализация Spotify OAuth
sp_oauth = SpotifyOAuth(
    client_id=app.config['SPOTIPY_CLIENT_ID'],
    client_secret=app.config['SPOTIPY_CLIENT_SECRET'],
    redirect_uri=app.config['SPOTIPY_REDIRECT_URI'],
    scope='user-library-read user-read-private playlist-read-private playlist-read-collaborative'
)

# Создаём директорию для хранения скачанных файлов
DOWNLOAD_DIR = os.path.join(os.getcwd(), 'downloads')
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Создаём директорию для хранения результатов анализа
ANALYSIS_DIR = os.path.join(os.getcwd(), 'analysis')
os.makedirs(ANALYSIS_DIR, exist_ok=True)

PITCH_SHIFT_DIR = os.path.join(os.getcwd(), 'pitch_shifted')
os.makedirs(PITCH_SHIFT_DIR, exist_ok=True)

# Параметры для вашего метода
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
            return tags[:3]  # Возвращаем первые 3 тега (обычно это жанры)
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
        current_user.spotify_token = token_info['refresh_token']  # Сохраняем refresh_token
        db.session.commit()
        flash("Вы успешно авторизованы через Spotify!")

        # Проверяем, есть ли сохранённый URL плейлиста для импорта
        if 'pending_playlist_url' in session:
            playlist_url = session.pop('pending_playlist_url')  # Извлекаем и удаляем из сессии
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
            current_user.spotify_token = token_info['refresh_token']  # Обновляем refresh_token
            db.session.commit()
            logging.debug("Токен успешно обновлён")
            return spotipy.Spotify(auth=token_info['access_token'])
        except Exception as e:
            logging.error(f"Ошибка обновления токена: {e}")
            current_user.spotify_token = None  # Очищаем недействительный токен
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
        return redirect(url_for('login_spotify'))  # Перенаправляем на авторизацию

    try:
        parsed_url = urlparse(song_url)
        track_id = parsed_url.path.split('/')[-1].split('?')[0]
        track = sp.track(track_id)
        artist_name = track['artists'][0]['name']  # Получаем имя исполнителя

        # Получаем жанры из Last.fm
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
        print(f"Песня найдена: {song.name}")
        print(f"ID пользователя, которому принадлежит плейлист: {song.playlist.user_id}")
        print(f"ID текущего пользователя: {current_user.id}")

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
        flash("Будь ласка, введіть посилання на плейлист або альбом Spotify.")
        return redirect(url_for('playlists_page'))

    sp = get_spotify_client()
    if not sp:
        session['pending_playlist_url'] = playlist_url  # Сохраняем URL для импорта после авторизации
        return redirect(url_for('login_spotify'))  # Перенаправляем на авторизацию

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

            # Получаем жанры из Last.fm
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
        # Получаем метаданные трека
        track = sp.track(song.spotify_id)
        track_name = track['name']
        artist_name = track['artists'][0]['name']
        query = f"{track_name} {artist_name}"

        # Настройки для yt-dlp
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

        # Скачивание с YouTube
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"ytsearch:{query}"])

        # Проверяем, был ли файл успешно скачан
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

@app.route('/analyze_song/<int:song_id>', methods=['POST'])
@login_required
def analyze_song(song_id):
    song = Song.query.get_or_404(song_id)
    if not song.file_path:
        flash('Песня не скачана.')
        return redirect(url_for('playlists_page'))

    # Проверяем, есть ли кэшированные результаты анализа
    if song.analysis_report_path and os.path.exists(song.analysis_report_path):
        flash(f'Анализ для песни "{song.name}" уже выполнен.')
        return redirect(url_for('view_analysis', song_id=song.id))

    try:
        analysis = analyze_song(song.file_path, song)
        flash(f'Анализ песни "{song.name}" успешно выполнен.')
        return redirect(url_for('view_analysis', song_id=song.id))
    except Exception as e:
        logging.error(f"Ошибка анализа: {e}")
        flash(f'Не удалось проанализировать песню: {str(e)}')
        return redirect(url_for('playlists_page'))

# Функция анализа песни
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
        spectrogram_path = os.path.join(ANALYSIS_DIR, f'spectrogram_{song.id}_{timestamp}.png')
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max),
                                 sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram: {song.name}')
        plt.savefig(spectrogram_path)
        plt.close()

        chromagram_path = os.path.join(ANALYSIS_DIR, f'chromagram_{song.id}_{timestamp}.png')
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr)
        plt.colorbar()
        plt.title(f'Chromagram: {song.name}')
        plt.savefig(chromagram_path)
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

        song.tempo = tempo
        song.duration = duration
        song.spectral_centroid = spectral_centroid
        song.onset_count = onset_count
        song.analysis_report_path = report_path
        song.spectrogram_path = spectrogram_path
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
            'report_path': report_path
        }

    except Exception as e:
        logging.error(f"Ошибка анализа: {e}")
        raise

@app.route('/analysis/<int:song_id>')
@login_required
def view_analysis(song_id):
    song = Song.query.get_or_404(song_id)
    if song.playlist.user_id != current_user.id:
        flash('Вы не можете просматривать анализ этой песни.')
        return redirect(url_for('playlists_page'))

    if not song.analysis_report_path or not os.path.exists(song.analysis_report_path):
        flash('Анализ для этой песни не найден.')
        return redirect(url_for('playlists_page'))

    # Читаем текстовый отчёт
    with open(song.analysis_report_path, 'r', encoding='utf-8') as f:
        report_content = f.read()

    # Собираем пути к изображениям
    images = []
    if song.spectrogram_path and os.path.exists(song.spectrogram_path):
        images.append({'path': song.spectrogram_path, 'title': 'Спектрограмма'})
    if os.path.exists(song.analysis_report_path.replace('report', 'chromagram')):
        chromagram_path = song.analysis_report_path.replace('report', 'chromagram')
        images.append({'path': chromagram_path, 'title': 'Хромаграмма'})

    return render_template('analysis.html', song=song, report_content=report_content, images=images)

@app.route('/analysis_file/<filename>')
@login_required
def serve_analysis_file(filename):
    file_path = os.path.join(ANALYSIS_DIR, filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    flash('Файл анализа не найден.')
    return redirect(url_for('playlists_page'))

# Функция для питч-шифтинга
def pitch_shift_song(file_path, song, semitones, method='standard'):
    try:
        # Загружаем аудиофайл
        y, sr = librosa.load(file_path, sr=44100)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Выбираем параметры метода
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

        # Питч-шифтинг
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_steps)

        # Сохраняем результат
        output_path = os.path.join(PITCH_SHIFT_DIR, f'{output_prefix}.mp3')
        sf.write(output_path, y_shifted, sr)
        logging.debug(f"Saved pitch-shifted file: {output_path}, exists: {os.path.exists(output_path)}")

        # Спектрограмма
        spectrogram_path = os.path.join(PITCH_SHIFT_DIR, f'spectrogram_{output_prefix}.png')
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y_shifted)), ref=np.max),
                                 sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram: {song.name} ({method}, {semitones} semitones)')
        plt.savefig(spectrogram_path)
        plt.close()
        logging.debug(f"Saved spectrogram: {spectrogram_path}, exists: {os.path.exists(spectrogram_path)}")

        # Проверка существования файлов
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Аудиофайл не сохранён: {output_path}")
        if not os.path.exists(spectrogram_path):
            raise FileNotFoundError(f"Спектрограмма не сохранена: {spectrogram_path}")

        return {
            'output_path': output_path,
            'spectrogram_path': spectrogram_path
        }

    except Exception as e:
        logging.error(f"Ошибка питч-шифтинга ({method}): {e}")
        raise

# Маршрут для питч-шифтинга
@app.route('/pitch_shift/<int:song_id>', methods=['POST'])
@login_required
def pitch_shift(song_id):
    song = Song.query.get_or_404(song_id)
    if not song.file_path:
        flash('Песня не скачана.')
        return redirect(url_for('playlists_page'))

    semitones = request.form.get('semitones')

    try:
        semitones = float(semitones)

        # Выполняем питч-шифтинг для обоих методов
        success = True
        if not (song.pitch_shifted_standard_path and os.path.exists(song.pitch_shifted_standard_path)):
            result_standard = pitch_shift_song(song.file_path, song, semitones, 'standard')
            song.pitch_shifted_standard_path = result_standard['output_path']
        else:
            logging.debug(f"Standard pitch-shift already exists: {song.pitch_shifted_standard_path}")

        if not (song.pitch_shifted_custom_path and os.path.exists(song.pitch_shifted_custom_path)):
            result_custom = pitch_shift_song(song.file_path, song, semitones, 'custom')
            song.pitch_shifted_custom_path = result_custom['output_path']
        else:
            logging.debug(f"Custom pitch-shift already exists: {song.pitch_shifted_custom_path}")

        # Проверяем, что оба результата существуют
        if not (os.path.exists(song.pitch_shifted_standard_path) and os.path.exists(song.pitch_shifted_custom_path)):
            success = False
            flash('Ошибка: Не удалось сохранить результаты питч-шифтинга.')
            logging.error(f"Missing files: standard={song.pitch_shifted_standard_path}, custom={song.pitch_shifted_custom_path}")
        else:
            db.session.commit()
            flash(f'Питч-шифтинг песни "{song.name}" успешно выполнен для обоих методов.')

        # Перенаправляем на страницу сравнения
        if success:
            return redirect(url_for('pitch_shift_compare', song_id=song.id))
        else:
            return redirect(url_for('playlists_page'))

    except Exception as e:
        logging.error(f"Ошибка питч-шифтинга: {e}")
        flash(f'Не удалось выполнить питч-шифтинг: {str(e)}')
        return redirect(url_for('playlists_page'))

# Маршрут для сравнения результатов питч-шифтинга
@app.route('/pitch_shift_compare/<int:song_id>')
@login_required
def pitch_shift_compare(song_id):
    song = Song.query.get_or_404(song_id)
    if song.playlist.user_id != current_user.id:
        flash('Вы не можете просматривать результаты этой песни.')
        return redirect(url_for('playlists_page'))

    # Собираем данные для отображения
    results = []
    # Оригинальный трек
    if song.file_path and song.spectrogram_path and os.path.exists(song.spectrogram_path):
        results.append({
            'method': 'Original',
            'audio_path': song.file_path,
            'spectrogram_path': song.spectrogram_path
        })
    else:
        logging.debug(f"Original file or spectrogram missing: file={song.file_path}, spectrogram={song.spectrogram_path}")

    # Стандартный метод
    if song.pitch_shifted_standard_path and os.path.exists(song.pitch_shifted_standard_path):
        standard_spectrogram = os.path.join(PITCH_SHIFT_DIR, f'spectrogram_{os.path.basename(song.pitch_shifted_standard_path).replace(".mp3", ".png")}')
        if not os.path.exists(standard_spectrogram) and os.path.exists(song.pitch_shifted_standard_path):
            # Пересоздаём спектрограмму, если она отсутствует
            try:
                y_shifted, sr = librosa.load(song.pitch_shifted_standard_path, sr=44100)
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y_shifted)), ref=np.max),
                                         sr=sr, x_axis='time', y_axis='log')
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'Spectrogram: {song.name} (Standard)')
                plt.savefig(standard_spectrogram)
                plt.close()
                logging.debug(f"Recreated spectrogram: {standard_spectrogram}")
            except Exception as e:
                logging.error(f"Ошибка пересоздания спектрограммы (Standard): {e}")
        if os.path.exists(standard_spectrogram):
            results.append({
                'method': 'Standard',
                'audio_path': song.pitch_shifted_standard_path,
                'spectrogram_path': standard_spectrogram
            })
        else:
            logging.warning(f"Standard spectrogram still missing: {standard_spectrogram}")

    # Ваш метод
    if song.pitch_shifted_custom_path and os.path.exists(song.pitch_shifted_custom_path):
        custom_spectrogram = os.path.join(PITCH_SHIFT_DIR, f'spectrogram_{os.path.basename(song.pitch_shifted_custom_path).replace(".mp3", ".png")}')
        if not os.path.exists(custom_spectrogram) and os.path.exists(song.pitch_shifted_custom_path):
            # Пересоздаём спектрограмму, если она отсутствует
            try:
                y_shifted, sr = librosa.load(song.pitch_shifted_custom_path, sr=44100)
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y_shifted)), ref=np.max),
                                         sr=sr, x_axis='time', y_axis='log')
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'Spectrogram: {song.name} (Custom)')
                plt.savefig(custom_spectrogram)
                plt.close()
                logging.debug(f"Recreated spectrogram: {custom_spectrogram}")
            except Exception as e:
                logging.error(f"Ошибка пересоздания спектрограммы (Custom): {e}")
        if os.path.exists(custom_spectrogram):
            results.append({
                'method': 'Custom',
                'audio_path': song.pitch_shifted_custom_path,
                'spectrogram_path': custom_spectrogram
            })
        else:
            logging.warning(f"Custom spectrogram still missing: {custom_spectrogram}")

    # Метрики
    metrics = {}
    if results:
        y_orig, sr = librosa.load(song.file_path, sr=44100)
        mfcc_orig = librosa.feature.mfcc(y=y_orig, sr=sr, n_mfcc=13).mean(axis=1)
        spectral_centroid_orig = librosa.feature.spectral_centroid(y=y_orig, sr=sr).mean()
        rms_orig = librosa.feature.rms(y=y_orig).mean()

        for result in results:
            if result['method'] == 'Original':
                metrics['Original'] = {
                    'mfcc_correlation': 1.0,
                    'spectral_centroid': spectral_centroid_orig,
                    'rms': rms_orig
                }
                continue
            try:
                y_shifted, _ = librosa.load(result['audio_path'], sr=44100)
                mfcc_shifted = librosa.feature.mfcc(y=y_shifted, sr=sr, n_mfcc=13).mean(axis=1)
                spectral_centroid_shifted = librosa.feature.spectral_centroid(y=y_shifted, sr=sr).mean()
                rms_shifted = librosa.feature.rms(y=y_shifted).mean()
                correlation = np.corrcoef(mfcc_orig, mfcc_shifted)[0, 1]
                metrics[result['method']] = {
                    'mfcc_correlation': correlation,
                    'spectral_centroid_diff': abs(spectral_centroid_orig - spectral_centroid_shifted),
                    'rms_diff': abs(rms_orig - rms_shifted)
                }
            except Exception as e:
                logging.error(f"Ошибка вычисления метрик для {result['method']}: {e}")
                metrics[result['method']] = {
                    'mfcc_correlation': 'Ошибка',
                    'spectral_centroid_diff': 'Ошибка',
                    'rms_diff': 'Ошибка'
                }

    logging.debug(f"Compare results: {results}")
    if not results:
        flash('Результаты питч-шифтинга отсутствуют. Выполните питч-шифтинг.')

    return render_template('pitch_shift_compare.html', song=song, results=results, metrics=metrics)

# Маршрут для отдачи файлов
@app.route('/pitch_shift_file/<filename>')
@login_required
def serve_pitch_shift_file(filename):
    file_path = os.path.join(PITCH_SHIFT_DIR, filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    flash('Файл не найден.')
    return redirect(url_for('playlists_page'))


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)