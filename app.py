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

with app.app_context():
    db.create_all()  # Создаём таблицы при запуске приложения

# Инициализация Spotify OAuth
sp_oauth = SpotifyOAuth(
    client_id=app.config['SPOTIPY_CLIENT_ID'],
    client_secret=app.config['SPOTIPY_CLIENT_SECRET'],
    redirect_uri=app.config['SPOTIPY_REDIRECT_URI'],
    scope='user-library-read user-read-private playlist-read-private playlist-read-collaborative'
)

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

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)