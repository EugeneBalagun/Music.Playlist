import json
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash
from flask_bcrypt import Bcrypt
from datetime import datetime
from sqlalchemy.orm import relationship
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from urllib.parse import urlparse
from flask_migrate import Migrate
import requests
import base64
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# Инициализация приложения
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SPOTIPY_CLIENT_ID'] = '79357f9118c243568eb3847e9a6baaa9'
app.config['SPOTIPY_CLIENT_SECRET'] = '7a301db0ee9041a7ac8685b5f6613a70'
app.config['SPOTIPY_REDIRECT_URI'] = 'http://localhost:5000/callback'
bcrypt = Bcrypt(app)
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
migrate = Migrate(app, db)

with app.app_context():
    db.create_all()  # Создаём таблицы при запуске приложения



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
    songs = db.relationship('Song', backref='playlist', lazy=True)


class Song(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    url = db.Column(db.String(255), nullable=False)
    playlist_id = db.Column(db.Integer, db.ForeignKey('playlists.id'), nullable=False)
    spotify_id = db.Column(db.String(255))
    rating = db.Column(db.Integer, nullable=True)
    notes = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f'<Song {self.name}>'




# Инициализация Spotify OAuth
sp_oauth = SpotifyOAuth(
    client_id=app.config['SPOTIPY_CLIENT_ID'],
    client_secret=app.config['SPOTIPY_CLIENT_SECRET'],
    redirect_uri=app.config['SPOTIPY_REDIRECT_URI'],
    scope='user-library-read user-read-private'
)


@app.route('/login_spotify')
@login_required
def login_spotify():
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)


@app.route('/logout_spotify')
@login_required
def logout_spotify():
    # Удаление токена доступа из данных пользователя
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
        current_user.spotify_token = token_info['access_token']
        db.session.commit()
        flash("Вы успешно авторизованы через Spotify!")
    except Exception as e:
        logging.error(f'Error in callback: {e}')
        flash('Ошибка при авторизации через Spotify.')
    return redirect(url_for('home'))



def get_spotify_client():
    token_info = sp_oauth.get_cached_token()
    if not token_info:
        if current_user.spotify_token:
            try:
                token_info = sp_oauth.refresh_access_token(current_user.spotify_token)
            except Exception as e:
                logging.error(f"Ошибка обновления токена: {e}")
                return None
    return spotipy.Spotify(auth=token_info['access_token']) if token_info else None



# Загрузка пользователя для Flask-Login
@login_manager.user_loader
def load_user(user_id):
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
        flash('Ошибка при подключении к Spotify.')
        return redirect(url_for('playlists_page'))

    try:
        parsed_url = urlparse(song_url)
        track_id = parsed_url.path.split('/')[-1]
        track = sp.track(track_id)

        new_song = Song(
            name=track['name'],
            url=track['external_urls']['spotify'],
            spotify_id=track_id,
            playlist_id=playlist_id
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
    # Отображаем только плейлисты текущего пользователя
    playlists = Playlist.query.filter_by(user_id=current_user.id).all()
    return render_template('playlists.html', playlists=playlists)

@app.route('/add_note/<int:song_id>', methods=['POST'])
@login_required
def add_note(song_id):
    note = request.form.get('note')  # Получаем текст заметки
    song = Song.query.get_or_404(song_id)

    if note:
        song.notes = note  # Обновляем заметку
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
        if playlist_name:
            new_playlist = Playlist(name=playlist_name, user_id=current_user.id)
            db.session.add(new_playlist)
            db.session.commit()
            flash('Плейлист успешно добавлен!')
        else:
            flash('Пожалуйста, введите название плейлиста.')
        return redirect(url_for('playlists_page'))  # Перенаправляем на страницу с плейлистами

    return render_template('add_playlist.html')  # Отображаем форму для добавления плейлиста


# Главная страница (переход на неё возможен только после логина)
@app.route('/')
@login_required
def home():
    playlists = Playlist.query.filter_by(user_id=current_user.id).all()

    # Проверка токена Spotify
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
            if new_name and new_name != playlist.name:
                playlist.name = new_name
                db.session.commit()
                flash(f"Плейлист '{new_name}' успешно обновлён!")
            else:
                flash("Имя плейлиста не изменилось.")
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
            # Сохраняем оценку
            song.rating = int(rating)
            db.session.commit()
            flash(f'Оценка для песни "{song.name}" успешно обновлена!')
        except ValueError:
            flash('Ошибка: оценка должна быть числом от 1 до 10.')
    else:
        flash('Ошибка: песня не найдена.')

    return redirect(url_for('playlists_page'))  # Возвращаем пользователя на страницу с плейлистами






if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Создание таблиц в базе данных
    app.run(debug=True)
