import os

from loguru import logger
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl


class VideoPlayer(QMainWindow):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.setWindowTitle('Video Player')
        self.resize(600, 400)
        self.video_widget = QVideoWidget()
        self.setCentralWidget(self.video_widget)
        self.media_player = QMediaPlayer()
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.mediaStatusChanged.connect(self.media_ended)

    def play(self, video_path, local_file=True):
        if local_file:
            path = os.path.dirname(os.path.abspath(__file__))
            video_path = os.path.join(path, '..', '..', video_path)
            media_content = QUrl.fromLocalFile(video_path)
        else:
            media_content = QUrl(video_path)
        self.show()
        self.media_player.setMedia(QMediaContent(media_content))
        self.media_player.setPosition(0)  # start from the beginning of the video
        self.media_player.play()

    def media_ended(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.close()