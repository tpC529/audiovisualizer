import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QSlider,
                             QFileDialog, QVBoxLayout, QWidget, QLabel)
from PyQt6.QtGui import QAction
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtCore import QUrl, Qt, QTimer
from moviepy import AudioFileClip, VideoClip

class CenteredScrollingPlayer(QMainWindow):
    def __init__(self, samples=None, sr=None):
        super().__init__()
        self.setWindowTitle("Centered Scrolling Waveform Player")
        self.setGeometry(100, 100, 1400, 800)

        self.window_sec = 10.0  # Total visible window (±5 seconds)
        self.half_window = self.window_sec / 2

        self.player = QMediaPlayer()
        # audio output is required in Qt6 to actually hear audio
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        # sensible default volume (0.0 - 1.0)
        try:
            self.audio_output.setVolume(0.8)
        except Exception:
            pass

        # UI
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setSpacing(5)
        layout.setContentsMargins(10, 10, 10, 10)

        self.canvas = FigureCanvas(plt.Figure(figsize=(14, 8)))
        layout.addWidget(self.canvas, stretch=1)
        # update background placement when canvas resizes
        self.canvas.mpl_connect('resize_event', lambda evt: self._place_background() if getattr(self, 'bg_pil', None) is not None else None)

        controls = QWidget()
        ctrl_layout = QVBoxLayout(controls)
        self.play_btn = QPushButton("Play/Pause (Space)")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.export_btn = QPushButton("Export Video")
        self.status = QLabel("Load an audio file")

        ctrl_layout.addWidget(self.play_btn)
        ctrl_layout.addWidget(self.slider)
        ctrl_layout.addWidget(self.export_btn)
        # allow loading an image after audio is loaded
        self.load_image_btn = QPushButton("Load Image (PNG/JPEG)")
        ctrl_layout.addWidget(self.load_image_btn)
        ctrl_layout.addWidget(self.status)
        layout.addWidget(controls)

        self.setCentralWidget(central)

        # Plot setup (x-axis starts at 0)
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.set_facecolor('black')
        self.line, = self.ax.plot([], [], color='cyan', lw=1.5)
        self.playhead = self.ax.axvline(0, color='red', lw=2, ls='--')
        # show from 0..window_sec initially (cursor at start)
        self.ax.set_xlim(0, self.window_sec)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_xlabel("Time (s)", color='white')
        self.ax.tick_params(colors='white')
        self.ax.grid(alpha=0.2)
        self.canvas.figure.tight_layout()

        # Data
        self.samples = samples
        self.sr = sr
        self.bg_image = None

        # Connections
        self.play_btn.clicked.connect(self.toggle_play)
        self.export_btn.clicked.connect(self.export_video)
        self.load_image_btn.clicked.connect(self.load_image)
        self.slider.sliderMoved.connect(lambda p: self.player.setPosition(p))
        self.player.positionChanged.connect(self.update_plot)
        self.player.durationChanged.connect(lambda d: self.slider.setMaximum(d))

        # Menu & shortcuts
        load_act = QAction("Load Audio", self)
        load_act.setShortcut("Ctrl+O")
        load_act.triggered.connect(self.load_file)
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction(load_act)
        self.play_btn.setShortcut("Space")

        # If samples were provided at construction, initialize slider and plot
        if self.samples is not None and self.sr is not None:
            duration_ms = int(len(self.samples) / self.sr * 1000)
            self.slider.setMinimum(0)
            self.slider.setMaximum(duration_ms)
            self.slider.setValue(0)
            self.player.setPosition(0)
            self.update_plot(0)

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Audio", "", "Audio Files (*.*)")
        if not path:
            return

        self.status.setText("Loading...")
        samples, sr = self.process_audio(path)
        if samples is None:
            self.status.setText(f"Failed to load: {path}")
            return

        self.samples = samples
        self.sr = float(sr)

        self.player.setSource(QUrl.fromLocalFile(path))
        # set slider range based on duration and show initial waveform
        duration_ms = int(len(self.samples) / self.sr * 1000)
        self.slider.setMinimum(0)
        self.slider.setMaximum(duration_ms)
        self.slider.setValue(0)
        self.player.setPosition(0)
        self.status.setText(path)
        self.update_plot(0)

    def prompt_load(self):
        # Called after the window shows to ask the user for a file
        self.load_file()

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        try:
            self.apply_theme_from_image(path)
            self.status.setText(f"Image applied: {path}")
        except Exception as e:
            self.status.setText(f"Failed to apply image: {e}")

    def apply_theme_from_image(self, img_path):
        # compute dominant color
        img = Image.open(img_path).convert('RGBA')
        small = img.resize((64, 64))
        result = small.convert('RGB').getcolors(64*64)
        if not result:
            return
        result.sort(key=lambda x: x[0], reverse=True)
        dominant = result[0][1]
        dom_rgb = tuple([c/255.0 for c in dominant])
        # compute audio brightness factor from RMS
        rms = 0.0
        if self.samples is not None:
            rms = float(np.sqrt(np.mean(self.samples.astype(np.float64)**2)))
        bright = 0.6 + min(1.0, rms*5.0)

        # set waveform color influenced by dominant color and audio brightness
        wave_rgb = tuple(min(1.0, c * bright) for c in dom_rgb)
        bg_lum = 0.2126*dom_rgb[0] + 0.7152*dom_rgb[1] + 0.0722*dom_rgb[2]
        if bg_lum < 0.5:
            bg_color = (1,1,1)
            text_color = 'black'
        else:
            bg_color = (0,0,0)
            text_color = 'white'

        self.ax.set_facecolor(bg_color)
        self.canvas.figure.set_facecolor(bg_color)
        self.ax.tick_params(colors=text_color)
        self.ax.xaxis.label.set_color(text_color)
        self.line.set_color(wave_rgb)
        self.playhead.set_color('red')
        self.canvas.draw_idle()
        # also apply image as background (keeps aspect ratio)
        try:
            self.apply_background_image(img_path)
        except Exception:
            pass

    def apply_background_image(self, img_path):
        img = Image.open(img_path).convert('RGBA')
        arr = np.asarray(img)
        # store PIL image and array for later placement/resizing
        self.bg_pil = img
        self.bg_arr = arr
        # remove existing bg image if present
        if getattr(self, 'bg_image', None) is not None:
            try:
                self.bg_image.remove()
            except Exception:
                pass
            self.bg_image = None
        # place background according to current axes limits
        self._place_background()

    @staticmethod
    def process_audio(path):
        """Try to load audio with librosa, fallback to moviepy if needed.

        Returns (samples: np.ndarray, sr: int) or (None, None) on failure.
        """
        try:
            y, sr = librosa.load(path, sr=None, mono=True)
            return np.asarray(y, dtype=np.float32), int(sr)
        except Exception:
            # Fallback to moviepy (uses ffmpeg) for formats like m4a
            try:
                clip = AudioFileClip(path)
                sr = int(clip.fps)
                arr = clip.to_soundarray()
                clip.close()
                # convert to mono if stereo
                if arr.ndim == 2:
                    arr = arr.mean(axis=1)
                # moviepy returns float in [-1,1], ensure float32
                return np.asarray(arr, dtype=np.float32), sr
            except Exception:
                return None, None

    def _place_background(self):
        """Place or update the background image to preserve aspect ratio inside axes."""
        if getattr(self, 'bg_pil', None) is None:
            return

        # remove old image artist
        if getattr(self, 'bg_image', None) is not None:
            try:
                self.bg_image.remove()
            except Exception:
                pass
            self.bg_image = None

        # current axes data limits
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        axis_w = x1 - x0
        axis_h = y1 - y0 if (y1 - y0) != 0 else 1.0

        iw, ih = self.bg_pil.size
        img_ratio = iw / ih if ih != 0 else 1.0
        axis_ratio = axis_w / axis_h

        # fit image to axes while preserving aspect ratio
        if img_ratio >= axis_ratio:
            # image is wider (relative); fit width to axis width
            target_w = axis_w
            target_h = axis_w / img_ratio
        else:
            # image is taller; fit height to axis height
            target_h = axis_h
            target_w = axis_h * img_ratio

        x0_img = x0 + (axis_w - target_w) / 2.0
        x1_img = x0_img + target_w
        y0_img = y0 + (axis_h - target_h) / 2.0
        y1_img = y0_img + target_h

        # create image artist behind waveform
        self.bg_image = self.ax.imshow(self.bg_arr, extent=(x0_img, x1_img, y0_img, y1_img), aspect='auto', zorder=0, alpha=0.6)
        self.line.set_zorder(2)
        self.playhead.set_zorder(3)
        self.canvas.draw_idle()

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def update_plot(self, pos_ms):
        if self.samples is None: return
        current_sec = pos_ms / 1000.0

        # sliding window anchored to start (show 0..window or current-centered window when past half)
        start_sec = max(0.0, current_sec - self.half_window)
        end_sec = start_sec + self.window_sec

        start_idx = max(0, int(start_sec * self.sr))
        end_idx = min(len(self.samples), int(end_sec * self.sr))

        visible = self.samples[start_idx:end_idx]
        if len(visible) < 2: return

        # t = absolute time positions for visible samples
        t = np.linspace(start_idx / self.sr, (start_idx + len(visible)) / self.sr, len(visible))
        self.line.set_data(t, visible)

        # update axis limits so the left edge is 0 initially
        self.ax.set_xlim(start_sec, end_sec)

        # update playhead to current absolute time
        try:
            self.playhead.set_xdata((current_sec, current_sec))
        except Exception:
            self.playhead = self.ax.axvline(current_sec, color='red', lw=2, ls='--')

        # reposition background to respect new axis limits (if present)
        if getattr(self, 'bg_pil', None) is not None:
            self._place_background()

        # Dynamic vertical fill
        margin = 0.1
        amp_max = np.max(np.abs(visible))
        if amp_max > 0:
            self.ax.set_ylim(-amp_max * (1 + margin), amp_max * (1 + margin))

        self.slider.blockSignals(True)
        self.slider.setValue(pos_ms)
        self.slider.blockSignals(False)

        self.canvas.draw_idle()

    def make_frame(self, t):
        pos_ms = int(t * 1000)
        self.update_plot(pos_ms)
        img = np.frombuffer(self.canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape(self.canvas.figure.canvas.get_width_height()[::-1] + (3,))
        return img

    def export_video(self):
        if self.samples is None: return
        out_path, _ = QFileDialog.getSaveFileName(self, "Export Video", "", "MP4 (*.mp4)")
        if not out_path: return

        self.status.setText("Exporting video...")
        audio = AudioFileClip(self.player.source().toLocalFile())
        video = VideoClip(self.make_frame, duration=self.player.duration() / 1000)
        video = video.set_audio(audio)
        video.write_videofile(out_path, fps=30, codec='libx264', audio_codec='aac')
        self.status.setText(f"Exported: {out_path}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # prompt for file before creating/showing main window so GUI populates after processing
    path, _ = QFileDialog.getOpenFileName(None, "Open Audio", "", "Audio Files (*.*)")
    if not path:
        sys.exit(0)

    samples, sr = CenteredScrollingPlayer.process_audio(path)
    if samples is None:
        print(f"Failed to load audio: {path}")
        sys.exit(1)

    win = CenteredScrollingPlayer(samples=samples, sr=sr)
    win.player.setSource(QUrl.fromLocalFile(path))
    win.show()
    # Prompt for background image right after showing window
    img_path, _ = QFileDialog.getOpenFileName(None, "Open Image (optional)", "", "Images (*.png *.jpg *.jpeg)")
    if img_path:
        try:
            win.apply_background_image(img_path)
        except Exception as e:
            print(f"Failed to apply background image: {e}")
    sys.exit(app.exec())