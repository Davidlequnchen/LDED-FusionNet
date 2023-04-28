import sys
import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QScreen, QGuiApplication
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout
from PyQt5.uic import loadUi
import os
import pyqtgraph as pg
from scipy.io import wavfile
from pydub import AudioSegment
from pydub.playback import play
from PyQt5.QtCore import QThread


# Load the annotations
def load_annotations(file_path):
    annotations = {}
    with open(file_path, 'r') as f:
        for line in f:
            timestamp, class_name, layer_number, _ = line.strip().split('\t')
            annotations[int(float(timestamp) * 10)] = (class_name, layer_number)
    return annotations



class ImageSequencePlayer(QMainWindow):
    def __init__(self, image_pattern, audio_pattern, annotations, total_frames,fps=20, parent=None):
        super().__init__(parent)
        loadUi("multimodal_player.ui", self)  # Load the .ui file

        self.image_pattern = image_pattern
        self.audio_pattern = audio_pattern
        self.annotations = annotations
        self.current_frame = 1
        self.total_frames = total_frames
        self.fps = fps
        self.record = False
        self.video_writer = None


        self.audio_waveform_plot = pg.PlotWidget()
        self.audio_waveform_plot.setBackground('w')  # 'w' stands for white
        self.audio_waveform_plot.setYRange(-4000, 4000) # set the audio range -- denoised; (-7000, 7000) for raw audio
        self.audio_waveform_widget.setLayout(QVBoxLayout())
        self.audio_waveform_widget.layout().addWidget(self.audio_waveform_plot)

        self.progress_slider.valueChanged.connect(self.slider_value_changed)
        self.progress_slider.setMaximum(self.total_frames)

        self.stop_button.clicked.connect(self.stop_button_clicked)
        self.start_button.clicked.connect(self.start_button_clicked)
        self.record_button.clicked.connect(self.toggle_record)

        self.audio_player_threads = []

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)


    def slider_value_changed(self, value):
        self.current_frame = value
        self.update_frame()



    def stop_button_clicked(self):
        self.timer.stop()
        # Stop all currently playing audio
        # for audio_player in self.audio_player_threads:
        #     audio_player.terminate()
        # self.audio_player_threads = []


    def toggle_record(self):
        self.record = not self.record
        if self.record:
            # Start recording
            if self.video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                output_filename = "output_video.avi"
                self.video_writer = cv2.VideoWriter(output_filename, fourcc, self.fps, (self.width(), self.height()))
        else:
            # Stop recording and release the video writer
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None



    def start_button_clicked(self):
        if not self.timer.isActive():
            self.timer.start(1000 / self.fps)
            # Resume audio playback for the current frame
            # audio_file = self.audio_pattern % (self.current_frame)
            # audio_player = AudioPlayer(audio_file)
            # audio_player.start()
            # self.audio_player_threads.append(audio_player)



    def closeEvent(self, event):
        # Stop all audio player threads
        # for audio_player in self.audio_player_threads:
        #     audio_player.terminate()
        # event.accept()
        if self.video_writer:
            self.video_writer.release()



    def update_frame(self):
        class_colors = {
            'Crack': 'rgb(245, 121, 0)',
            'Keyhole pores': 'rgb(239, 41, 41)',
            'Laser-off': 'rgb(46, 52, 54)',
            'Defect-free': 'rgb(114, 159, 207)',
        }
        if self.current_frame < self.total_frames:
            image_path = self.image_pattern % self.current_frame
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Resize the image
            new_width, new_height = 583, 326  # Change these values as needed
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            h, w, _ = image.shape
            qt_image = QImage(image.data, w, h, w * 3, QImage.Format_RGB888)
            pixmap = QPixmap(qt_image)
            self.image_label.setPixmap(pixmap)
            annotation = self.annotations.get(self.current_frame)

            if annotation:
                class_name, layer_number = annotation
                predicted_class_name = class_name

                self.class_name_label.setStyleSheet(f"""
                    background-color: {class_colors[class_name]};
                    color: white;
                    font-size: 15px;
                    padding: 5px;
                    border-radius: 2px;
                """)

                self.predicted_class_label.setStyleSheet(f"""
                    background-color: {class_colors[predicted_class_name]};
                    color: white;
                    font-size: 15px;
                    padding: 5px;
                    border-radius: 2px;
                """)

                self.layer_number_label.setStyleSheet(f"""
                    background-color: black;
                    color: white;
                    font-size: 15px;
                    padding: 5px;
                    border-radius: 2px;
                """)

                self.class_name_label.setText(class_name)
                self.predicted_class_label.setText(predicted_class_name)
                self.layer_number_label.setText(f"Layer number: {layer_number}")
            
            # Load and display the audio waveform
            audio_chunk_path = self.audio_pattern % self.current_frame
            sample_rate, audio_data = wavfile.read(audio_chunk_path)

            self.audio_waveform_plot.clear()
            # self.audio_waveform_plot.plot(audio_data)
            self.audio_waveform_plot.plot(audio_data, pen=pg.mkPen('b', width=1))  # 'b' stands for blue

            self.progress_slider.setValue(self.current_frame)

            self.current_frame += 1

            # if self.timer.isActive():
            #     # Start audio playback for the current frame
            #     audio_player = AudioPlayer(audio_chunk_path)
            #     audio_player.start()
            #     self.audio_player_threads.append(audio_player)

                # # Stop and remove the previous audio player
                # if len(self.audio_player_threads) > 1:
                #     previous_audio_player = self.audio_player_threads.pop(0)
                #     previous_audio_player.terminate()

            # if self.record:
            #     screen = QGuiApplication.primaryScreen()
            #     screenshot = screen.grabWindow(self.winId())  # Capture a screenshot of the current window
            #     screenshot = screenshot.toImage().convertToFormat(QImage.Format_RGB888)
            #     screenshot_array = np.array(screenshot.constBits()).reshape(screenshot.height(), screenshot.width(), 3)
            #     self.video_writer.write(screenshot_array)

            if self.record:
                screenshot = QScreen.grabWindow(QGuiApplication.primaryScreen(), self.winId())
                screenshot = screenshot.toImage().convertToFormat(QImage.Format_RGB888)
                screenshot_array = np.ndarray((screenshot.height(), screenshot.width(), 3), buffer=screenshot.bits(), dtype=np.uint8, strides=(screenshot.bytesPerLine(), 3, 1))
                self.video_writer.write(screenshot_array)




                # screenshot = QScreen.grabWindow(self.winId())
                # screenshot = screenshot.toImage().convertToFormat(QImage.Format_RGB888)
                # screenshot_array = np.array(screenshot.constBits()).reshape(screenshot.height(), screenshot.width(), 3)
                # self.video_writer.write(screenshot_array)

        else:
            self.timer.stop()


class AudioPlayer(QThread):
    def __init__(self, audio_file):
        super().__init__()
        self.audio_file = audio_file

    def run(self):
        sound = AudioSegment.from_wav(self.audio_file)
        play(sound)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    Dataset = '/home/chenlequn/Dataset/LDED_acoustic_visual_monitoring_dataset/segmented/22'
    image_pattern = os.path.join(Dataset, "images", 'sample_22_%d.jpg')
    audio_pattern = os.path.join(Dataset, "denoised_audio", 'sample_22_%d.wav')
    annotations_file = os.path.join(Dataset, 'annotations_22.txt')
    total_frames = 1849  # Replace with the actual number of frames

    annotations = load_annotations(annotations_file)

    player = ImageSequencePlayer(image_pattern, audio_pattern, annotations, total_frames, fps=15)
    player.show()

    sys.exit(app.exec_())
