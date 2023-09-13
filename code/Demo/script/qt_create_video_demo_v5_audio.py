import sys
import cv2
import time
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap, QScreen, QGuiApplication
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
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
            values = line.strip().split('\t')
            if len(values) == 4:
                timestamp, class_name, class_name_v2, layer_number = values
            else:
                # Use dummy values for unannotated samples
                timestamp = values[0]
                class_name, layer_number = "Unknown", "Unknown"
            
            annotations[int(float(timestamp) * 25)] = (class_name, layer_number)
    return annotations


def count_files_in_directory(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])


class ImageSequencePlayer(QMainWindow):
    def __init__(self, image_pattern, audio_pattern, annotations, total_frames,fps=25, parent=None):
        super().__init__(parent)
        loadUi("multimodal_player.ui", self)  # Load the .ui file

        self.image_pattern = image_pattern
        self.audio_pattern = audio_pattern
        self.annotations = annotations
        self.current_frame = 1
        self.total_frames = total_frames
        self.fps = fps
        self.record = False
        self.playing = False
        self.video_writer = None

        self.audio_data_cache = {}
        for frame_number in range(1, total_frames+1):
            audio_chunk_path = self.audio_pattern % frame_number
            sample_rate, audio_data = wavfile.read(audio_chunk_path)
            self.audio_data_cache[frame_number] = audio_data


        self.audio_waveform_plot = pg.PlotWidget()
        self.audio_waveform_plot.setBackground('w')  # 'w' stands for white
        self.audio_waveform_plot.setYRange(-6000, 6000) # set the audio range -- denoised; (-7000, 7000) for raw audio
        self.audio_waveform_widget.setLayout(QVBoxLayout())
        self.audio_waveform_widget.layout().addWidget(self.audio_waveform_plot)

        self.progress_slider.valueChanged.connect(self.slider_value_changed)
        self.progress_slider.setMaximum(self.total_frames)
        self.stop_button.clicked.connect(self.stop_button_clicked)
        self.start_button.clicked.connect(self.start_button_clicked)
        # self.record_button.clicked.connect(self.toggle_record)

        self.media_player = QMediaPlayer(self)
        self.media_player.setNotifyInterval(1000 // fps)
        self.media_player.positionChanged.connect(self.update_frame)
        self.media_player.error.connect(self.media_player_error)

        self.audio_player_threads = []


    def run(self):
        while self.playing and self.current_frame < self.total_frames:
            start_time = time.time()
            self.update_frame()

            elapsed_time = time.time() - start_time
            frame_duration = 1 / self.fps
            time_to_sleep = max(frame_duration - elapsed_time, 0)
            time.sleep(time_to_sleep)

            # Break the loop if the user closes the window
            if cv2.getWindowProperty("Image Sequence Player", cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()


    def slider_value_changed(self, value):
        self.current_frame = value
        self.update_frame()

    def media_player_error(self, error):
        print(f"Media player error: {error}")


    def start_button_clicked(self):
        if not self.playing:
            self.playing = True
            self.run()

    def stop_button_clicked(self):
        if self.playing:
            self.playing = False


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


    def update_frame(self):
        class_colors = {
            'Crack': 'rgb(245, 121, 0)',
            'Keyhole pores': 'rgb(239, 41, 41)',
            'Laser-off': 'rgb(46, 52, 54)',
            'Defect-free': 'rgb(114, 159, 207)',
            'Unknown': 'rgb(140, 140, 140)',
        }
        # class_colors = {
        #     'Defective': 'rgb(239, 41, 41)',
        #     'Laser-off': 'rgb(46, 52, 54)',
        #     'Defect-free': 'rgb(114, 159, 207)',
        #     'Unknown': 'rgb(140, 140, 140)',
        # }
        if self.current_frame < self.total_frames:


            image_path = self.image_pattern % self.current_frame
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Resize the image
            new_width, new_height = 512, 384  # Change these values as needed. Original 640*480
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

             # Display the image using OpenCV
            cv2.imshow("Image Sequence Player", image)
            
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
            audio_data = self.audio_data_cache[self.current_frame]
            self.audio_waveform_plot.clear()
            self.audio_waveform_plot.plot(audio_data, pen=pg.mkPen('b', width=1))  # 'b' stands for blue

            self.progress_slider.setValue(self.current_frame)

            self.current_frame += 1

            if self.record:
                screenshot = QScreen.grabWindow(QGuiApplication.primaryScreen(), self.winId())
                screenshot = screenshot.toImage().convertToFormat(QImage.Format_RGB888)
                screenshot_array = np.ndarray((screenshot.height(), screenshot.width(), 3), buffer=screenshot.bits(), dtype=np.uint8, strides=(screenshot.bytesPerLine(), 3, 1))
                self.video_writer.write(screenshot_array)


        else:
            self.timer.stop()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    sample_index = 23

    Dataset = f'/home/chenlequn/Dataset/LDED_acoustic_visual_monitoring_dataset/segmented_25Hz/{sample_index}'
    image_dir = os.path.join(Dataset, "images")
    image_pattern = os.path.join(Dataset, "images", f'sample_{sample_index}_%d.jpg')
    audio_pattern = os.path.join(Dataset, "denoised_audio", f'sample_{sample_index}_%d.wav')
    annotations_file = os.path.join(Dataset, f'annotations_{sample_index}.txt')

    total_frames = count_files_in_directory(image_dir)

    annotations = load_annotations(annotations_file)

    player = ImageSequencePlayer(image_pattern, audio_pattern, annotations, total_frames, fps=25)
    player.show()

    player.run()


