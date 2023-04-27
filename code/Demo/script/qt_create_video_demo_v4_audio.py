import sys
import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout
from PyQt5.uic import loadUi
import os
import pyqtgraph as pg
from scipy.io import wavfile


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

        self.audio_waveform_plot = pg.PlotWidget()
        self.audio_waveform_widget.setLayout(QVBoxLayout())
        self.audio_waveform_widget.layout().addWidget(self.audio_waveform_plot)


        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000/fps)

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
            new_width, new_height = 729, 405  # Change these values as needed
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
            self.audio_waveform_plot.plot(audio_data)
                
            self.current_frame += 1
        else:
            self.timer.stop()


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
