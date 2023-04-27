import sys
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QGraphicsDropShadowEffect, QHBoxLayout,QGridLayout
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import pyautogui
import time
import numpy as np


# Load the annotations
def load_annotations(file_path):
    annotations = {}
    with open(file_path, 'r') as f:
        for line in f:
            timestamp, class_name, layer_number, _ = line.strip().split('\t')
            annotations[int(float(timestamp) * 10)] = (class_name, layer_number)
    return annotations


# Create the main window
class MainWindow(QWidget):
    def __init__(self, image_pattern, annotations, total_frames, fps=20):
        super().__init__()

        self.image_pattern = image_pattern
        self.annotations = annotations
        self.total_frames = total_frames
        self.current_frame = 1

        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_image)
        self.timer.start(1000 / fps)


    # def init_ui(self):
    #     # Create the color legend
    #     self.color_legend = QLabel(self)
    #     self.color_legend.setText(
    #         "Color legend:\n"
    #         "<font color='orange'>Crack</font>\n"
    #         "<font color='red'>Keyhole pores</font>\n"
    #         "<font color='black'>Laser-off</font>\n"
    #         "<font color='blue'>Defect-free</font>"
    #     )


    #     self.image_label = QLabel(self)
    #     self.image_label.setAlignment(Qt.AlignCenter)
    #     self.class_name_label = QLabel(self)
    #     self.class_name_label.setAlignment(Qt.AlignCenter)
    #     self.layer_number_label = QLabel(self)
    #     self.layer_number_label.setAlignment(Qt.AlignCenter)
    #     self.predicted_class_label = QLabel(self)  
    #     self.predicted_class_label.setAlignment(Qt.AlignCenter)

    #     # Create the layout for the color legend
    #     legend_layout = QVBoxLayout()
    #     legend_layout.addWidget(self.color_legend)

    #     # Create the layout for the ground truth and predicted labels
    #     info_layout = QVBoxLayout()
    #     info_layout.addWidget(QLabel("Ground truth:"))
    #     info_layout.addWidget(self.class_name_label)
    #     info_layout.addWidget(QLabel("Predicted:"))
    #     info_layout.addWidget(self.predicted_class_label)

    #     # Create a horizontal layout for the labels
    #     label_layout = QHBoxLayout()
    #     label_layout.addWidget(self.class_name_label)
    #     label_layout.addWidget(self.layer_number_label)

    #     # Set the class_name_label style
    #     self.class_name_label.setStyleSheet("""
    #         background-color: rgba(0, 0, 0, 180);
    #         color: white;
    #         font-size: 18px;
    #         padding: 10px;
    #         border-radius: 5px;
    #     """)

    #     # Set the layer_number_label style
    #     self.layer_number_label.setStyleSheet("""
    #         background-color: black;
    #         color: white;
    #         font-size: 18px;
    #         padding: 10px;
    #         border-radius: 5px;
    #     """)

    #     main_layout = QVBoxLayout(self)
    #     main_layout.addWidget(self.image_label)
    #     # main_layout.addLayout(label_layout)  # Add the QHBoxLayout to the main layout
    #     main_layout.addLayout(legend_layout)
    #     main_layout.addLayout(info_layout)

    #     self.setLayout(main_layout)
    
    def init_ui(self):
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        # Create the color legend labels
        self.color_legend = QLabel("Legends", self)

        self.legend_crack = QLabel("Crack", self)
        self.legend_crack.setStyleSheet("background-color: red; padding: 5px; border-radius: 5px;")
        self.legend_keyhole_pores = QLabel("Keyhole pores", self)
        self.legend_keyhole_pores.setStyleSheet("background-color: green; padding: 5px; border-radius: 5px;")
        self.legend_laser_off = QLabel("Laser-off", self)
        self.legend_laser_off.setStyleSheet("background-color: blue; padding: 5px; border-radius: 5px;")
        self.legend_defect_free = QLabel("Defect-free", self)
        self.legend_defect_free.setStyleSheet("background-color: yellow; padding: 5px; border-radius: 5px;")

        # Create the layout for the color legend
        legend_layout = QGridLayout()
        legend_layout.addWidget(self.color_legend, 0, 0, 1, 2)
        legend_layout.addWidget(self.legend_crack, 1, 0)
        legend_layout.addWidget(self.legend_keyhole_pores, 1, 1)
        legend_layout.addWidget(self.legend_laser_off, 2, 0)
        legend_layout.addWidget(self.legend_defect_free, 2, 1)

        self.class_name_label = QLabel(self)
        self.class_name_label.setAlignment(Qt.AlignCenter)
        self.layer_number_label = QLabel(self)
        self.layer_number_label.setAlignment(Qt.AlignCenter)
        self.predicted_class_label = QLabel(self)
        self.predicted_class_label.setAlignment(Qt.AlignCenter)

        # Create the layout for the ground truth, predicted labels, and layer number
        info_stacked_layout = QVBoxLayout()
        info_stacked_layout.addWidget(QLabel("Ground truth:"))
        info_stacked_layout.addWidget(self.class_name_label)
        info_stacked_layout.addWidget(QLabel("Predicted:"))
        info_stacked_layout.addWidget(self.predicted_class_label)
        info_stacked_layout.addWidget(self.layer_number_label)

        # Create the main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(info_stacked_layout)
        main_layout.addLayout(legend_layout)

        self.setLayout(main_layout)


  


    def update_image(self):
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

                predicted_class = class_name

                # Set the class_name_label color based on the class
                class_colors = {
                    'Crack': 'orange',
                    'Keyhole pores': 'red',
                    'Laser-off': 'black',
                    'Defect-free': 'blue',
                }
                self.class_name_label.setStyleSheet(f"""
                    background-color: {class_colors[class_name]};
                    color: white;
                    font-size: 18px;
                    padding: 10px;
                    border-radius: 5px;
                """)

                self.class_name_label.setText(class_name)
                self.layer_number_label.setText(f"Layer: {layer_number}")
                self.predicted_class_label.setText(predicted_class)

            self.current_frame += 1     
        else:
            self.timer.stop()

          

if __name__ == '__main__':
    app = QApplication(sys.argv)
    image_pattern = 'sample_22_%d.jpg'
    annotations_file = 'annotations_22.txt'
    total_frames = 1849  # Replace with the actual number of frames

    annotations = load_annotations(annotations_file)

    window = MainWindow(image_pattern, annotations, total_frames, fps=20)
    window.setWindowTitle('Ground Truth - Thin wall')
    window.show()

    sys.exit(app.exec_())
