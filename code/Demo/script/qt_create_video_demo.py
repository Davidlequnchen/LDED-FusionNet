import sys
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QGraphicsDropShadowEffect
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

# Load the annotations
def load_annotations(file_path):
    annotations = {}
    with open(file_path, 'r') as f:
        for line in f:
            timestamp, class_name, layer_number, _ = line.strip().split()
            annotations[int(timestamp * 10)] = (class_name, layer_number)
    return annotations

# Create the main window
class MainWindow(QWidget):
    def __init__(self, image_pattern, annotations, total_frames, fps=10):
        super().__init__()

        self.image_pattern = image_pattern
        self.annotations = annotations
        self.total_frames = total_frames
        self.current_frame = 0

        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_image)
        self.timer.start(1000 / fps)

    def init_ui(self):
        layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.annotation_label = QLabel(self)

        # Set the annotation_label style
        self.annotation_label.setStyleSheet("""
            background-color: rgba(0, 0, 0, 180);
            color: white;
            font-size: 18px;
            padding: 10px;
            border-radius: 5px;
        """)

        # Add a drop shadow effect to the annotation_label
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0))
        shadow.setOffset(2, 2)
        self.annotation_label.setGraphicsEffect(shadow)

        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.annotation_label, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    def update_image(self):
        if self.current_frame < self.total_frames:
            image_path = self.image_pattern % self.current_frame
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            h, w, _ = image.shape
            qt_image = QImage(image.data, w, h, w * 3, QImage.Format_RGB888)
            pixmap = QPixmap(qt_image)

            self.image_label.setPixmap(pixmap)

            annotation = self.annotations.get(self.current_frame)
            if annotation:
                class_name, layer_number = annotation
                self.annotation_label.setText(f"{class_name}, Layer: {layer_number}")

            self.current_frame += 1
        else:
            self.timer.stop()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    image_pattern = 'sample_22_%d.jpg'
    annotations_file = 'annotations_22.txt'
    total_frames = 1849  # Replace with the actual number of frames

    annotations = load_annotations(annotations_file)

    window = MainWindow(image_pattern, annotations, total_frames)
    window.setWindowTitle('Image Sequence with Annotations')
    window.show()

    sys.exit(app.exec_())
