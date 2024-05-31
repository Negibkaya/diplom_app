import sys
import cv2
import pyttsx3
import time
import random
from PyQt5.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QHBoxLayout)
from PyQt5.QtGui import QPixmap, QImage, QColor, QPalette, QFont
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO
from deepsort_tracker import Tracker
import threading

# Класс для обработки изображения (ресайз, рисование боксов)


class ImageProcessor:
    SCALE_SHOW = 100

    # Ресайз изображения
    def resize(img, scale_percent):
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return resized

    # Рисование боксов
    def draw_boxes(img, result, class_list, box_thickness, label_text_size, is_video, deep_id=""):
        xyxy = result.boxes.xyxy.numpy()
        confidence = result.boxes.conf.numpy()
        class_id = result.boxes.cls.numpy().astype(int)
        class_name = [class_list[x] for x in class_id]
        out_image = img.copy()
        for label, con, box in zip(class_name, confidence, xyxy):
            box_color = (0, 0, 255)
            text_color = (255, 255, 255)
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cv2.rectangle(out_image, (x1, y1), (x2, y2),
                          box_color, thickness=box_thickness)
            if is_video:
                text_print = f'ID - {deep_id} {label} {con:.2f}'
            else:
                text_print = f'{label} {con:.2f}'
            text_location = (x1, y1 - 10)
            labelSize, baseLine = cv2.getTextSize(
                text_print, cv2.FONT_HERSHEY_SIMPLEX, label_text_size, box_thickness)
            cv2.rectangle(out_image, (x1, y1 - labelSize[1] - 10),
                          (x1 + labelSize[0], y1 + baseLine - 10), box_color, cv2.FILLED)
            cv2.putText(out_image, text_print, text_location, cv2.FONT_HERSHEY_SIMPLEX,
                        label_text_size, text_color, box_thickness, cv2.LINE_AA)
        return out_image


# Класс для работы с YOLOv8 (для работы с другими моделям нужно изменить путь к модели, а также изменить CLASS_LIST)
class YOLOv8:
    # Все возможные классы
    CLASS_LIST = {
        0: 'artificial_unevenness',
        1: 'bicycle_lane',
        2: 'bus',
        3: 'children',
        4: 'gas_station',
        5: 'give_way',
        6: 'intersection_with_bicycle_path',
        7: 'main_road',
        8: 'no_entry',
        9: 'no_stopping',
        10: 'no_traffic',
        11: 'pedestrian_crossing',
        12: 'residential_area',
        13: 'speed_limit_20',
        14: 'speed_limit_40',
        15: 'trucks',
        16: 'uneven_road'
    }
    # CLASS_LIST = {
    #     0: 'forb_ahead',
    #     1: 'forb_left',
    #     2: 'forb_overtake',
    #     3: 'forb_right',
    #     4: 'forb_speed_over_10',
    #     5: 'forb_speed_over_100',
    #     6: 'forb_speed_over_130',
    #     7: 'forb_speed_over_20',
    #     8: 'forb_speed_over_30',
    #     9: 'forb_speed_over_40',
    #     10: 'forb_speed_over_5',
    #     11: 'forb_speed_over_50',
    #     12: 'forb_speed_over_60',
    #     13: 'forb_speed_over_70',
    #     14: 'forb_speed_over_80',
    #     15: 'forb_speed_over_90',
    #     16: 'forb_stopping',
    #     17: 'forb_trucks',
    #     18: 'forb_u_turn',
    #     19: 'forb_weight_over_3.5t',
    #     20: 'forb_weight_over_7.5t',
    #     21: 'info_bus_station',
    #     22: 'info_crosswalk',
    #     23: 'info_highway',
    #     24: 'info_one_way_traffic',
    #     25: 'info_parking',
    #     26: 'info_taxi_parking',
    #     27: 'mand_bike_lane',
    #     28: 'mand_left',
    #     29: 'mand_left_right',
    #     30: 'mand_pass_left',
    #     31: 'mand_pass_left_right',
    #     32: 'mand_pass_right',
    #     33: 'mand_right',
    #     34: 'mand_roundabout',
    #     35: 'mand_straigh_left',
    #     36: 'mand_straight',
    #     37: 'mand_straight_right',
    #     38: 'prio_give_way',
    #     39: 'prio_priority_road',
    #     40: 'prio_stop',
    #     41: 'warn_children',
    #     42: 'warn_construction',
    #     43: 'warn_crosswalk',
    #     44: 'warn_cyclists',
    #     45: 'warn_domestic_animals',
    #     46: 'warn_other_dangers',
    #     47: 'warn_poor_road_surface',
    #     48: 'warn_roundabout',
    #     49: 'warn_slippery_road',
    #     50: 'warn_speed_bumper',
    #     51: 'warn_traffic_light',
    #     52: 'warn_tram',
    #     53: 'warn_two_way_traffic',
    #     54: 'warn_wild_animals'
    # }

    # Конструктор
    def __init__(self):
        self.model = YOLO("trained-models/ru.pt")
        # self.model = YOLO("trained-models/eu.pt")


# Класс для работы с камерой (вызов и остановка камеры)
class Camera:
    def __init__(self):
        self.video_capture = None
        self.is_video = False

    def start(self, source):
        if source == "webcam":
            self.video_capture = cv2.VideoCapture(0)
        else:
            self.video_capture = cv2.VideoCapture(source)
        self.is_video = True

    def stop(self):
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.is_video = False

# Класс для работы с GUI


class GUIYoloV8(QWidget):

    # CONSTANTS
    CONFIDENCE_THRESHOLD = 80
    BOX_THICKNESS = 2
    LABEL_TEXT_SIZE = 1
    COLORS = [(random.randint(0, 255), random.randint(0, 255),
               random.randint(0, 255)) for _ in range(10)]
    FONT_SIZES = {
        'title': 24,
        'label': 16
    }

    # Инициализация приложения
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система распознования дорожных знаков")
        self.setFixedSize(1200, 800)
        self.layout = QVBoxLayout()
        self.default_image_path = "interface.png"

        self.tracker = Tracker()
        self.tts_engine = pyttsx3.init()
        self.last_spoken_sign = None
        self.last_spoken_time = time.time()
        self.ids_announced = set()

        title_font = QFont("Arial", GUIYoloV8.FONT_SIZES['title'], QFont.Bold)
        label_font = QFont("Arial", GUIYoloV8.FONT_SIZES['label'])

        title_label = QLabel("Система распознования дорожных знаков (YOLOv8)")
        title_label.setAlignment(Qt.AlignHCenter)
        title_label.setStyleSheet("color: white;")
        title_label.setFont(title_font)
        self.layout.addWidget(title_label)

        self.result_label = QLabel()
        self.result_label.setStyleSheet(
            "border: 1px solid rgb(127, 129, 130); margin: 10px; padding: 10px; border-radius: 10px; background-color: black;")
        self.result_label.setFixedHeight(550)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.result_label)

        default_image = QPixmap(self.default_image_path)
        self.result_label.setPixmap(default_image.scaled(
            self.result_label.width(), self.result_label.height(), Qt.KeepAspectRatio))

        self.class_label = QLabel()
        self.class_label.setAlignment(Qt.AlignCenter)
        self.class_label.setStyleSheet(
            "border: 1px solid rgb(127, 129, 130); font-size: 16px; font-weight: bold; margin: 10px; padding: 10px; border-radius: 10px; background-color: rgb(25, 25, 25); color: white;")
        self.class_label.setFont(label_font)
        self.layout.addWidget(self.class_label)

        button_layout = QHBoxLayout()

        image_button = QPushButton("Выбрать изображение")
        image_button.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: white; padding: 5px; margin: 5px; border-radius: 10px; background-color: rgb(127, 129, 130);")
        image_button.clicked.connect(self.select_image)
        button_layout.addWidget(image_button)

        video_button = QPushButton("Выбрать видео")
        video_button.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: white; padding: 5px; margin: 5px; border-radius: 10px; background-color: rgb(127, 129, 130);")
        video_button.clicked.connect(self.select_video)
        button_layout.addWidget(video_button)

        self.start_webcam_button = QPushButton("Включить видеокамеру")
        self.start_webcam_button.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: white; padding: 5px; margin: 5px; border-radius: 10px; background-color: rgb(127, 129, 130);")
        self.start_webcam_button.clicked.connect(self.start_webcam)
        button_layout.addWidget(self.start_webcam_button)

        self.layout.addLayout(button_layout)

        end_button_layout = QHBoxLayout()

        self.stop_button = QPushButton("Остановить предсказание")
        self.stop_button.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: white; padding: 5px; margin: 5px; border-radius: 10px; background-color: rgb(127, 129, 130);")
        self.stop_button.clicked.connect(self.stop_prediction)
        end_button_layout.addWidget(self.stop_button)

        close_button = QPushButton("Закрыть")
        close_button.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: white; padding: 5px; margin: 5px; border-radius: 10px; background-color: rgb(127, 129, 130);")
        close_button.clicked.connect(self.close)
        end_button_layout.addWidget(close_button)

        self.layout.addLayout(end_button_layout)

        self.setLayout(self.layout)

        self.yolov8 = YOLOv8()
        self.camera = Camera()

    # Выбрать изображение
    def select_image(self):
        image_file, _ = QFileDialog.getOpenFileName(
            self, "Выбрать изображение", "", "Изображения (*.png *.jpg *.jpeg)")
        if image_file:
            self.start_prediction(image_file, is_video=False)

    # Выбрать видео
    def select_video(self):
        video_file, _ = QFileDialog.getOpenFileName(
            self, "Выбрать видео", "", "Видео (*.mp4 *.avi *.mov)")
        if video_file:
            self.start_prediction(video_file, is_video=True)

    # Включить видеокамеру
    def start_webcam(self):
        self.result_label.setPixmap(QPixmap(self.default_image_path).scaled(
            self.result_label.width(), self.result_label.height(), Qt.KeepAspectRatio))
        self.start_prediction("webcam", is_video=True)

    # Остановить предсказание
    def stop_prediction(self):
        self.camera.stop()
        self.result_label.clear()
        self.class_label.clear()
        self.result_label.setPixmap(QPixmap(self.default_image_path).scaled(
            self.result_label.width(), self.result_label.height(), Qt.KeepAspectRatio))

    # Начать предсказание
    def start_prediction(self, file, is_video=False):
        if is_video:
            self.camera.start(file)
            self.update_video_frame()
        else:
            frame = cv2.imread(file)
            self.process_frame(frame)

    # Обновить видео кадр
    def update_video_frame(self):
        if self.camera.video_capture is not None:
            start_time = time.time()

            ret, frame = self.camera.video_capture.read()
            if not ret:
                self.stop_prediction()
                return

            self.process_frame(frame)

            end_time = time.time()
            frame_processing_time = end_time - start_time
            print(
                f"Frame processing time: {frame_processing_time:.4f} seconds")
            QTimer.singleShot(0, self.update_video_frame)

    # Обработать кадр
    def process_frame(self, frame):
        start_time = time.time()

        results = self.yolov8.model.predict(frame)

        if self.camera.is_video:
            detections = self.extract_detections(results)
            self.tracker.update(frame, detections)
            frame = self.draw_tracks(frame, results)
        else:
            frame = ImageProcessor.draw_boxes(
                frame, results[0], YOLOv8.CLASS_LIST, GUIYoloV8.BOX_THICKNESS, GUIYoloV8.LABEL_TEXT_SIZE, False)

        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Processing time: {processing_time:.4f} seconds")

        self.display_frame(frame, results)

    # Извлечь обнаружения из кадра
    def extract_detections(self, results):
        detections = []
        for i in range(len(results[0].boxes)):
            x1, y1, x2, y2 = results[0].boxes.xyxy[i].numpy()
            score = results[0].boxes.conf[i].numpy()
            if score > 0.5:
                detections.append([int(x1), int(y1), int(x2), int(y2), score])
        return detections

    # Отрисовать треков
    def draw_tracks(self, frame, results):
        display_img = frame
        for track in self.tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            track_id = track.track_id
            if track_id not in self.ids_announced:
                self.ids_announced.add(track_id)
                self.speak_detected_signs(results[0].boxes, False)

            display_img = ImageProcessor.draw_boxes(
                display_img, results[0], YOLOv8.CLASS_LIST, GUIYoloV8.BOX_THICKNESS, GUIYoloV8.LABEL_TEXT_SIZE, True, track_id)
        return display_img

    # Отображение кадра
    def display_frame(self, frame, results):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        q_image = QImage(
            rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], rgb_image.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.result_label.setPixmap(pixmap.scaled(
            self.result_label.width(), self.result_label.height(), Qt.KeepAspectRatio))

        labels = self.generate_labels(results)
        label_text = "\n".join(labels) if labels else "No detection"
        self.class_label.setText(label_text)

        QApplication.processEvents()

        self.speak_detected_signs(results[0].boxes, False)

    # Отображение подписей
    def generate_labels(self, results):
        labels = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for i in range(len(results[0].boxes)):
                class_id = results[0].boxes.cls[i].item()
                if class_id in YOLOv8.CLASS_LIST:
                    class_name = YOLOv8.CLASS_LIST[class_id]
                    confidence = results[0].boxes.conf[i] * 100
                    if confidence > GUIYoloV8.CONFIDENCE_THRESHOLD:
                        label = f"{class_name}: {confidence:.2f}%"
                        labels.append(label)
        return labels

    # Голосовое оповещение
    def speak_detected_signs(self, boxes, image_selection):
        if boxes is not None and len(boxes) > 0:
            detected_signs = []
            for i in range(len(boxes)):
                class_id = boxes.cls[i].item()
                if class_id in YOLOv8.CLASS_LIST:
                    class_name = YOLOv8.CLASS_LIST[class_id]
                    confidence = boxes.conf[i] * 100
                    if confidence > GUIYoloV8.CONFIDENCE_THRESHOLD:
                        detected_signs.append(class_name)

            if detected_signs:
                current_time = time.time()
                if detected_signs != self.last_spoken_sign or current_time - self.last_spoken_time > 5:
                    threading.Thread(target=self.speak_text, args=(
                        " и ".join(detected_signs),)).start()
                    self.last_spoken_sign = detected_signs
                    self.last_spoken_time = current_time
            else:
                if image_selection:
                    threading.Thread(target=self.speak_text,
                                     args=("No detection",)).start()

    def speak_text(self, text):
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(40, 40, 40))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    app.setPalette(palette)

    gui = GUIYoloV8()
    gui.show()

    sys.exit(app.exec_())
