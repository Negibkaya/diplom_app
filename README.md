# Road Sign Recognition System (YOLOv8)

Welcome to the GitHub repository of our innovative Road Sign Recognition System utilizing the power of YOLOv8. This application is designed to accurately detect and recognize various road signs, assisting drivers in navigating more safely and effectively. The system employs the Ultralytics YOLO model alongside a Deep SORT tracker to ensure real-time detection and tracking of road signs.

## Features

- **Speed Limit and Traffic Sign Detection:** Identifies and interprets a wide range of traffic signs including speed limits, no entry, pedestrian crossings, and more.
- **Real-Time Video Processing:** Capable of processing live video streams to detect and classify road signs instantaneously.
- **Image Processing Capabilities:** Analyzes static images for road sign detection and classification.
- **Voice Notifications:** Utilizes a text-to-speech engine to audibly announce detected road signs, enhancing driver awareness.
- **Custom User Interface:** Provides a user-friendly GUI built with PyQt5 for easy interaction with the application.

## Installation

To run this road sign recognition system, ensure you have Python 3.6 or newer installed on your machine. Clone this repository, and then install the required dependencies:

```bash
git clone https://github.com/your-repository/road-sign-recognition.git
cd road-sign-recognition
pip install -r requirements.txt
```

## Usage

Run the application by executing the following command:

```bash
python road_sign_recognition.py
```

Upon launching, you will be greeted with a simple and intuitive interface. You can choose to process a static image, a video file, or real-time video feed from your webcam.

## Technologies

- **OpenCV:** For image and video processing.
- **YOLOv8:** For state-of-the-art object detection.
- **PyQt5:** For building the graphical user interface.
- **Pyttsx3:** For enabling text-to-speech capabilities.
- **Deep SORT:** For robust object tracking.

## Configuration and Customization

The system is highly configurable with several parameters such as confidence thresholds, detection box thickness, and label text sizes adjustable according to user preference. It is designed for flexibility and can be adapted to various types of road signs by modifying the `CLASS_LIST` within the `YOLOv8` class.

## Contributing

Contributions to this project are welcome! Whether it's improving the detection algorithms, enhancing the user interface, or providing translations for non-English speakers, we appreciate your input.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
