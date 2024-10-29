import os
import cv2
import requests
import numpy as np
from keras.models import load_model
from configs import ALARM_SOUND, MAIN_LOG_LEVEL, MAIN_LOG_PATH
from core.logger import logger
from utils import load_sound, play_sound, stop_sound

image_size = (224, 224)
THRESHOLD = 0.2  # 0.0 - 1
model_to_use = 'annotations_magic_mouse_model.h5'


class VideoObjectDetection:
    def __init__(self) -> None:
        self.logging = logger(MAIN_LOG_PATH, MAIN_LOG_LEVEL)
        self.load_configs()
        self.object_detected = None

    def load_configs(self) -> None:
        self.load_keras_model()
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        self.alarm_sound = load_sound(ALARM_SOUND)
        self.logging.info('> VideoObjectDetection configs loaded.')

    def load_keras_model(self) -> None:
        self.object_detection_model = load_model(
            os.path.join(os.getcwd(), 'models', 'keras', model_to_use)
        )

    def video_capture_get(self) -> cv2.VideoCapture:
        device_number = 0
        cap = cv2.VideoCapture(device_number)
        if cap.isOpened():
            self.logging.info('> Video stream open.')
        else:
            msg = "Problem opening video stream."
            self.logging.error(f'> {msg}')
            raise Exception(msg)
        return cap

    def video_capture_release(self, cap: cv2.VideoCapture) -> None:
        cap.release()
        cv2.destroyAllWindows()

    def start_video_analysis(self) -> None:
        cap = self.video_capture_get()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            height, width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Draw black bars for text
            cv2.rectangle(frame, (0, height - 50), (width, height), (0, 0, 0), thickness=cv2.FILLED)

            # Object detection
            processed_frame = self.preprocess_frame_for_model(frame)
            predictions = self.object_detection_model.predict(processed_frame)

            # Apply threshold and display results
            detected = self.process_predictions(predictions, frame, width, height)

            # Display detection status
            if detected:
                play_sound(self.alarm_sound)
                cv2.putText(frame, "Object Detected", (10, height - 20), self.font, 1, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                stop_sound(self.alarm_sound)
                cv2.putText(frame, "No Object", (10, height - 20), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            # Display the frame
            cv2.imshow('frame', frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_capture_release(cap)

    def preprocess_frame_for_model(self, frame: np.ndarray) -> np.ndarray:
        resized_frame = cv2.resize(frame, image_size)
        normalized_frame = resized_frame / 255.0
        return np.expand_dims(normalized_frame, axis=0)

    def process_predictions(self, predictions: np.ndarray, frame: np.ndarray, width: int, height: int) -> bool:
        """
        Processes predictions, applies threshold, and draws bounding boxes if confidence is above the threshold.
        """
        x_min, y_min, x_max, y_max = predictions[0]

        # Scale bounding box to original frame size
        x_min = int(x_min * width)
        y_min = int(y_min * height)
        x_max = int(x_max * width)
        y_max = int(y_max * height)

        # Calculate confidence score for the bounding box
        box_area = (x_max - x_min) * (y_max - y_min)
        if box_area > 0.3 * width * height:
            # Tick border on screen
            cv2.rectangle(frame, (0,0), (width,height), (0,0,255), thickness=5)
            
            if self.object_detected is None:
                self.object_detected = "coca_cola"
                self.trigger_url_call(self.object_detected)
            
            # Draw bounding box
            #cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            return True  # Object detected
        else:
            self.object_detected = None
            self.trigger_url_call(None)
        return False

    def trigger_url_call(self, product_name: str | None) -> None:
        """Triggers a URL call to the Flask server with the product name."""
        if product_name:
            url = f"http://localhost:8042/?product={product_name}"
        else:
            url = "http://localhost:8042/?product=None"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                self.logging.info(f'> Successfully triggered URL: {url}')
            else:
                self.logging.error(f'> Failed to trigger URL: {url} - Status code: {response.status_code}')
        except requests.RequestException as e:
            self.logging.error(f'> Error triggering URL: {e}')

if __name__ == '__main__':
    video_object_detection = VideoObjectDetection()
    video_object_detection.start_video_analysis()
