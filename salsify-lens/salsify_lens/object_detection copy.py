import os
import cv2
import numpy as np
from keras.models import load_model

from configs import ALARM_SOUND, MAIN_LOG_LEVEL, MAIN_LOG_PATH
from core.logger import logger
from utils import load_sound, play_sound, stop_sound

image_size = (224, 224)
image_channels = 3


class VideoObjectDetection:
    def __init__(self) -> None:
        self.logging = logger(MAIN_LOG_PATH, MAIN_LOG_LEVEL)
        self.load_configs()

    def load_configs(self) -> None:
        self.load_keras_model()
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        self.alarm_sound = load_sound(ALARM_SOUND)
        self.logging.info('> VideoObjectDetection configs loaded.')

    def load_keras_model(self) -> None:
        self.object_detection_model = load_model(
            os.path.join(os.getcwd(), 'models', 'keras', 'object_detection_model.h5')
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

            # Draw black bars top and bottom
            cv2.rectangle(frame, (0, height-50), (width, height), (0, 0, 0), thickness=cv2.FILLED)

            # Object detection with the Keras model
            processed_frame = self.preprocess_frame_for_model(frame)
            prediction = (self.object_detection_model.predict(processed_frame) > 0.5).astype("int32")[0][0]

            # Display detection result
            if prediction == 1:
                cv2.putText(frame, "Object Detected", (10, height - 20), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                play_sound(self.alarm_sound)
            else:
                cv2.putText(frame, "No Object", (10, height - 20), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                stop_sound(self.alarm_sound)

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


if __name__ == '__main__':
    video_object_detection = VideoObjectDetection()
    video_object_detection.start_video_analysis()
