Sources:
cnn_drowsiness_model.h5: https://github.com/abhishek351/Driver-drowsiness-detection-CNN-Keras-OpenCV/blob/main/CNN__model.h5
opencv kernels: https://github.com/opencv/opencv/tree/master/data/haarcascades
Code strattegy inspired from: https://www.christianhaller.me/blog/projectblog/2020-06-27-Drowsiness-Detector/

To run:
Install with:
poetry update
poetry shell
poetry install
create folder/file at (if not exists): saves/logs/main.log

Run:
cd into auto_safe folder
Run with: poetry run python video_sleep_detection.py

Notes:
on 1st run will ask for camera permission and fail. Allow and re-run.
command+c to cancel run
