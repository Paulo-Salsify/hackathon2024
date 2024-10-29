import os
from typing import Any

import cv2
from keras.models import load_model
import numpy as np
from pydantic import BaseModel

from configs import ALARM_SOUND, MAIN_LOG_LEVEL, MAIN_LOG_PATH
from core.logger import logger
from utils import load_sound, play_sound, stop_sound


class SleepInferenceConfigs(BaseModel):
    """
    Default values for closed-eyes inferences
    """
    score: int = 0
    threshold: int = 6
    thicc: int  = 2
    rpred: list[int] = [99]
    lpred: list[int] = [99]


class VideoSleepDetection:
    def __init__(self) -> None:
        self.logging = logger(MAIN_LOG_PATH, MAIN_LOG_LEVEL)
        self.load_configs()

    def load_configs(self) -> None:
        self.load_opencv_kernels()
        self.load_keras_models()
        self.inference_configs = SleepInferenceConfigs()
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        self.alarm_sound = load_sound(ALARM_SOUND)
        self.logging.info('> VideoSleepDetection configs loaded.')


    def video_capture_get(self) -> cv2.VideoCapture:
        """
        Start capturing the video feed.
        """
        # Set configs to allow changing number, if user has more than one camera device source
        device_number = 0
        cap = cv2.VideoCapture(device_number)
        if cap.isOpened() == True:
            self.logging.info('> Video stream open.')
        else:
            msg = "Problem opening video stream."
            self.logging.error(f'> {msg}')
            raise Exception(msg)
    
        return cap
    
    def video_capture_release(self, cap: cv2.VideoCapture) -> None:
        """
        Release the video capture.
        """
        cap.release()
        cv2.destroyAllWindows()    

    def start_video_analyzis(self) -> None:
        cap = self.video_capture_get()
        while(True):
            ret, frame = cap.read()
            height,width = frame.shape[:2]
            
            # convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Haar Cascade object detection in OpenCV to gray frame
            faces = self.face_kernel.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
            left_eye = self.leye_kernel.detectMultiScale(gray)
            right_eye =  self.reye_kernel.detectMultiScale(gray)
            
            # draw black bars top and bottom
            cv2.rectangle(frame, (0,height-50) , (width,height) , (0,0,0) , thickness=cv2.FILLED )
            
            # draw face box
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

            # draw left_eye
            for (x,y,w,h) in left_eye:
                cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,255,0) , 1 )

            # draw right_eye
            for (x,y,w,h) in right_eye:
                cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,255,0) , 1 )

            # take detected RIGHT eye, preprocess and make CNN prediction
            #self.inference_configs.rpred = self.process_eye(frame, right_eye)
            for (x,y,w,h) in right_eye:
                r_eye = frame[y:y+h,x:x+w]
                r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
                r_eye = cv2.resize(r_eye,(100,100))
                r_eye = r_eye/255
                r_eye =  r_eye.reshape(100,100,-1)
                r_eye = np.expand_dims(r_eye,axis=0)
                #self.inference_configs.rpred = self.drowsiness_model.predict_classes(r_eye)
                #self.inference_configs.rpred = self.drowsiness_model.predict_step(r_eye)
                #self.inference_configs.rpred = np.argmax(self.drowsiness_model.predict(r_eye), axis=-1)
                self.inference_configs.rpred = (self.drowsiness_model.predict(r_eye) > 0.5).astype("int32")
                break

            # take detected LEFT eye, preprocess and make CNN prediction
            #self.inference_configs.lpred = self.process_eye(frame, left_eye)
            for (x,y,w,h) in left_eye:
                l_eye = frame[y:y+h,x:x+w]
                l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
                l_eye = cv2.resize(l_eye,(100,100))
                l_eye = l_eye/255
                l_eye =l_eye.reshape(100,100,-1)
                l_eye = np.expand_dims(l_eye,axis=0)
                #self.inference_configs.lpred = self.drowsiness_model.predict_classes(l_eye)
                #self.inference_configs.lpred = self.drowsiness_model.predict_step(l_eye)
                #self.inference_configs.lpred = self.drowsiness_model.predict(l_eye)
                #self.inference_configs.lpred = np.argmax(self.drowsiness_model.predict(l_eye), axis=-1)
                self.inference_configs.lpred = (self.drowsiness_model.predict(l_eye) > 0.5).astype("int32")
                break


            # labeling for frame, if BOTH eyes close, print CLOSED and adding/subtracting from score tally
            #if not faces:
            if len(left_eye) == 0 and len(right_eye) == 0:
                cv2.putText(frame,"Empty",(10,height-20), self.font, 1,(255,255,255),1,cv2.LINE_AA)
                stop_sound(self.alarm_sound)
            elif(self.inference_configs.rpred[0]==0 and self.inference_configs.lpred[0]==0):
                self.inference_configs.score += 1
                cv2.putText(frame,"Closed",(10,height-20), self.font, 1,(255,255,255),1,cv2.LINE_AA)
                # prevent a runaway score beyond threshold
                if self.inference_configs.score > self.inference_configs.threshold + 1:
                    self.inference_configs.score = self.inference_configs.threshold
            else:
                self.inference_configs.score -= 1
                cv2.putText(frame,"Open",(10,height-20), self.font, 1,(255,255,255),1,cv2.LINE_AA)
                stop_sound(self.alarm_sound)
            
            # SCORE HANDLING
            # print current score to screen
            if(self.inference_configs.score < 0):
                self.inference_configs.score = 0   
            cv2.putText(frame,'Drowsiness Score:'+str(self.inference_configs.score),(100,height-20), self.font, 1,(255,255,255),1,cv2.LINE_AA)
            
            # threshold exceedanceq
            if(self.inference_configs.score>self.inference_configs.threshold) and len(left_eye) != 0 and len(right_eye) != 0:
                # save a frame when threshold exceeded and play sound
                cv2.imwrite(os.path.join(os.getcwd(), 'saves', 'closed_eyes_screencap.jpg'), frame)
                try:
                    play_sound(self.alarm_sound)
                except:  # isplaying = False
                    pass
                
                # add red box as warning signal and make box thicker
                if(self.inference_configs.thicc<16):
                    self.inference_configs.thicc += 2
                # make box thinner again, to give it a pulsating appearance
                else:
                    self.inference_configs.thicc -= 2
                    if(self.inference_configs.thicc<2):
                        self.inference_configs.thicc=2
                cv2.rectangle(frame, (0,0), (width,height), (0,0,255), thickness=self.inference_configs.thicc)
                
            # draw frame with all the stuff we have added
            cv2.imshow('frame',frame)
            
            # break the infinite loop when pressing q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # release the capture
        self.video_capture_release(cap)


    def process_eye(self, frame: np.ndarray, eye: list) -> Any:
        for (x,y,w,h) in eye:
            single_eye = frame[y:y+h,x:x+w]
            single_eye = cv2.cvtColor(single_eye,cv2.COLOR_BGR2GRAY)  
            single_eye = cv2.resize(single_eye,(100,100))
            single_eye = single_eye/255
            single_eye =single_eye.reshape(100,100,-1)
            single_eye = np.expand_dims(single_eye,axis=0)
            return (self.drowsiness_model.predict(single_eye) > 0.5).astype("int32")

    def load_opencv_kernels(self) -> None:
        self.face_kernel = cv2.CascadeClassifier(
            os.path.join(os.getcwd(), 'models', 'opencv', 'haarcascade_frontalface_alt.xml')
        )
        self.leye_kernel = cv2.CascadeClassifier(
            os.path.join(os.getcwd(), 'models', 'opencv', 'haarcascade_lefteye_2splits.xml')
        )
        self.reye_kernel = cv2.CascadeClassifier(
            os.path.join(os.getcwd(), 'models', 'opencv', 'haarcascade_righteye_2splits.xml')
        )

    def load_keras_models(self) -> None:
        self.drowsiness_model = load_model(
            os.path.join(os.getcwd(), 'models', 'keras', 'cnn_drowsiness_model.h5')
        )


if __name__ == '__main__':
    video_sleep_detection = VideoSleepDetection()
    video_sleep_detection.start_video_analyzis()
