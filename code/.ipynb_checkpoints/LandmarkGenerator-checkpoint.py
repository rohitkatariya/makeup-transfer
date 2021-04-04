import sys
import os
import dlib
import glob
from skimage import io
import numpy as np
import random
import cv2



class LandmarkGenerator:
    color_idx = {}
    color_idx[0]=[255,0,0]       # red
    color_idx[1]=[255,165,0]     # orange
    color_idx[2]=[255,255,0]     # yellow
    color_idx[3]=[0,255,0]       # green
    color_idx[4]=[0,0,255]       # blue
    color_idx[5]=[75,0,130]      # indigo
    color_idx[6]=[238,130,238]   # violet
    color_idx[7]=[0,0,0]         # black
    color_idx[8]=[127,127,127]   # grey
    color_idx[9]=[255,255,255]   # white
    def split_landmarks(self):
        self.landmarks_list = ['eyes_r','eyes_l','brows_r','brows_l','nose','lips','mouth','others']
        self.landmarks_dict = {k:[] for k in self.landmarks_list}
        self.landmarks_dict_rev = {}
        for i in range(len(self.landmarks)):
            if (i>=17 and i<=21):
                self.landmarks_dict['brows_l'].append( i)
                self.landmarks_dict_rev[i] = 'brows_l'
            elif (i>=22 and i<= 26):
                self.landmarks_dict['brows_r'].append( i)
                self.landmarks_dict_rev[i] = 'brows_r'
            elif (i>=36 and i<=41):
                self.landmarks_dict['eyes_l'].append( i)
                self.landmarks_dict_rev[i] = 'eyes_l'
            elif (i>=42 and i<= 47):
                self.landmarks_dict['eyes_r'].append( i)
                self.landmarks_dict_rev[i] = 'eyes_r'
            elif (i>=27 and i<= 35):
                self.landmarks_dict['nose'].append( i)
                self.landmarks_dict_rev[i] = 'nose'
            elif (i>=48 and i<= 58):
                self.landmarks_dict['lips'].append( i)
                self.landmarks_dict_rev[i] = 'lips'
            elif (i>=60 and i<= 67):
                self.landmarks_dict['mouth'].append( i)
                self.landmarks_dict_rev[i] = 'mouth'
            else:
                self.landmarks_dict['others'].append( i)
                self.landmarks_dict_rev[i] = 'others'
        
    def create_landmarks_frame(self):
        landmarks_frame = self.frame_orig.copy()
        
        for num in range(self.shape.num_parts):
            cv2.putText(landmarks_frame, '{}'.format(num), (self.shape.parts()[num].x, self.shape.parts()[num].y), self.font, 
                       self.fontScale, self.color, self.thickness, cv2.LINE_AA)  
        self.landmarks_frame = landmarks_frame
                
    def get_landmarks(self):
        frame_orig = cv2.imread(self.input_image_path)
        
        detected_rect = self.detector(frame_orig, 0)
        # print(detected_rect)
        shape = self.predictor(frame_orig, detected_rect[0]) # 2nd argument is the face number. 0 because we want the first(largest face)
        landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
        self.landmarks = landmarks
        self.shape = shape
        self.detected_rect = detected_rect
        self.frame_orig=frame_orig
        
    
    def __init__(self,predictor,input_image_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = predictor
        self.input_image_path = input_image_path
        
        # font
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        self.org = (50, 50)

        # fontScale
        self.fontScale = 0.8

        # Blue color in BGR
        self.color = (255, 0, 0)

        # Line thickness of 2 px
        self.thickness = 1
        
        self.get_landmarks()
        
        self.split_landmarks()
        
        self.create_landmarks_frame()