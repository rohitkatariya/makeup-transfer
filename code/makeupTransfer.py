#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dlib
import cv2
import os
from LandmarkGenerator import LandmarkGenerator
import Triangulation
import config
import numpy as np

from matplotlib import pyplot as plt
from skimage.io import imread, imsave
import random
from imp import reload
import Warping
import LayerDecomposition
# In[130]:


list_images = os.listdir(config.input_dir)
print(list(enumerate(list_images)))
list_images = [list_images[4]]


# In[131]:


# list_images = [list_images[4]]


# In[132]:


predictor_path = './../data/shape_predictor_81_face_landmarks.dat'
shape_predictor = dlib.shape_predictor(predictor_path)


# In[133]:


reload(Triangulation)


# In[134]:


input_image_path = random.choice(list_images)


# In[135]:
def resize_img(image):
    stretch_near = cv2.resize(image, (540, int(540. * image.shape[1]/image.shape[0])),
                interpolation = cv2.INTER_NEAREST)
    return stretch_near

for input_image_path in list_images:
    print(input_image_path)
    src_landmarkGenerator = LandmarkGenerator(shape_predictor,config.input_dir+"example.jpg")
    dest_landmarkGenerator = LandmarkGenerator(shape_predictor,config.input_dir+input_image_path)
    rd_idx=0
    warp_obj = Warping.Warping(src_landmarkGenerator,dest_landmarkGenerator)
    LayerDecomposition.LayerDecomposition(warp_obj)
