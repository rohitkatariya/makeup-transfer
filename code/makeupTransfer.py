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
import Warping as Warping
import LayerDecomposition

list_images = os.listdir(config.input_dir)
print(list(enumerate(list_images)))
# list_images = [list_images[4]]
list_images = ['XDOG_image.jpg']
# list_images = ['h.jpg']
# list_images = ['color_ja.jpg']
# list_images = ['m.jpg']
# list_images = ['mom2.jpg']
# list_images = ['openmouth.jpg']
# list_images = ['tiger.jpg']
# list_images = ['XDOG_b_t_image.jpg']
list_images = ['XDOG_b_image.jpg']
# list_images = ['base.jpg']

predictor_path = './../data/shape_predictor_81_face_landmarks.dat'
shape_predictor = dlib.shape_predictor(predictor_path)

input_image_path = random.choice(list_images)

def resize_img(image):
    stretch_near = cv2.resize(image, (540, int(540. * image.shape[1]/image.shape[0])),
                interpolation = cv2.INTER_NEAREST)
    return stretch_near

for input_image_path in list_images:
    print(input_image_path)
    src_landmarkGenerator = LandmarkGenerator(shape_predictor,config.input_dir+"example.jpg")
    dest_landmarkGenerator = LandmarkGenerator(shape_predictor,config.input_dir+input_image_path)
    dest_triangulation = Triangulation.Triangulation(dest_landmarkGenerator)
    # my_landmarkGenerator.save_mesh_img()
    rd_idx=0
    warp_obj = Warping.Warping(src_landmarkGenerator,dest_landmarkGenerator,dest_triangulation.triangle_indices)
    LayerDecomposition.LayerDecomposition(warp_obj,dest_landmarkGenerator)
