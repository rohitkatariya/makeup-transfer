import sys
import os
import dlib
import glob
from skimage import io
import numpy as np
import random
import cv2
import config
import json
import pdb
class LandmarkGenerator:
    
    def split_landmarks(self):
        self.landmarks_list = ['others','lips','mouth','nose','brows_l','brows_r','eyes_l','eyes_r']
        self.image_regions = [
            {'others','nose','brows_l','brows_r'},
            {'lips'},
            {'eyes_l','eyes_r','mouth'}
        ]
        self.landmarks_dict = {k:[] for k in self.landmarks_list}
        self.landmarks_dict_rev = {}
        for i in range(len(self.landmarks)):
            if (i>=17 and i<=21):
                self.landmarks_dict['brows_l'].append( self.landmarks[i] )
                self.landmarks_dict_rev[i] = 'brows_l'
            elif (i>=22 and i<= 26):
                self.landmarks_dict['brows_r'].append( self.landmarks[i] )
                self.landmarks_dict_rev[i] = 'brows_r'
            elif (i>=36 and i<=41):
                self.landmarks_dict['eyes_l'].append( self.landmarks[i] )
                self.landmarks_dict_rev[i] = 'eyes_l'
            elif (i>=42 and i<= 47):
                self.landmarks_dict['eyes_r'].append( self.landmarks[i] )
                self.landmarks_dict_rev[i] = 'eyes_r'
            elif (i>=27 and i<= 35):
                self.landmarks_dict['nose'].append( self.landmarks[i] )
                self.landmarks_dict_rev[i] = 'nose'
            elif (i>=48 and i<= 59):
                self.landmarks_dict['lips'].append( self.landmarks[i] )
                self.landmarks_dict_rev[i] = 'lips'
            elif (i>=60 and i<= 67):
                self.landmarks_dict['mouth'].append( self.landmarks[i] )
                self.landmarks_dict_rev[i] = 'mouth'
            else:
                self.landmarks_dict['others'].append( self.landmarks[i] )
                self.landmarks_dict_rev[i] = 'others'
    
    def get_point_class_color(self,this_point_class):
        this_color_idx = self.landmarks_list.index(this_point_class)
        this_color = self.color_idx[this_color_idx]
        return this_color

    def create_landmarks_frame(self):
        landmarks_frame = self.frame_orig.copy()
        
        for num in range(self.shape.num_parts):
            this_point_class = self.landmarks_dict_rev[num]
            this_color = self.get_point_class_color(this_point_class)
            this_point_coordinates = (self.shape.parts()[num].x, self.shape.parts()[num].y)
            # cv2.putText(landmarks_frame, '{}'.format(num), , self.font, 
            #            self.fontScale, this_color, self.thickness, cv2.LINE_AA)  
            cv2.circle(landmarks_frame, this_point_coordinates, 2, this_color, self.thickness)
        self.landmarks_frame = landmarks_frame
        cv2.imwrite(self.output_dir+'landmarks.jpg', landmarks_frame)
        # cv2.imshow('landmarks.jpg', landmarks_frame)
                
    def get_landmarks(self):
        
        detected_rect = self.detector(self.frame_orig, 0)
        # print(detected_rect)
        # this_read_image=cv2.imread("./../data/input/base.jpg")
        shape = self.predictor(self.frame_orig, detected_rect[0]) # 2nd argument is the face number. 0 because we want the first(largest face)
        landmarks = shape.parts()
        landmarks_points = [ (k.x,k.y) for k in landmarks]
        
        new_landmark_points = []
        for landmark_point in landmarks_points:
            this_landmark_point = list(landmark_point)
            for i in range(2):
                if this_landmark_point[i]<0:
                    this_landmark_point[i] = 0
                if this_landmark_point[i]>= self.frame_orig.shape[1-i]: # x and y are reversed of matrices index. i.e. x corrresponds to idex 1
                    # print(i, this_landmark_point,self.frame_orig.shape)
                    # pdb.set_trace()
                    this_landmark_point[i] = self.frame_orig.shape[1-i]-1
            new_landmark_points.append(tuple(this_landmark_point))
        landmarks_points = new_landmark_points

        # print(len(landmarks_points),"landmarks_points")
        self.landmarks = landmarks_points
        self.shape = shape
        self.detected_rect = detected_rect
        # self.frame_orig=frame_orig
        
    def define_color_index(self):
        color_idx = [0]*17
        color_idx[1]=	(255,0,0)	#Red 
        color_idx[2]=	(0,0,255)	#Blue
        color_idx[3]=	(0,128,0)	#Green
        color_idx[4]=	(128,128,0)	#Olive
        color_idx[5]=	(255,255,255)	#White
        color_idx[6]=	(255,255,0)	#Yellow
        color_idx[7]=	(0,255,255)	#Cyan / Aqua
        color_idx[8]=	(255,0,255)	#Magenta / Fuchsia
        color_idx[9]=	(192,192,192)	#Silver
        color_idx[10]=	(128,128,128)	#Gray
        color_idx[11]=	(128,0,0)	#Maroon
        color_idx[12]=	(0,255,0)	#Lime
        color_idx[13]=	(0,0,0)	#Black
        color_idx[14]=	(128,0,128)	#Purple
        color_idx[15]=	(0,128,128)	#Teal
        color_idx[16]=	(0,0,128)	#Navy
        color_idx = color_idx[1:]

        self.color_idx = color_idx
    
    def get_convex_hull(self,landmark_list):
        this_hull_points = np.array(landmark_list) #np.array([ (z.x,z.y) for z in landmark_list])
        this_hull = cv2.convexHull(this_hull_points)
        return this_hull
    
    def createAllMask(self):
        masks = {}
        for landmark_class in self.landmarks_list:
            mask_frame = np.zeros( self.frame_orig_gray.shape)
            landmark_list = self.landmarks_dict[landmark_class]
            
            this_hull = self.get_convex_hull(landmark_list)
            cv2.fillConvexPoly(mask_frame, this_hull, 1)
            masks[landmark_class] = mask_frame
        masks['others'] = masks['others'] - masks['lips'] - masks['brows_l'] - masks['brows_r'] - masks['eyes_l'] - masks['eyes_r']
        masks['lips'] = masks['lips'] - masks['mouth']
        # for mask_name,mask_this in masks.items():
        #     cv2.imshow(mask_name,mask_this)  
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows() 
        self.masks_for_beta = masks

    def createFaceMask(self):
        mask_frame = np.zeros_like( self.frame_orig)
        for landmark_class in self.landmarks_list:
            landmark_list = self.landmarks_dict[landmark_class]
            this_hull = self.get_convex_hull(landmark_list)
            # print(landmark_class,this_hull)
            # cv2.polylines(mask_frame, [this_hull], True, self.get_point_class_color(landmark_class), 3)
            cv2.fillConvexPoly(mask_frame, this_hull, self.get_point_class_color(landmark_class))
        cv2.imwrite(self.output_dir+'mask_frame.jpg', mask_frame)
        # cv2.imshow('mask_frame.jpg', mask_frame)
        #create region masks
        
        region_masks= []
        for region_idx in range(3):
            this_mask_frame = np.zeros_like( self.frame_orig_gray)
            this_region = self.image_regions[region_idx]
            # for point_class in this_region:
            for point_class in this_region:
                landmark_list = self.landmarks_dict[point_class]
                # this_hull_points = np.array([ (z.x,z.y) for z in landmark_list])
                # this_hull = cv2.convexHull(this_hull_points)
                this_hull = self.get_convex_hull(landmark_list)
                cv2.fillConvexPoly(this_mask_frame, this_hull, (255,255,255))
            
            if region_idx == 0 :
                regions_to_mask = [1,2]
            elif region_idx == 1 :
                regions_to_mask = [2]
            else:
                regions_to_mask=[]
            for region_to_mask_idx in regions_to_mask:
                # print('masking {} for {}'.format(region_to_mask_idx,region_idx))
                c3_region = self.image_regions[region_to_mask_idx]
                for point_class in c3_region:
                    landmark_list = self.landmarks_dict[point_class]
                    # this_hull_points = np.array([ (z.x,z.y) for z in landmark_list])
                    # this_hull = cv2.convexHull(this_hull_points)
                    this_hull = self.get_convex_hull(landmark_list)
                    cv2.fillConvexPoly(this_mask_frame, this_hull, (0,0,0))   
            region_masks.append(this_mask_frame)
            ##################################  
            ## INCLUDE EDGE POINTS IN BOTH. ##
            ## WRITE THIS CODE IF REQUIRED. ##
            ##################################  
            self.region_masks = region_masks                                         
            cv2.imwrite(self.output_dir+'region_mask_{}.jpg'.format(region_idx), this_mask_frame)
            face_image_this = cv2.bitwise_and(self.frame_orig, self.frame_orig, mask=this_mask_frame)
            cv2.imwrite(self.output_dir+'region_frame_{}.jpg'.format(region_idx), face_image_this)
            
    def save_mesh_img(self,triangle_indices):
        mesh_image = self.frame_orig.copy()
        
        for (t_idx,tr_region) in triangle_indices:
            
            tr_pts = [self.landmarks[ t_idx[z] ] for z in range(3)]
                
            # print(tr_region) 
            cv2.drawContours(mesh_image, np.array([tr_pts]), 0, self.color_idx[tr_region], -1)
            for i in range(3):
                cv2.line(mesh_image, tr_pts[i], tr_pts[(i+1)%3], self.color_idx[4], 1)
        for num in range(self.shape.num_parts):
            this_point_class = self.landmarks_dict_rev[num]
            this_color = self.get_point_class_color(this_point_class)
            this_point_coordinates = (self.shape.parts()[num].x, self.shape.parts()[num].y)
            # cv2.putText(landmarks_frame, '{}'.format(num), , self.font, 
            #            self.fontScale, this_color, self.thickness, cv2.LINE_AA)  
            cv2.circle(mesh_image, this_point_coordinates, 2, this_color, self.thickness)

        cv2.imwrite(self.output_dir+'mesh_image_idx.jpg', mesh_image)
    
    def __init__(self,predictor,input_image_path):
        self.output_dir = config.output_dir+input_image_path.split('/')[-1].split('.')[0]+"/"
        if os.path.isdir(self.output_dir)== False:
            os.mkdir(self.output_dir)
        self.define_color_index()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = predictor
        self.input_image_path = input_image_path
        self.frame_orig = cv2.imread(self.input_image_path)
        self.frame_orig_gray = cv2.cvtColor(self.frame_orig, cv2.COLOR_BGR2GRAY)
        # font
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        self.org = (50, 50)

        # fontScale
        self.fontScale = 0.8

        # Blue color in BGR
        self.color = (255, 0, 0)

        # Line thickness of 2 px
        self.thickness = -1
        
        self.get_landmarks()
        self.image_convex_hull = self.get_convex_hull(self.landmarks)
        self.split_landmarks()
        
        self.create_landmarks_frame()
        
        self.createAllMask()
        self.createFaceMask()
        # self.triangle_indices = json.load(open("./../data/mesh.json",'r'))