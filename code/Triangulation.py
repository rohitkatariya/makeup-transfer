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

class Triangulation:
    def computeTriangles(self):
        rect = cv2.boundingRect(self.landmarks_obj.image_convex_hull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(self.landmarks_obj.landmarks)
        triangles = subdiv.getTriangleList()
        self.triangles = np.array(triangles, dtype=np.int32)

    def get_landmark_index(self, this_pt):
        for l_idx in range(len(self.landmarks_obj.landmarks)):
            this_ldmark = self.landmarks_obj.landmarks[l_idx]
            if (this_ldmark[0] == this_pt[0]) and  (this_ldmark[1] == this_pt[1]):
                return l_idx
        return -1
    
    def getMeshIndices(self):
        mesh_indices = []
        self.triangle_indices = []
        print("computing regions")
        for t in self.triangles:
            tr_pts =[]
            tr_indices = []
            for i in range(3):
                this_pt = (t[2*i],t[2*i+1])
                tr_pts.append(  this_pt )
                this_pt_idx = self.get_landmark_index( this_pt )
                tr_indices.append(this_pt_idx)
            tr_region = self.get_triangle_class(tr_indices)
            self.triangle_indices.append((tr_indices,tr_region))
        # self.save_mesh_img()
    
    def get_triangle_class(self,tr_indices):
        region_counts = [0,0,0]
        for point_idx in tr_indices:
            this_point_class = self.landmarks_obj.landmarks_dict_rev[point_idx]
            # print(this_point_class)
            for i in range(len(self.landmarks_obj.image_regions)) :
                if this_point_class in self.landmarks_obj.image_regions[i]:
                    region_counts[i]+=1
        
        if region_counts[0]>0:
            region = 0
        elif region_counts[1]>0 :
            region = 1
        else:
            region =2
        # print(region_counts, '->',region)
        return region


    def save_mesh_img(self):
        mesh_image = self.landmarks_obj.frame_orig.copy()
        for (t_idx,tr_region) in self.triangle_indices:
            tr_pts = [self.landmarks_obj.landmarks[ t_idx[z] ] for z in range(3)]
            for i in range(3):
                cv2.line(mesh_image, tr_pts[i], tr_pts[(i+1)%3], (0, 0, 255), 1)
        cv2.imwrite(self.landmarks_obj.output_dir+'mesh_image.jpg', mesh_image)
        # cv2.imshow(self.landmarks_obj.output_dir+'mesh_image.jpg', mesh_image)
        # # json.dumps()
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows() 
        json.dump(self.triangle_indices, open(self.landmarks_obj.output_dir+'mesh.json', "w") , indent = 4)

    def __init__(self,landmarks_obj):
        self.landmarks_obj = landmarks_obj
        self.computeTriangles()
        self.getMeshIndices()
        self.save_mesh_img()