import random
import cv2
import numpy as np
import pdb
import config
import time

class Warping:
    def get_bounding_rect(triangle_points):
        triangle_arr = np.array(triangle_points, np.int32)
        (x,y,w,h) = cv2.boundingRect(triangle_arr)
        return (x,y,w,h)
    
    def get_masked_warped_triangle(warp_image_rect, main_image_rect):
        # return warp_image_rect
        main_image_rect_gray = cv2.cvtColor(main_image_rect, cv2.COLOR_BGR2GRAY)
        _,binary_thresh = cv2.threshold(main_image_rect_gray,0,255,cv2.THRESH_BINARY_INV)
        masked_for_warp_output = np.zeros_like(warp_image_rect)
        for i in range(3):
            masked_for_warp_output[:,:,i] = cv2.bitwise_and(binary_thresh,warp_image_rect[:,:,i])
        # cv2.imshow('warp_image_rect',warp_image_rect)
        # cv2.imshow('binary_thresh',binary_thresh)
        # cv2.imshow('result',masked_for_warp_output+main_image_rect)
        # cv2.imshow('result_prev',warp_image_rect+main_image_rect)
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows() 
        return masked_for_warp_output

    def getLayerBitmaskColor(this_color_img,thresh_type):
        grayscale_image = cv2.cvtColor(this_color_img,cv2.COLOR_BGR2GRAY)
        _,bitmask_other = cv2.threshold(grayscale_image,0,255,thresh_type) 
        return bitmask_other
    
    def applyBitmaskColor(bitmask,img_color):
        new_image = np.zeros(img_color.shape,dtype=np.uint8)
        for i in range(3):
            new_image[:,:,i]=cv2.bitwise_and(bitmask,img_color[:,:,i])
        return new_image
    def warp_image(self):
        list_warped_images = [this_landmarks_obj.frame_orig.copy()  for this_landmarks_obj in self.list_landmarks_obj] 
        self.orig_img = self.list_landmarks_obj[1].frame_orig
        transferred_background = np.zeros(self.list_landmarks_obj[1].frame_orig.shape,np.uint8)
        transferred_other = np.zeros(self.list_landmarks_obj[1].frame_orig.shape,np.uint8)
        transferred_lips = np.zeros(self.list_landmarks_obj[1].frame_orig.shape,np.uint8)
        mask_image = np.zeros(self.list_landmarks_obj[1].frame_orig.shape,np.uint8)
        for this_triangle_indices,triangle_region in self.dest_triangle_indices:
            this_bounding_rect_list = []
            for idx_image in range(2):
                this_landmark_obj = self.list_landmarks_obj[idx_image]
                this_triangle = [ this_landmark_obj.landmarks[tr_idx] for tr_idx in this_triangle_indices]
                (x,y,w,h) = Warping.get_bounding_rect(this_triangle)
                this_rect = list_warped_images[idx_image][y:y+h, x:x+w,:]
                triangle_points_cropped = np.array([ (tr_pt[0] - x , tr_pt[1] - y ) for tr_pt in this_triangle],dtype=np.int32)
                cropping_mask = np.zeros((h,w),np.uint8)
                cv2.fillConvexPoly(cropping_mask,triangle_points_cropped,255)
                cropped_triangle = np.zeros_like(this_rect)
                for i in range(3):
                    cropped_triangle[:,:,i] = cv2.bitwise_and(cropping_mask,this_rect[:,:,i])
                this_bounding_rect_list.append((triangle_points_cropped,cropped_triangle,(x,y,w,h)))
            affineM =  cv2.getAffineTransform(np.float32(this_bounding_rect_list[0][0]),np.float32(this_bounding_rect_list[1][0]))
            warped_triangle = cv2.warpAffine(this_bounding_rect_list[0][1],affineM,
                                    (this_bounding_rect_list[1][1].shape[1],this_bounding_rect_list[1][1].shape[0]), flags=cv2.INTER_NEAREST
                                    )
            (start_x,start_y,tr_w,tr_h) = this_bounding_rect_list[1][2]
            triangle_points_cropped=[ tuple(z) for z in triangle_points_cropped]
            masked_warped_triangle = Warping.get_masked_warped_triangle(warped_triangle, mask_image[start_y:start_y+h,start_x:start_x+w])
            mask_image[start_y:start_y+h,start_x:start_x+w] = cv2.add(mask_image[start_y:start_y+h,start_x:start_x+w] , masked_warped_triangle)
            
            if triangle_region==0:
                masked_warped_triangle = Warping.get_masked_warped_triangle(warped_triangle,transferred_background[start_y:start_y+h,start_x:start_x+w])
                warped_triangle_added = cv2.add(transferred_background[start_y:start_y+h,start_x:start_x+w] , masked_warped_triangle)
                transferred_background[start_y:start_y+h,start_x:start_x+w] = warped_triangle_added

                masked_warped_triangle = Warping.get_masked_warped_triangle(warped_triangle,transferred_other[start_y:start_y+h,start_x:start_x+w])
                warped_triangle_added = cv2.add(transferred_other[start_y:start_y+h,start_x:start_x+w] , masked_warped_triangle)
                transferred_other[start_y:start_y+h,start_x:start_x+w] = warped_triangle_added
                
            if triangle_region==1:
                masked_warped_triangle = Warping.get_masked_warped_triangle(warped_triangle,transferred_lips[start_y:start_y+h,start_x:start_x+w])
                warped_triangle_added = cv2.add(transferred_lips[start_y:start_y+h,start_x:start_x+w] , masked_warped_triangle)
                transferred_lips[start_y:start_y+h,start_x:start_x+w] = warped_triangle_added
                
                masked_warped_triangle = Warping.get_masked_warped_triangle(warped_triangle,transferred_background[start_y:start_y+h,start_x:start_x+w])
                warped_triangle_added = cv2.add(transferred_background[start_y:start_y+h,start_x:start_x+w] , masked_warped_triangle)
                transferred_background[start_y:start_y+h,start_x:start_x+w] = warped_triangle_added

        # cv2.imshow('mask_image1234',mask_image)  
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows() 
        
        # self.layers_transferred_other = {
        #     'others':transferred_other,
        #     'lips':transferred_lips
        # }

        bitmask_other = Warping.getLayerBitmaskColor(transferred_other,cv2.THRESH_BINARY)
        bitmask_lips = Warping.getLayerBitmaskColor(transferred_lips,cv2.THRESH_BINARY) 
        bitmask_background = Warping.getLayerBitmaskColor(transferred_background,cv2.THRESH_BINARY_INV) 
        
        _,bitmask_background_inv = cv2.threshold(bitmask_background,0,255,cv2.THRESH_BINARY_INV) 
        bitmask_other = cv2.bitwise_and(bitmask_other,bitmask_background_inv)
        sum_bitmasks = cv2.add(bitmask_background,bitmask_other)

        _,sum_bitmasks_inv = cv2.threshold(sum_bitmasks,0,255,cv2.THRESH_BINARY_INV) 
        bitmask_lips = cv2.bitwise_and(bitmask_lips,sum_bitmasks_inv)
        sum_bitmasks = cv2.add(sum_bitmasks,bitmask_lips)
        # pdb.set_trace()
        # cv2.imshow('bitmask_lips',np.uint8(bitmask_lips))
        # cv2.imshow('bitmask_other',np.uint8(bitmask_other))
        # cv2.imshow('bitmask_background',np.uint8(bitmask_background))
        # cv2.imshow('transfered_layer',np.uint8(self.transfered_layer))
        
        # # cv2.imshow('background_extracted',np.uint8(Warping.applyBitmaskColor(bitmask_background,self.list_landmarks_obj[1].frame_orig)))
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows()
        
        self.orig_image = self.list_landmarks_obj[1].frame_orig
        self.transfered_image = mask_image
        # bitmask_background


        self.bitmasks = {
            "background" : bitmask_background,
            "others" : bitmask_other,
            "lips" : bitmask_lips
        }
        # background_extracted = Warping.applyBitmaskColor(bitmask_background,self.list_landmarks_obj[1].frame_orig) 
        # other_extracted = Warping.applyBitmaskColor(bitmask_other,self.list_landmarks_obj[1].frame_orig) 
        # lips_extracted = Warping.applyBitmaskColor(bitmask_lips,self.list_landmarks_obj[1].frame_orig) 
        
        # cv2.imshow('background_extracted',np.uint8(background_extracted))
        # cv2.imshow('other_extracted',np.uint8(other_extracted))
        # cv2.imshow('lips_extracted',np.uint8(lips_extracted))
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows()
        
         
        # transfered_image = cv2.medianBlur(transfered_image,3)
        # cv2.imshow('transferred_other',transferred_other)
        # cv2.imshow('transferred_lips',transferred_lips)
        # cv2.imshow('transferred_other_b',cv2.medianBlur(transferred_other,3))
        # cv2.imshow('transferred_lips_b',cv2.medianBlur(transferred_lips,3))
        # face swaping 
        # mask_image_1_gray = cv2.cvtColor(bitmask_background,cv2.COLOR_BGR2GRAY)
        # _,bitmask_other = cv2.threshold(transferred_other,0,255,cv2.THRESH_BINARY) 
        # _,bitmask_lips = cv2.threshold(transferred_lips,0,255,cv2.THRESH_BINARY)
        # _,bitmask_background = cv2.threshold(transferred_background,0,255,cv2.THRESH_BINARY_INV)
        # bit_masked_image = cv2.bitwise_and(list_warped_images[1], list_warped_images[1], mask=bg)
        # transfered_image = cv2.add(bit_masked_image,bitmask_background)
        # transfered_image = cv2.medianBlur(transfered_image,3)
        # print(type(bg))
        # pdb.set_trace()
        # cv2.imshow('bitmask_lips',bitmask_lips)        
        # cv2.imshow('bitmask_background',bitmask_background)  
        # cv2.imshow('bitmask_other',bitmask_other)    
        # background_extracted = cv2.bitwise_and(bitmask_background,self.list_landmarks_obj[1].frame_orig)
        # # cv2.imshow('background_extracted',background_extracted)
        # other_extracted = cv2.bitwise_and(bitmask_other,self.list_landmarks_obj[1].frame_orig)
        # # cv2.imshow('other_extracted',other_extracted)
        # lips_extracted = cv2.bitwise_and(bitmask_lips,self.list_landmarks_obj[1].frame_orig)
        # # cv2.imshow('lips_extracted',lips_extracted)
        
        # self.original_layers = {
        #     "background" : background_extracted,
        #     "others" : other_extracted,
        #     "lips" : lips_extracted
        # }

            
    def __init__(self, src_landmarks_obj,dest_landmarks_obj,dest_triangle_indices):
        self.list_landmarks_obj = [src_landmarks_obj,dest_landmarks_obj]
        self.dest_triangle_indices = dest_triangle_indices
        transfered_img = self.warp_image()
        image_name = dest_landmarks_obj.input_image_path.split('/')[-1].split('.')[0]

        # cv2.imwrite(config.output_dir+'transferred_{}.jpg'.format(image_name), transfered_img)
        