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
    def warp_image(self):
        list_warped_images = [this_landmarks_obj.frame_orig.copy()  for this_landmarks_obj in self.list_landmarks_obj] 
        self.orig_img = self.list_landmarks_obj[1].frame_orig
        transferred_background = np.zeros(self.list_landmarks_obj[1].frame_orig.shape,np.uint8)
        transferred_other = np.zeros(self.list_landmarks_obj[1].frame_orig.shape,np.uint8)
        transferred_lips = np.zeros(self.list_landmarks_obj[1].frame_orig.shape,np.uint8)
        mask_image = np.zeros(self.list_landmarks_obj[1].frame_orig.shape,np.uint8)
        # this_triangle_indices,triangle_index = random.choice(self.list_landmarks_obj[0].triangle_indices)
        # if True:
        # pdb.set_trace()
        for this_triangle_indices,triangle_region in self.list_landmarks_obj[0].triangle_indices:
            this_bounding_rect_list = []
            for idx_image in range(2):
                this_landmark_obj = self.list_landmarks_obj[idx_image]
                # print(this_triangle)
                this_triangle = [ this_landmark_obj.landmarks[tr_idx] for tr_idx in this_triangle_indices]
                
                # for i in range(3):
                #     cv2.line(list_warped_images[idx_image], this_triangle[i], this_triangle[(i+1)%3], this_landmark_obj.color_idx[5], 1)
                #     pdb.set_trace()
                (x,y,w,h) = Warping.get_bounding_rect(this_triangle)
                # print('(x,y,w,h)',(x,y,w,h),this_triangle)
                # print(list_warped_images[idx_image].shape)
                this_rect = list_warped_images[idx_image][y:y+h, x:x+w,:]
                
                # print('this_rect:',this_rect.shape)
                # cv2.imshow('img_{}'.format(idx_image),this_rect)
                # cv2.waitKey(0) 
                # cv2.destroyAllWindows() 
                # creating mask for triangle
                triangle_points_cropped = np.array([ (tr_pt[0] - x , tr_pt[1] - y ) for tr_pt in this_triangle],dtype=np.int32)
                cropping_mask = np.zeros((h,w),np.uint8)
                cv2.fillConvexPoly(cropping_mask,triangle_points_cropped,255)
                # print('cropping_mask: ',cropping_mask.shape,'this_rect:',this_rect.shape)
                cropped_triangle = np.zeros_like(this_rect)
                for i in range(3):
                    cropped_triangle[:,:,i] = cv2.bitwise_and(cropping_mask,this_rect[:,:,i])
                this_bounding_rect_list.append((triangle_points_cropped,cropped_triangle,(x,y,w,h)))
            affineM =  cv2.getAffineTransform(np.float32(this_bounding_rect_list[0][0]),np.float32(this_bounding_rect_list[1][0]))
            warped_triangle = cv2.warpAffine(this_bounding_rect_list[0][1],affineM,
                                    (this_bounding_rect_list[1][1].shape[1],this_bounding_rect_list[1][1].shape[0])
                                    )
            (start_x,start_y,tr_w,tr_h) = this_bounding_rect_list[1][2]
            triangle_points_cropped=[ tuple(z) for z in triangle_points_cropped]
            for i in range(3):
                # print((mask_image, triangle_points_cropped[i], triangle_points_cropped[(i+1)%3], (0,0,0), 1))
                cv2.line(mask_image, this_triangle[i], this_triangle[(i+1)%3], (0,0,0), 1)
            warped_triangle_added = cv2.add(mask_image[start_y:start_y+h,start_x:start_x+w] , warped_triangle)
            mask_image[start_y:start_y+h,start_x:start_x+w] = warped_triangle_added
            # cv2.imshow('mask_image1234',cv2.medianBlur(mask_image,3))  
            # cv2.waitKey(0) 
            # cv2.destroyAllWindows() 
            if triangle_region==0:
                for i in range(3):
                    cv2.line(transferred_background[start_y:start_y+h,start_x:start_x+w], triangle_points_cropped[i], triangle_points_cropped[(i+1)%3], (0,0,0), 1)
                    cv2.line(transferred_other[start_y:start_y+h,start_x:start_x+w], triangle_points_cropped[i], triangle_points_cropped[(i+1)%3], (0,0,0), 1)
                warped_triangle_added = cv2.add(transferred_background[start_y:start_y+h,start_x:start_x+w] , warped_triangle)
                transferred_background[start_y:start_y+h,start_x:start_x+w] = warped_triangle_added
                warped_triangle_added = cv2.add(transferred_other[start_y:start_y+h,start_x:start_x+w] , warped_triangle)
                transferred_other[start_y:start_y+h,start_x:start_x+w] = warped_triangle_added
                
            if triangle_region==1:
                for i in range(3):
                    cv2.line(transferred_background[start_y:start_y+h,start_x:start_x+w], triangle_points_cropped[i], triangle_points_cropped[(i+1)%3], (0,0,0), 1)
                    cv2.line(transferred_lips[start_y:start_y+h,start_x:start_x+w], triangle_points_cropped[i], triangle_points_cropped[(i+1)%3], (0,0,0), 1)
                warped_triangle_added = cv2.add(transferred_lips[start_y:start_y+h,start_x:start_x+w] , warped_triangle)
                transferred_lips[start_y:start_y+h,start_x:start_x+w] = warped_triangle_added
                warped_triangle_added = cv2.add(transferred_background[start_y:start_y+h,start_x:start_x+w] , warped_triangle)
                transferred_background[start_y:start_y+h,start_x:start_x+w] = warped_triangle_added
            # cv2.imshow('warped_triangle_{}'.format(idx_image),warped_triangle)
            # # time.sleep(0.3)
            # # cv2.imshow('img_{}'.format(idx_image),cropped_triangle)
            # cv2.waitKey(0) 
            # cv2.destroyAllWindows()

        cv2.imshow('mask_image1234',cv2.medianBlur(mask_image,7))  
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
        # cv2.imshow('transferred_background',transferred_background)
        self.layers_transferred_other = {
            'other':transferred_other,
            'lips':transferred_lips
        }

        
        
        # transfered_image = cv2.medianBlur(transfered_image,3)
        # cv2.imshow('transferred_other',transferred_other)
        # cv2.imshow('transferred_lips',transferred_lips)
        # cv2.imshow('transferred_other_b',cv2.medianBlur(transferred_other,3))
        # cv2.imshow('transferred_lips_b',cv2.medianBlur(transferred_lips,3))
        # face swaping 
        # mask_image_1_gray = cv2.cvtColor(bitmask_background,cv2.COLOR_BGR2GRAY)
        _,bitmask_other = cv2.threshold(transferred_other,0,255,cv2.THRESH_BINARY)
        _,bitmask_lips = cv2.threshold(transferred_lips,0,255,cv2.THRESH_BINARY)
        _,bitmask_background = cv2.threshold(transferred_background,0,255,cv2.THRESH_BINARY_INV)
        # bit_masked_image = cv2.bitwise_and(list_warped_images[1], list_warped_images[1], mask=bg)
        # transfered_image = cv2.add(bit_masked_image,bitmask_background)
        # transfered_image = cv2.medianBlur(transfered_image,3)
        # print(type(bg))
        # pdb.set_trace()
        # cv2.imshow('bitmask_lips',bitmask_lips)        
        # cv2.imshow('bitmask_background',bitmask_background)  
        # cv2.imshow('bitmask_other',bitmask_other)    

        

        background_extracted = cv2.bitwise_and(bitmask_background,self.list_landmarks_obj[1].frame_orig)
        # cv2.imshow('background_extracted',background_extracted)

        other_extracted = cv2.bitwise_and(bitmask_other,self.list_landmarks_obj[1].frame_orig)
        # cv2.imshow('other_extracted',other_extracted)

        lips_extracted = cv2.bitwise_and(bitmask_lips,self.list_landmarks_obj[1].frame_orig)
        # cv2.imshow('lips_extracted',lips_extracted)
        
        self.original_layers = {
            "background" : background_extracted,
            "other" : other_extracted,
            "lips" : lips_extracted
        }

        # cv2.imshow('frame_orig',self.list_landmarks_obj[1].frame_orig)
        # # for idx_image in range(2):
        # #     cv2.imshow('full_{}'.format(idx_image),list_warped_images[idx_image])        
        # # cv2.imshow('mask_image_1',mask_image_1)
        # # cv2.imshow('transfered_image',transfered_image)  
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows() 
        
                # cv2.rectangle(list_warped_images[idx_image][:,:,i],(x,y),(x+w , y+h) , this_landmark_obj.color_idx[6], 3 )
        # self.list_warped_images = list_warped_images
        # self.this_bounding_rect_list =this_bounding_rect_list
        # return transfered_image
            
    def __init__(self, src_landmarks_obj,dest_landmarks_obj):
        self.list_landmarks_obj = [src_landmarks_obj,dest_landmarks_obj]
        transfered_img = self.warp_image()
        image_name = dest_landmarks_obj.input_image_path.split('/')[-1].split('.')[0]

        # cv2.imwrite(config.output_dir+'transferred_{}.jpg'.format(image_name), transfered_img)
        