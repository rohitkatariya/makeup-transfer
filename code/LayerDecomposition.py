import cv2
import numpy as np
import pdb
from  Warping import Warping
import math
import config
from skimage.exposure import match_histograms
class LayerDecomposition:
    def layer_decomposition(self, image_bgr):
        this_image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
        this_image_lab = np.float32(this_image_lab)
        structure_layer = cv2.bilateralFilter(this_image_lab[:,:,0],8,25,25)
        detail_layer = this_image_lab[:,:,0] - structure_layer
        color_layer = this_image_lab[:, :, 1:3]
        layersImage = {
            "structure_layer":structure_layer,
            "detail_layer": detail_layer, 
            "color_layer":color_layer
            }
        return layersImage
    def layer_recomposition(self,dict_layers):
        final_image = np.zeros((dict_layers['structure_layer'].shape[0] , dict_layers['structure_layer'].shape[1],3),np.uint8)
        final_image[:,:,0] = dict_layers['detail_layer'] + dict_layers['structure_layer']
        final_image[:,:,1:3] = dict_layers['color_layer']
        final_image = cv2.cvtColor(np.uint8( final_image), cv2.COLOR_Lab2BGR)
        return final_image

    def apply_layer_highlights_transfer(self,dict_layers_example,dict_layers_subject):
        # layersImage = {
        #     "structure_layer":structure_layer,
        #     "detail_layer": detail_layer, 
        #     "color_layer":color_layer
        #     }
        # pdb.set_trace()
        # grad_Es = cv2.Sobel(dict_layers_example['structure_layer'],cv2.CV_32F ,0,1,ksize=5) + cv2.Sobel(dict_layers_example['structure_layer'],cv2.CV_32F ,1,0,ksize=5)  # change this to gradient + laplacian
        # grad_Is = cv2.Sobel(dict_layers_subject['structure_layer'],cv2.CV_32F, 0,1,ksize=5)  + cv2.Sobel(dict_layers_subject['structure_layer'],cv2.CV_32F, 1,0,ksize=5)
        grad_Es = cv2.laplacian(dict_layers_example['structure_layer'],cv2.CV_32F )
        grad_Is = cv2.laplacian(dict_layers_subject['structure_layer'],cv2.CV_32F)
        gauss_Is = cv2.GaussianBlur(dict_layers_subject['structure_layer'], (5,5), 1.5 )
        size = grad_Is.shape
        beta = self.dest_landmarkGenerator.masks_for_beta['others']
        grad_Rs = np.zeros(grad_Is.shape)
        for i in range(size[0]):
            for j in range(size[1]):
                if(abs(grad_Es[i, j])*beta[i, j] > abs(grad_Is[i, j])):
                    grad_Rs[i, j] = grad_Es[i, j] 
                else:
                    grad_Rs[i, j] = grad_Is[i, j]
        R_struct =  grad_Rs + gauss_Is
        return R_struct

    def other_layer_transfers(self,warp_obj):
        # Layer Decomposition
        layers_example_img = self.layer_decomposition(warp_obj.transfered_image)
        layers_destination_img = self.layer_decomposition(warp_obj.orig_image) 
        p = [layers_example_img,layers_destination_img]
        
        # Skin Detail Transfer
        sd_eg_wt = 0.8
        sd_src_wt = 0.2
        other_skin_detail_final = sd_eg_wt * layers_example_img['detail_layer'] + sd_src_wt * layers_destination_img['detail_layer']
        
        # Skin Color Transfer
        lambda_color = self.lambda_color
        other_skin_color_final = lambda_color * layers_example_img['color_layer'] + (1-lambda_color) * layers_destination_img['color_layer']

        # Highlight and Shading Transfer
        Rs_others = layers_destination_img['structure_layer']
        # Rs_others = self.apply_layer_highlights_transfer(layers_example_img,layers_destination_img)
        
        R_layers = {
            "structure_layer":Rs_others,
            "detail_layer": other_skin_detail_final, 
            "color_layer":other_skin_color_final
            }

        return self.layer_recomposition(R_layers)
    
    # def gaussian_np_mat():


    def get_Gaussian(x, a , c_sq ):
        return a*np.exp(- np.square(x)/(2*c_sq))
    
    def get_Gaussian_num(x, a , c_sq ):
        return a*math.exp(- (x**2)/(2*c_sq))

    def get_euclid(p, q):
        ex = float(p[0]-q[0])
        ey = float(p[1]-q[1])
        return math.sqrt( 1.*ex*ex + 1.*ey*ey )
    def lip_makeup_transfer2(self):
        print("applying makeup transfer")
        orig_image = self.warp_obj.orig_image
        transfered_image = self.warp_obj.transfered_image
        mask_lips = self.warp_obj.bitmasks['lips']
        # fetch only lips area
        lips_landmark_points = self.dest_landmarkGenerator.landmarks_dict['lips']
        (x,y,w,h) = Warping.get_bounding_rect(lips_landmark_points)
        orig_image_lips = orig_image[y:y+h, x:x+w,:].copy()
        transfered_image_lips = transfered_image[y:y+h, x:x+w,:].copy()
        mask_lips_cropped = mask_lips[y:y+h, x:x+w].copy()
        # LAB Conversion
        orig_image_lips_lab = cv2.cvtColor(orig_image_lips, cv2.COLOR_BGR2Lab)
        transfered_image_lips_lab = cv2.cvtColor(transfered_image_lips, cv2.COLOR_BGR2Lab)

        orig_l = orig_image_lips_lab[:,:,0]
        trans_l = transfered_image_lips_lab[:,:,0]

        # histogram equalization
        orig_image_lips_eq = cv2.equalizeHist(orig_l)
        transfered_image_lips_eq = cv2.equalizeHist(trans_l)
        

        lips_Result = orig_image_lips_eq.copy()
        lips_Result = np.zeros_like(orig_image_lips_lab)
        c_sq =  config.gaussian_lips_c_sq
        a = config.gaussian_lips_a
        dist_window = 5
        for p_i in range(lips_Result.shape[0]):
            for p_j in range(lips_Result.shape[1]):
                if mask_lips_cropped[p_i][p_j]==0:
                    continue
                Ip = float(orig_image_lips_eq[p_i][p_j])
                best_val = -1
                for q_i in range(max(0,p_i-dist_window),min( lips_Result.shape[0] , p_i+dist_window)):
                    for q_j in range(max(0,p_j-dist_window),min( lips_Result.shape[1] , p_j+dist_window)):
                        if mask_lips_cropped[q_i][q_j]==0:
                            continue
                        Eq = float(transfered_image_lips_eq[q_i][q_j])
                        dist_pixels = LayerDecomposition.get_euclid( (p_i,p_j), (q_j,q_j))
                        G_d = LayerDecomposition.get_Gaussian_num( dist_pixels/math.sqrt(2*dist_window) , a , c_sq )
                        # print('deq',(Eq-Ip)/255.)
                        G_l = LayerDecomposition.get_Gaussian_num((Eq-Ip)/255. , a , c_sq )
                        this_val = (G_d*G_l)
                        if best_val<this_val:
                            best_val= this_val
                            best_point = (q_i,q_j)
                if best_val == -1:
                    print("Not found lip point")
                lips_Result[p_i,p_j,:] = transfered_image_lips_lab[best_point[0],best_point[1],:] 
        # pdb.set_trace()

        # lips_Result[:,:,0] = match_histograms(lips_Result[:,:,0], orig_l, multichannel=False)

        lips_Result_BGR = cv2.cvtColor(lips_Result, cv2.COLOR_Lab2BGR)        
        lips_Result_full = orig_image.copy()
        lips_Result_full[y:y+h, x:x+w,:] = lips_Result_BGR#[y:y+h, x:x+w,:]
        # cv2.imshow( 'mask_lips_cropped' ,mask_lips_cropped)
        # cv2.imshow( 'lips_Result_full' ,lips_Result_full)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # pdb.set_trace()
        # bounding_rect  = self.dest_landmarkGenerator.
        return lips_Result_full
        # lips_bounding_box = None #self.dest_landmarkGenerator.
    

    def lip_makeup_transfer(self):
        print("applying makeup transfer")
        orig_image = self.warp_obj.orig_image.copy()
        # return orig_image
        transfered_image = self.warp_obj.transfered_image.copy()
        orig_image_lips_lab = cv2.cvtColor(orig_image, cv2.COLOR_BGR2Lab)
        transfered_image_lips_lab = cv2.cvtColor(transfered_image, cv2.COLOR_BGR2Lab)
        
        orig_l = orig_image_lips_lab[:,:,0]
        trans_l = transfered_image_lips_lab[:,:,0]
        # histogram equalization
        orig_l_eq = cv2.equalizeHist(orig_l)
        trans_l_eq = cv2.equalizeHist(trans_l)
        
        mask_lips = self.warp_obj.bitmasks['lips']
        # # fetch only lips area
        # lips_landmark_points = self.dest_landmarkGenerator.landmarks_dict['lips']
        # (x,y,w,h) = Warping.get_bounding_rect(lips_landmark_points)
        # orig_image_lips = orig_image[y:y+h, x:x+w,:].copy()
        # transfered_image_lips = transfered_image[y:y+h, x:x+w,:].copy()
        # mask_lips_cropped = mask_lips[y:y+h, x:x+w].copy()
        
        # LAB Conversion
        

        
        # histogram equalization
        # orig_image_lips_eq = cv2.equalizeHist(orig_l)
        # transfered_image_lips_eq = cv2.equalizeHist(trans_l)
        

        lips_Result = transfered_image_lips_lab.copy()
        # lips_Result = np.zeros_like(transfered_image_lips_lab)
        c_sq =  config.gaussian_lips_c_sq
        a = config.gaussian_lips_a
        dist_window = 5
        for p_i in range(lips_Result.shape[0]):
            for p_j in range(lips_Result.shape[1]):
                if mask_lips[p_i][p_j]==0:
                    continue
                Ip = float(orig_l_eq[p_i][p_j])
                best_val = -1
                for q_i in range(max(0,p_i-dist_window),min( lips_Result.shape[0] , p_i+dist_window)):
                    for q_j in range(max(0,p_j-dist_window),min( lips_Result.shape[1] , p_j+dist_window)):
                        if mask_lips[q_i][q_j]==0:
                            continue
                        Eq = float(trans_l_eq[q_i][q_j])
                        dist_pixels = LayerDecomposition.get_euclid( (p_i,p_j), (q_j,q_j))
                        G_d = LayerDecomposition.get_Gaussian_num( dist_pixels/math.sqrt(2*dist_window) , a , c_sq )
                        # print('deq',(Eq-Ip)/255.)
                        G_l = LayerDecomposition.get_Gaussian_num((Eq-Ip)/255. , a , c_sq )
                        this_val = (G_d*G_l)
                        if best_val<this_val:
                            best_val= this_val
                            best_point = (q_i,q_j)
                if best_val == -1:
                    print("Not found lip point")
                lips_Result[p_i,p_j,:] = transfered_image_lips_lab[best_point[0],best_point[1],:] 
        # pdb.set_trace()

        # lips_Result[:,:,0] = match_histograms(lips_Result[:,:,0], orig_image_lips_lab[:,:,0], multichannel=False)

        lips_Result_BGR = cv2.cvtColor(lips_Result, cv2.COLOR_Lab2BGR)        
        # lips_Result_full = orig_image.copy()
        # lips_Result_full[y:y+h, x:x+w,:] = lips_Result_BGR#[y:y+h, x:x+w,:]
        # cv2.imshow( 'mask_lips_cropped' ,mask_lips_cropped)
        # cv2.imshow( 'lips_Result_full' ,lips_Result_full)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # pdb.set_trace()
        # bounding_rect  = self.dest_landmarkGenerator.
        return lips_Result_BGR
        # lips_bounding_box = None #self.dest_landmarkGenerator.

    def combine_multiple_layers(list_layers):
        sum_image = np.zeros(list_layers[0].shape,dtype=np.uint8)
        # for layer_this in list_layers:
        #     sum_image = cv2.add(sum_image , layer_this)
        # return sum_image
        # for i in range(3):
        #     cv2.imshow(str(i),list_layers[i])
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows()
        for layer_this in list_layers:
            
            grayscale_image = cv2.cvtColor(sum_image,cv2.COLOR_BGR2GRAY)
            _,bitmask_sum_image = cv2.threshold(grayscale_image,0,255,cv2.THRESH_BINARY_INV) 
            
            layer_this_masked = np.zeros(sum_image.shape,dtype=np.uint8)
            for i in range(3):
                layer_this_masked[:,:,i]=cv2.bitwise_and(bitmask_sum_image,layer_this[:,:,i])
            sum_image = cv2.add(sum_image,layer_this_masked)
           
        return sum_image

    def __init__(self,warp_obj,dest_landmarkGenerator, lambda_color=0.99):
        self.lambda_color = lambda_color
        self.dest_landmarkGenerator = dest_landmarkGenerator
        self.warp_obj = warp_obj

        # Skin and other image part:
        image_other_unmasked = self.other_layer_transfers(warp_obj)
        
        # cv2.imshow('self.warp_obj.bitmasks',self.warp_obj.bitmasks['others'])
        other_image_masked = Warping.applyBitmaskColor(self.warp_obj.bitmasks['others'],image_other_unmasked)
        # cv2.imshow('other_image_masked',image_other_unmasked)
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows()
        # Lip Makeup 
        image_lips_unmasked = self.lip_makeup_transfer()
        lips_image_masked = Warping.applyBitmaskColor(self.warp_obj.bitmasks['lips'],image_lips_unmasked)
        

        # final_image_lips = self.layer_recomposition(layers_destination_img['lips'])
        # background
        image_bg_unmasked = warp_obj.orig_image
        background_image_masked = Warping.applyBitmaskColor(self.warp_obj.bitmasks['background'],image_bg_unmasked)
        # pdb.set_trace()
        # cv2.imshow('other_image_masked',other_image_masked)
        # cv2.imshow('lips_image_masked',lips_image_masked)
        # cv2.imshow('image_bg_unmasked',image_bg_unmasked)
        # cv2.imshow('background_image_masked',background_image_masked)

        # Combining 3 images
        combined_layer = LayerDecomposition.combine_multiple_layers([background_image_masked,other_image_masked,lips_image_masked])
        combined_layer = np.uint8(combined_layer)
        # cv2.imwrite(self.dest_landmarkGenerator.output_dir+"combined_layer.jpg",combined_layer)
        # cv2.imwrite(self.landmarks_obj.output_dir+'mesh_image.jpg', mesh_image)
        
        # cv2.imshow('orig_image',self.warp_obj.orig_image)
        # cv2.imshow('transfered_image',self.warp_obj.transfered_image)
        # cv2.imshow('combined_layer',combined_layer)
        
        cv2.imwrite(self.dest_landmarkGenerator.output_dir+"combined_layer.jpg",combined_layer)
        cv2.imwrite(self.dest_landmarkGenerator.output_dir+"transfered_image.jpg",self.warp_obj.transfered_image)
        

        # cv2.waitKey(0) 
        # cv2.destroyAllWindows()
        # print(Rd)



    

