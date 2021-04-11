import cv2
import numpy as np
import pdb
class LayerDecomposition:
    def layer_decomposition(self, image_bgr):
        this_image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
        this_image_lab = np.float32(this_image_lab)
        structure_layer = cv2.bilateralFilter(this_image_lab[:,:,0],8,75,75)
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

    def __init__(self,warp_obj,lambda_color=0.8):
        

        # Layer Decomposition
        layers_example_img = {layer_key: self.layer_decomposition(layer_img) for layer_key,layer_img in warp_obj.layers_transferred_other.items()}
        layers_destination_img = {layer_key: self.layer_decomposition(layer_img) for layer_key,layer_img in warp_obj.original_layers.items()}
        # for k,v in layers_example_img['other'].items():
        #     print(v.dtype)
            # if k!='color_layer':
            #     cv2.imshow(k,v)
        # pdb.set_trace()
        # other_img = self.layer_recomposition(layers_example_img['other'])
        # cv2.imshow('other_img',other_img)
        # Skin Detail Transfer
        sd_eg_wt = 0.8
        sd_src_wt = 0.2
        other_skin_detail_final = sd_eg_wt * layers_example_img['other']['detail_layer'] + sd_src_wt * layers_destination_img['other']['detail_layer']
        
        # Skin Color Transfer
        # lambda_color = 0.2
        other_skin_color_final = lambda_color * layers_example_img['other']['color_layer'] + (1-lambda_color) * layers_destination_img['other']['color_layer']

        final_image_other = self.layer_recomposition({
            'detail_layer':other_skin_detail_final,
            'color_layer':other_skin_color_final,
            'structure_layer':layers_destination_img['other']['structure_layer']
        })
        # Highlight and Shading Transfer
        final_image_lips = self.layer_recomposition(layers_destination_img['lips'])
        final_image_bg = self.layer_recomposition(layers_destination_img['background'])
        cv2.imshow('final_image_other',final_image_other+final_image_lips+final_image_bg)
        # print(warp_obj.orig_img)
        cv2.imshow('otig',warp_obj.orig_img)
        # pdb.set_trace()
        # Lip Makeup
            
        # Rd = 0.8 * image_layers[0]['detail_layer'] + 0.2 * image_layers[1]['detail_layer']
        # Rd = np.uint8(Rd)
        # Cd = (1-lambda_color) * image_layers[0]['color_layer'] + lambda_color * image_layers[1]['color_layer']
        # Cd = np.uint8(Cd)
        
        # cv2.imshow('Rd',Rd)
        # for i in range(2):
        #     cv2.imshow('Cd{}'.format(i),Cd[:,:,i])
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
        # print(Rd)



    # def __init__(self,image_landmarks_list,lambda_color=0.8):
    #     image_layers = []
    #     for image_idx in range(2):
    #         this_image = image_landmarks_list[image_idx].frame_orig
    #         this_image_lab = cv2.cvtColor(this_image, cv2.COLOR_BGR2Lab)

    #         structure_layer = cv2.bilateralFilter(this_image_lab[:,:,0],8,75,75)
    #         detail_layer = this_image_lab[:,:,0] - structure_layer
    #         color_layer = this_image_lab[:, :, 1:3]
    #         layersImage = {
    #             "structure_layer":structure_layer,
    #             "detail_layer": detail_layer, 
    #             "color_layer":color_layer
    #             }
    #         image_layers.append(layersImage)
    #         # cv2.imshow('structure_layer{}'.format(image_idx),structure_layer)
    #         # cv2.imshow('detail_layer{}'.format(image_idx),detail_layer)
    #         # for i in range(2):
    #         #     cv2.imshow('color_layer_{}_{}'.format(i,image_idx),color_layer[:,:,i])
            
    #     Rd = 0.8 * image_layers[0]['detail_layer'] + 0.2 * image_layers[1]['detail_layer']
    #     Rd = np.uint8(Rd)
    #     Cd = (1-lambda_color) * image_layers[0]['color_layer'] + lambda_color * image_layers[1]['color_layer']
    #     Cd = np.uint8(Cd)
        
    #     cv2.imshow('Rd',Rd)
    #     for i in range(2):
    #         cv2.imshow('Cd{}'.format(i),Cd[:,:,i])
    #     cv2.waitKey(0) 
    #     cv2.destroyAllWindows()
    #     print(Rd)

