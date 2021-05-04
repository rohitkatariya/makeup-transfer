# Digital Makeup Transfer
Digital Makeup Transfer based on paper by Dong Guo and Terence Sim

### Procedure Followed:
- Landmark creation
  - Facial landmark detection
      - We used a python library function dlib to denote the pixel locations of 81 control points of the face which cover all the face control points. These include eyebrows, eyes, nose, lips and boundary of face. 
  - Landmark point division
      - Here we segmented the landmark points to assign them the region that they represent. <br>
      <img src="data/files_for_md/landmarks.jpg" width = 300>  <br>
  - Create image masks using convex hull
      - We do this to display the different regions of the image.
  
  <img src="data/files_for_md/mask_frame.jpg" width = 300>  <br>
  - We then divide these regions into diffrent sets:
      - C1 - Skin, eyebrows,nose [we call this "others" region]<br>
      <img src="data/files_for_md/region_mask_0.jpg" width = 300> <img src="data/files_for_md/region_frame_0.jpg" width = 300>  <br>
      - C2 lips <br>
      <img src="data/files_for_md/region_mask_1.jpg" width = 300> <img src="data/files_for_md/region_frame_1.jpg" width = 300><br>
      - C3 eyes, mouth cavity <br>
      <img src="data/files_for_md/region_mask_2.jpg" width = 300> <img src="data/files_for_md/region_frame_2.jpg" width = 300><br>
      

- Triangulation / mesh creation
  - Compute triangles using cv2.Subdiv2D
  - Convert triangle points to landmark indices so that this mesh can be universally used in different images
  
  <img src="data/files_for_md/mesh_image.jpg" width = 300> <img src="data/files_for_md/eg_mesh_image.jpg" width = 300>
  

- Assign every triangle the region it belongs to.
  - Each category is further divided into 3 regions on which different transfer method is to be used.
  - Apply affine transformation to resize and reorient corresponding triangles to get a warped image. 
  - Combining different triangles to get the transformed image.
      - created masks for combining different triangles. Here we faced a lot of issues because a lot of mesh lines were showing up. So we developed an innovative way to mask away the common boundary that was getting added multiple times when it is shared among different triangles.<br>
      Image after Afine Transfer:<br>
      <img src="data/files_for_md/transfered_image.jpg" width = 300>  
 

 - Applying Transfers
      - We divided the image into 3 regions: 
          - C1 - Skin, eyebrows,nose [we call this "others" region]
          - C2 lips 
          - C3 eyes, mouth cavity 
      - We apply different transfers to different regions
      

# C1 Region
- Skin transfer
  - Here we divide the images into 3 layer components.
  - we first convert to LAB color space
  - Detail layer, structure layer, color layer
## Structure layer
- structure layer is obtained by applying Bilateral filtering
  - Highlight transfer: here we get try getting gradient with laplacian and sobel operators and we try to add it to smoothened structure layer. 
  <figure>
    <img src="data/files_for_md/dest_structure_layer.jpg" width = 300>  
    <figcaption>Structure Layer</figcaption>
    </figure> 

## Detail layer
- Detail layer is got by subtracting structure layer from lightness component
  - Skin detail transfer - here we use a weighted sum of detailed layer from both example and the subject image
  <figure>
<img src="data/files_for_md/dest_detail_layer.jpg" width = 300>  
<figcaption>Detail Layer</figcaption>
</figure> 

## Color layer
- Color layer is the a,b component of LAB layer
  - Color transfer - where we apply alpha blending of color layers.
<figure>
<img src="data/files_for_md/dest_0color_layer.jpg" width = 300>  
<figcaption>Color Layer</figcaption>
</figure> 
<figure>
<img src="data/files_for_md/dest_1color_layer.jpg" width = 300>  
<figcaption>Color Layer</figcaption>
</figure> 

- Lip tansfers
  - Here we convert the image into LAB color space.
  - we apply histogram equaliztion to both subject and example image.
  - Now for every point p in the subject image's lip region we find the closest lip point in the transfered image using the following similarity function
      {G(|q−p|)G(|E(q)− I(p)|)} where G is the gaussian function.
  - once we get the closest point we copy the color and lightness values to the result image.

- in the eyes mouth cavity and background region no transfers are applied and we use these as it is.

Lips, color, skin detail transfer:<br>
<img src="data/files_for_md/combined_layer.jpg" width=300>  
Lips, color, skin detail, highlights transfer:<br>
<img src="data/files_for_md/highlights.jpg" width=300>      

# Effect of highlight effect

Highlight effect can be better viewed in the following image(leftmost: example image; middle: without highlight transfer; rightmost: with highlights transfer ) 

Here we observe the effect of lighting on both ends of the cheeks where we see that highlights transfer capture the shaddow and gradient changes much better. <br>
<img src="data/files_for_md/highlight_effect.png" width=1200>   

# Other Examples

Destination image; Transferred image 

<img src="data/files_for_md/h.jpg" width=500>  <img src="data/files_for_md/h_combined_layer.jpg" width=500> 

<img src="data/files_for_md/m.jpg" width=500>  <img src="data/files_for_md/m_combined_layer.jpg" width=500> 

<img src="data/files_for_md/eyes.jpg" width=500>  <img src="data/files_for_md/eyes_combined_layer.jpg" width=500> 
<img src="data/files_for_md/man.jpg" width=500>  <img src="data/files_for_md/man_combined_layer.jpg" width=500> 

#  X-DOG 
- In this part we apply xdog with different thresholds to get multiple images. 
- Once we get this stylized image, we input this into above pipeline to get makeup transfer on this image.  

<img src="data/files_for_md/xdog_combined_layer.jpg" width = 600>              
      

# References
- Digital Makeup Transfer based on paper by Dong Guo and Terence Sim
- dlib predictor downloaded from https://github.com/codeniko/shape_predictor_81_face_landmarks


```python

```
