# Digital Makeup Transfer
Digital Makeup Transfer based on paper by Dong Guo and Terence Sim

- Landmark creation
  - facial landmark detection
  - landmark point division
  - create image masks using convex hull
- Triangulation / mesh creation
  - compute triangles using cv2.Subdiv2D
  - convert triangle points to landmark indices so that this mesh can be universally used in different images
  - give every triangle the region it belongs to


# References
- Digital Makeup Transfer based on paper by Dong Guo and Terence Sim
- dlib predictor downloaded from https://github.com/codeniko/shape_predictor_81_face_landmarks
