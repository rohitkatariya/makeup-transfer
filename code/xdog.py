import os 
import cv2
import config
import numpy as np
list_images = os.listdir(config.input_dir)

img_rgb = cv2.imread("./../data/input/openmouth.jpg")
gray_image = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

gaussian_smoothened_1 = cv2.GaussianBlur(gray_image,(5,5),1.2)
gaussian_smoothened_2 = cv2.GaussianBlur(gray_image,(5,5),1.4*1.2)
p=3
XDOG_image = (1+p)*gaussian_smoothened_1 - p*gaussian_smoothened_2

DOG_image = gaussian_smoothened_1 - p*gaussian_smoothened_2

# plt.imshow( (XDOG_image), cmap='gray')
# print(np.max(XDOG_image))
# plt.show()

k= np.array((XDOG_image/255.)>0.70, dtype=np.float32);
cv2.imwrite('./../data/input/XDOG_image.jpg',np.uint8(k*255))
# time.sleep(0.3)
# cv2.imshow('img_{}'.format(idx_image),cropped_triangle)
cv2.waitKey(0) 
cv2.destroyAllWindows()