import os 
import cv2
import config
import numpy as np
list_images = os.listdir(config.input_dir)

img_rgb = cv2.imread("./../data/xdog.png")
gray_image = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

gaussian_smoothened_1 = cv2.GaussianBlur(gray_image,(5,5),1.2)
gaussian_smoothened_2 = cv2.GaussianBlur(gray_image,(5,5),1.4*1.2)
p=3
XDOG_image = (1+p)*gaussian_smoothened_1 - p*gaussian_smoothened_2

DOG_image = gaussian_smoothened_1 - p*gaussian_smoothened_2

# plt.imshow( (XDOG_image), cmap='gray')
# print(np.max(XDOG_image))
# plt.show()
for i in range(5):
    k= np.array((XDOG_image/255.)>0.1*i, dtype=np.float32);
    cv2.imshow('XDOG_image{}'.format(i),k)
# time.sleep(0.3)
# cv2.imshow('img_{}'.format(idx_image),cropped_triangle)
cv2.waitKey(0) 
cv2.destroyAllWindows()