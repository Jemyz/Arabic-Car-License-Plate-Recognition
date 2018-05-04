import os
import cv2
import numpy as np
from package.plate_detection.detect_plate import PlateDetection
pd = PlateDetection()
imageg = cv2.imread("C:\\Users\\324\\Desktop\\fady\\1.png")
imageg = np.asarray(imageg)
box, img = pd.find(imageg)
print(img)
print(img.shape)
cv2.imshow("adsad",img)
cv2.waitKey()