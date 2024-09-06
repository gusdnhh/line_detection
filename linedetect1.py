import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import exposure

cap = cv2.VideoCapture('data/ride_video.mp4')

ret, frame = cap.read()

height, width = frame.shape[:2]
print(frame.shape[:2])

# 2. gradient combine 
# Find lane lines with gradient information of Red channel
temp = cv2.cvtColor(frame[180:height-50, 40:width], cv2.COLOR_BGR2GRAY)

# setting thresholds (hls, sobel)
th_h, thl, th_s = (160,255), (50,160), (0,255)
th_sobelx, th_sobely, th_mag, th_dir = (35,100), (30,255), (30,255), (0.7, 1.3)

def sobel_xy(img, orient='x', thresh=(20,100)):
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))

    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img,cv2.CV_64F, 0 ,1))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 255

    return binary_output

# th_sobelx = (35, 100)
# sobel_x = sobel_xy(temp, 'x', th_sobelx)

temp = cv2.cvtColor(frame[180:height-50, 40:width], cv2.COLOR_BGR2GRAY)
abs_sobel = np.absolute(cv2.Sobel(temp, cv2.CV_64F, 1, 0))
scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
sobel_x = np.zeros_like(scaled_sobel)

th_sobelx = (35, 100)
sobel_x[(scaled_sobel >= th_sobelx[0]) & (scaled_sobel <= th_sobelx[1])] = 255
plt.imshow(sobel_x)
plt.show()

# https://velog.io/@choonsik_mom/Lane-Detection%EC%B0%A8%EC%84%A0-%EC%9D%B8%EC%8B%9D