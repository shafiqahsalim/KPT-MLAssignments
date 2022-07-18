# C) Calculating the number of coins in an image using contours. 
#mount the google drive
from google.colab import drive
drive.mount('/content/gdrive')

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

#Load the image
image = cv2.imread('/content/coins2.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Coins')
plt.show()

#Load the image in gray
image = cv2.imread('/content/coins2.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')

blur = cv2.GaussianBlur(gray, (11,11), 0) #(11,11) is the size of kernel window and 0 is the standard dev
plt.imshow(blur, cmap='gray')

#use canny edge algorithm to detect the edges
canny = cv2.Canny(blur, 30, 150, 3) 
#30 and 150 is the min and max value of threshold. 3 is the kernel size of sobel
plt.imshow(canny, cmap='gray')

#connect the edge by make the edges thicker
dilated = cv2.dilate(canny, (1,1), iterations = 2)
plt.imshow(dilated, cmap='gray')

#calculate contour
(cnt, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#RETR_EXT only consider external contour
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, cnt, -1, (0,255,0), 2)
#-1 to draw all contour. (0,255,0) color of the contour. 2 is the thickness
plt.imshow(rgb)

print('Coins in the image: ', len(cnt))
