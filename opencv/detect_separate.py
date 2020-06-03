import cv2
import numpy as np
import random
import pathlib

# ファイル
p_file = pathlib.Path('./input/Kobe_sumiyoshi.png')
path = str(p_file)
name = p_file.name
stem = p_file.stem
# original data
img = cv2.imread(path)
# Convert
# BGR to HSV
img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# BGR to GRAY
img_GLAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(name+'_HSV.jpg', img_HSV)

# # Gaussian Filter
# img_HSV = cv2.GaussianBlur(img_HSV, (9, 9), 3)
# cv2.imwrite(name+'_Gauss.jpg', img_HSV)

def zeroImg(img):
    # ゼロ埋めの画像配列
    if len(img.shape) == 3:
        height, width, channels = img.shape[:3]
    else:
        height, width = img.shape[:2]
        channels = 1
    zeros = np.zeros((height, width), img.dtype)
    return zeros
zeros = zeroImg(img)

# detect tulips
img_R, img_G, img_B = cv2.split(img)
img_H, img_S, img_V = cv2.split(img_HSV)
# color set 
# img_R = cv2.merge((img_R, zeros, zeros))
# img_G = cv2.merge((zeros, img_G, zeros))
# img_B = cv2.merge((zeros, zeros, img_B))
cv2.imwrite(name+'_R.jpg', img_R)
cv2.imwrite(name+'_G.jpg', img_G)
cv2.imwrite(name+'_B.jpg', img_B)
cv2.imwrite(name+'_H.jpg', img_H)
cv2.imwrite(name+'_S.jpg', img_S)
cv2.imwrite(name+'_V.jpg', img_V)
_thre, img_flowers = cv2.threshold(img_R, 100, 120, cv2.THRESH_BINARY_INV)
# _thre, img_flowers = cv2.threshold(img_H, 80, 150, cv2.THRESH_BINARY)
cv2.imwrite(name+'_mask.jpg', img_flowers)

# find tulips
nlabels, labels = cv2.connectedComponents(img_flowers)


img = np.zeros(img.shape[0:3])
height, width = img.shape[0:2]
cols = []

# background is label=0, objects are started from 1
for i in range(1, nlabels):
    cols.append(np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]))

for i in range(1, nlabels):
    img[labels == i, ] = cols[i - 1]

# save
cv2.imwrite(name+'_object.jpg', img)