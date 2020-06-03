import cv2
import numpy as np
import random
import pathlib
import os

def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

output_dir = "./output/tulip/"
makedirs(output_dir)

# load image, change color spaces, and smoothing
img = cv2.imread("./input/tulip.jpg")
img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_HSV = cv2.GaussianBlur(img_HSV, (9, 9), 3)

# detect tulips
img_H, img_S, img_V = cv2.split(img_HSV)
_thre, img_flowers = cv2.threshold(img_H, 140, 255, cv2.THRESH_BINARY)
cv2.imwrite(output_dir+"tulips_mask.jpg", img_flowers)

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
cv2.imwrite(output_dir+"tulips_object.jpg", img)