import numpy as np
import cv2
from matplotlib import pyplot as plt

import tkinter as tk
# python2の場合は、import Tkinter as tk
import tkinter.ttk as ttk
# python2の場合は、import ttk
import sqlite3

colormap_table_count = 0
colormap_table = [
    ['COLORMAP_AUTUMN',  cv2.COLORMAP_AUTUMN ],
    ['COLORMAP_JET',     cv2.COLORMAP_JET    ],
    ['COLORMAP_WINTER',  cv2.COLORMAP_WINTER ],
    ['COLORMAP_RAINBOW', cv2.COLORMAP_RAINBOW],
    ['COLORMAP_OCEAN',   cv2.COLORMAP_OCEAN  ],
    ['COLORMAP_SUMMER',  cv2.COLORMAP_SUMMER ],
    ['COLORMAP_SPRING',  cv2.COLORMAP_SPRING ],
    ['COLORMAP_COOL',    cv2.COLORMAP_COOL   ],
    ['COLORMAP_HSV',     cv2.COLORMAP_HSV    ],
    ['COLORMAP_PINK',    cv2.COLORMAP_PINK   ],
    ['COLORMAP_HOT',     cv2.COLORMAP_HOT    ]]

# img = cv2.imread('./coins.png', 0)
img = cv2.imread('./input/test1000750.tif', 0)
# 閾値の設定
threshold = 100

# 二値化(閾値100を超えた画素を255にする。)
ret, img_thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
ret2, img_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
# cv2.THRESH_BINARY_INVを追加することで反転
ret2, img_otsu_inv = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(img_otsu_inv,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
# markers = markers+1

# Now, mark the region of unknown with zero
# markers[unknown==255] = 0

apply_color_map_image = cv2.applyColorMap(img_otsu_inv, colormap_table[1][1])

#閾値がいくつになったか確認
print("ret2: {}".format(ret2))

# 二値化画像の表示
# cv2.imshow("img_th", img_thresh)
# cv2.imshow("img_otsu", img_otsu)
# 二値化画像の書き出し
cv2.imwrite("./img_th.png", img_thresh)
cv2.imwrite("./img_otsu.png", img_otsu)
cv2.imwrite("./img_otsu_inv.png", img_otsu_inv)
cv2.imwrite("./img_d.png", dist_transform)
cv2.imwrite("./img_s.png", sure_fg)
cv2.imwrite("./img_m.png", markers)
cv2.imwrite("./img_color.png", apply_color_map_image)

print("complete!")