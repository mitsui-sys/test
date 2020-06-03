#!/usr/bin/env python
        # -*- coding: utf-8 -*-

import cv2
import numpy as np
import random
import sys

if __name__ == '__main__':

    # 対象画像を指定
    input_image_path = './coins.png'

    # 画像をグレースケールで読み込み
    gray_src = cv2.imread(input_image_path, 0)

    # 前処理（平準化フィルターを適用した場合）
    # 前処理が不要な場合は下記行をコメントアウト
    blur_src = cv2.GaussianBlur(gray_src, (5, 5), 2)

    # 二値変換
    # 前処理を使用しなかった場合は、blur_srcではなくgray_srcに書き換えるする
    mono_src = cv2.threshold(blur_src, 48, 255, cv2.THRESH_BINARY_INV)[1]

    # ラベリング処理
    ret, markers = cv2.connectedComponents(mono_src)

    # ラベリング結果書き出し準備
    color_src = cv2.cvtColor(mono_src, cv2.COLOR_GRAY2BGR)
    height, width = mono_src.shape[:2]
    colors = []

    for i in range(1, ret + 1):
        colors.append(np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]))

    # ラベリング結果を画面に表示
    # 各オブジェクトをランダム色でペイント
    for y in range(0, height):
    
        for x in range(0, width):
            if markers[y, x] > 0:
                color_src[y, x] = colors[markers[y, x]]
            else:
                color_src[y, x] = [0, 0, 0]

    # オブジェクトの総数を黄文字で表示
    cv2.putText(color_src, str(ret - 1), (100, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))

    # 結果の表示
    cv2.imshow("color_src", color_src)

    cv2.waitKey(0)
    cv2.destroyAllWindows()