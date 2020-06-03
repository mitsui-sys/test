import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
from pathlib import Path
import gdal
import osr
from fiona.crs import from_epsg
import argparse
from PIL import Image, ImageOps

def  analysis_cell_image(img):
    height = img.shape[0]
    width = img.shape[1]

    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 0)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=2)


    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 0)
    ret, sure_fg = cv2.threshold(dist_transform,  0.05*  dist_transform.max(), 255, 0)
#    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)


 #  ret, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.min(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    labelnum, labelimg, contours, GoCs = cv2.connectedComponentsWithStats(sure_fg)
    return labelnum,labelimg,contours,GoCs


def  filter_area_matching(contours):
    contours_out = np.empty((0, 5), int)
    overlap = np.empty((0, 5), int)
    overlap_line = np.empty((0, 5), int)
    overlap_offset = np.empty((0, 5), int)
    contours2=contours[contours[:, 0].argsort(), :]
    labelnum=len(contours)
    for label in range(1, labelnum-1):
      x, y, w, h, size = contours2[label]
      overlap_status = 0
      for label2 in range(label+1, labelnum - label+1):
         x2, y2, w2, h2, size2 = contours2[label2]
#         if x2 - w < x < x2 + w2 and y2 - h < y < y2 + h2:
         if (x <= x2 < x + w) and (y <= y2 < y + h):
             overlap_status=1
             rec_base=[0,0,0,0,0]
             rec_data=[0,0,0,0,0]
             if x2+w2 < x + w:
                 rec_data[2]=w
             else:
                 rec_data[2]=w2
             if y2 + h2 < y + h:
                   rec_data[3]=h
             else:
                   rec_data[3]=h2
             rec_base[0]=x
             rec_base[1]=y
             b_status=(overlap_line == rec_base).all(axis=1).any()
             if(b_status):
                   pos = list(nd.sum() for nd in (overlap_line == rec_base))
                   offset=pos.index(5)
                   if offset>=0:
                     rec_data2=overlap_offset[offset]
                     if rec_data2[2]>rec_data[2]:
                        rec_data[2]=rec_data2[2]
                     if rec_data2[3]>rec_data[3]:
                       rec_data[3]=rec_data2[3]
                     overlap_offset=np.delete(overlap_offset,offset,0)
                     overlap_offset = np.append(overlap_offset, [rec_data], axis=0)
             else:
               overlap_line = np.append(overlap_line, [rec_base], axis=0)
               overlap_offset=np.append(overlap_offset,[rec_data],axis=0)

      rec_data=contours2[label]
      if overlap_status==0:
         if np.all(contours_out==rec_data, axis=1).sum():
               print("OK")
         else:
               contours_out = np.append(contours_out, [rec_data], axis=0)
    return contours_out,overlap_line,overlap_offset

def min_area_remove(contours,minsize,Gocs):
    contours_out=np.empty((0,5),int)
    GoCs2=np.empty((0,2),int)
    labelnum=len(contours)
    for label in range(1, labelnum):
      x, y, w, h, size = contours[label]
      if (minsize<size):
          contours_out = np.append(contours_out, [contours[label]], axis=0)
          GoCs2 = np.append(GoCs2, [GoCs[label]], axis=0)
    return contours_out,GoCs2

def check_image_area(img,x,y,w,h):

    img_g=img[y:y + h, x:x + w]
    imgF = img_g.copy()
    labelnum,labelimg,contours,GoCs=analysis_cell_image(imgF)
    contours2,_,_=filter_area_matching(contours)
    return labelnum,labelimg,contours2,GoCs

def _to_lonlat( pix_x, pix_y, resolution,ul_lon,ul_lat):
    #ピクセル座標を経度/緯度に変換します
        lon =ul_lon + (pix_x * resolution)
        lat = ul_lat - (pix_y * resolution)
        return lon, lat

def _to_colrow(lon, lat, resolution,ul_lon,ul_lat):
    #経度/緯度をピクセル座標に変換します
    x = (lon - ul_lon) / resolution
    y = (ul_lat - lat) / resolution
    if isinstance(x, type(y)):
        if isinstance(x, float):
            return int(x), int(y)
        if isinstance(x, (np.ndarray, pd.Series)):
            return np.array([x, y], dtype=int)
    else:
        raise Exception("Can't handle different input types for x, y.")

df = pd.DataFrame()
target_filelist = []
trees_top = pd.DataFrame(columns=[
    'top_x', 'top_y',
])
trees_crown = pd.DataFrame(columns=[
    'x', 'y', 'width','hight'
])

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default='./input/', help='photo data file path')
parser.add_argument('--out_dir', type=str, default='./output/', help='output file path')
parser.add_argument('--crown_size', type=int, default=500, help='limit crown size')

FLAGS = parser.parse_args()
image_dir = FLAGS.img_path
out_dir = FLAGS.out_dir
limit_size = FLAGS.crown_size
target_files = os.listdir(image_dir)

for target_file in target_files:
    if target_file == '.DS_Store':
        continue

    if target_file == 'Thumbs.db':
        continue
    # 入力画像の読み込み
    print("データ：", image_dir+target_file)
    img = cv2.imread(image_dir+target_file,0)
    basefile = Path(target_file).stem

    # マルチバンド画像(GeoTIFF)の読み込み
    try:
        chm_gdal = gdal.Open(str(image_dir+target_file), gdal.GA_ReadOnly)
    except RuntimeError as e:
        raise IOError(e)
    proj = osr.SpatialReference(wkt=chm_gdal.GetProjection())
    print(proj.GetAttrValue('AUTHORITY', 1))
    # epsg = int(proj.GetAttrValue('AUTHORITY', 1))
    # srs = from_epsg(epsg)
    geotransform = chm_gdal.GetGeoTransform()
    resolution = abs(geotransform[-1])
    ul_lon = chm_gdal.GetGeoTransform()[0]
    ul_lat = chm_gdal.GetGeoTransform()[3]
    chm0 = chm_gdal.GetRasterBand(1).ReadAsArray()
    chm_gdal = None

    height = img.shape[0]
    width = img.shape[1]


 #   th2=cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,51,20)
    # 適応型しきい値処理
    # 画像中の小領域ごとにしきい値の値を計算
    th2=cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,51,0)
    #   th2 = cv2.bitwise_not(th2)

    # モルフォロジー変換、膨張処理と収縮処理
    kernel = np.ones((2, 2), np.uint8)
    # オープニング、収縮後に膨張する処理
    opening = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel, iterations=2)
    # 膨張
    sure_bg = cv2.dilate(opening, kernel, iterations=2)


    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 0)
    ret, sure_fg = cv2.threshold(dist_transform,  0.05*  dist_transform.max(), 255, 0)
#    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)


 #  ret, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.min(), 255, 0)
    minsize=20
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

#    labelnum, labelimg, contours, GoCs = cv2.connectedComponentsWithStats(sure_fg)
    labelnum, labelimg, contours, GoCs = cv2.connectedComponentsWithStats(unknown)

    contours,GoCs=min_area_remove(contours,minsize,GoCs)

    imgF = img.copy()
    height = img.shape[0]
    width = img.shape[1]
#    img=unknown
    contours2, _, _ = filter_area_matching(contours)

    labelnum=len(contours)
    for label in range(1, labelnum):
        x0, y0 = GoCs[label]
        lon0,lat0=_to_lonlat(x0, y0, resolution, ul_lon, ul_lat)
        img = cv2.circle(img, (int(x0), int(y0)), 1, (255, 0, 0), -1)
        x, y, w, h, size = contours[label]
        if (height==h) &(width==w):
            continue
        if w>limit_size:
            labelnum2,labelimg2,contours2,GoCs2=check_image_area(imgF, x, y, w, h)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 1)
            lon, lat = _to_lonlat(x, y, resolution, ul_lon, ul_lat)
            for label2 in range(1, labelnum2):
                x2, y2 = GoCs2[label2]
                img = cv2.circle(img, (x + int(x2), y + int(y2)), 3, (255, 0, 0), -1)
                # (height,width)=(y,x)   3750,5000
                lon2, lat2 = _to_lonlat(x2, y2, resolution, ul_lon, ul_lat)
                tmp_se = pd.Series([lon2, lat2], index=trees_top.columns)
                trees_top = trees_top.append(tmp_se, ignore_index=True)
                x2, y2, w2, h2, size = contours2[label2]

    #              img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 1)
                img = cv2.rectangle(img, (x + x2, y + y2), (x + x2 +w2, y + y2 + h2), (255, 255, 0), 1)
                lon2, lat2 = _to_lonlat(x+x2, y+y2, resolution, ul_lon, ul_lat)
        else:
            img=cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 1)
            lon,lat=_to_lonlat(x, y, resolution, ul_lon, ul_lat)
            tmp_se = pd.Series([lon0,lat0], index=trees_top.columns)
            trees_top = trees_top.append(tmp_se, ignore_index=True)

    cv2.imwrite(out_dir+basefile+"_labeling.png", img)
    cv2.imwrite(out_dir+basefile+"_labeling2.png", sure_fg)
    trees_top.to_csv(out_dir+target_file+"_trees.csv")