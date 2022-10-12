# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 15:16:33 2021

Theory of Canny method
=====================================================================================================
refer to: https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html
Canny Edge Detection is a popular edge detection algorithm. It was developed by John F. Canny in
1.It is a multi-stage algorithm and we will go through each stages.
2.Noise Reduction
  Since edge detection is susceptible to noise in the image, first step is to remove the noise in
  the image with a 5x5 Gaussian filter.
3.Finding Intensity Gradient of the Image
  Smoothened image is then filtered with a Sobel kernel in both horizontal and vertical direction
  to get first derivative in horizontal direction and vertical direction. 
4.Non-maximum Suppression
  After getting gradient magnitude and direction, a full scan of image is done to remove any
  unwanted pixels which may not constitute the edge. For this, at every pixel, pixel is checked
  if it is a local maximum in its neighborhood in the direction of gradient. 
5.Hysteresis Thresholding
  This stage decides which are all edges are really edges and which are not. For this, we need two
  threshold values, minVal and maxVal. Any edges with intensity gradient more than maxVal are sure
  to be edges and those below minVal are sure to be non-edges, so discarded. Those who lie between
  these two thresholds are classified edges or non-edges based on their connectivity. If they are
  connected to "sure-edge" pixels, they are considered to be part of edges. 
=====================================================================================================

@author: Ruanyuezhe
"""

import cv2

import numpy as np
import os

from Main.Metrics import SegmentationMetric
import PIL.Image as Image


filepath = "file_path_of_images"
dirs = os.listdir(filepath)

means = []
mious = []
mrecall = []
mf1 = []

init_threhold = 90
canny_up = 110
canny_down = 90
kernel_size = (6, 4)

for ind, dir_ in enumerate(dirs):
    label = np.array(Image.open(filepath + "\\" + dir_ + "\\label.png").convert('P'))
    src = cv2.imread(filepath + "\\" + dir_ + "\\img.png")
    hsv_ori = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(hsv_ori)
    r[r < init_threhold] = 255
    r = cv2.GaussianBlur(r, (5, 5), 0)
    
    # -- Edge detection -------------------------------------------------------------------
    edges = cv2.Canny(r, canny_down, canny_up)
    KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    edges = cv2.dilate(edges, KERNEL, iterations=1)
    edges = cv2.erode(edges, KERNEL, iterations=1)
    
    # -- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        contour_info.append((c, cv2.isContourConvex(c), cv2.contourArea(c)))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]
    mask = np.zeros(edges.shape)
    cv2.fillPoly(mask, [max_contour[0]], (255))
    
    # -- Smooth mask, then blur it --------------------------------------------------------
    mask = cv2.dilate(mask, KERNEL, iterations=1)
    mask = cv2.erode(mask, KERNEL, iterations=1)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    
    # -- Save image -----------------------------------------------------------------------
    ori = src.astype('float32')
    mask = mask.astype('float32')
    result = cv2.bitwise_and(ori, ori, mask)
    result[mask == 0] = 255  # white background

    os.chdir("save_image_directory")
    cv2.imwrite("{}_canny.png".format(ind), result)

    # -- Metrics for evaluation ------------------------------------------------------------
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    result[result == 255] = 0
    result[result != 0] = 1
    
    metric = SegmentationMetric(2)
    metric.addBatch(result.astype(np.int64), label)
    
    c_precision = metric.classPixelAccuracy()
    c_recall = metric.classRecall()
    F1_score = metric.class_F1_score()
    c_iou = metric.meanIntersectionOverUnion()[1]
    m_precision = metric.meanPixelAccuracy()
    m_recall = metric.meanRecall()
    Micro_F1 = metric.F1_score()
    miou = metric.meanIntersectionOverUnion()[0]
    
    print('current img index is:', ind)            
    print('precision is : %f' % (m_precision))
    print('recall is : %f' % (m_recall))
    print('F1_score is : %f' % (Micro_F1))
    print('miou is : %f' % (miou))
    means.append(m_precision)
    mious.append(miou)
    mrecall.append(m_recall)
    mf1.append(F1_score)
    metric.reset()