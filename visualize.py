import argparse
import cv2
import numpy as np
import imageio
import plotmatch
from lib.cnn_feature import cnn_feature_extract
import matplotlib.pyplot as plt
import time
from skimage import measure
from skimage import transform
from func import *
#time count
start = time.perf_counter()
dir_path = "512/jpg8k/"
import os
cuda = True
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


noise = 400
_RESIDUAL_THRESHOLD = 30
rootfolder = "/mnt/c/Users/kamanberu88/Desktop/JAXA_database/"
#Test1nThbg6kXUpJWGl7E1IGOCspRomTxdCARLviKw6E5SY8
imgfile1 = "/mnt/c/Users/kamanberu88/Desktop/JAXA_database/512/jpg8k/400/sim0001.jpg"
imgfile2 = img2_path = rootfolder + "mapimg/CST1/TCO_CST1_TM_SIM_a7351_i3438_h36000_lanczos3.bmp"
#imgfile1 = 'df-ms-data/1/df-uav-sar-500.jpg'

def matchRatiodayo(keyPoint1, feature1, keyPoint2, feature2, knn, ratio):
    matcher = cv2.BFMatcher()
    #flann = cv2.FlannBasedMatcher(index_params, search_params)
    a = None
    if type(feature2) == type(a):
        return 0, 0

    else:
        matches = matcher.knnMatch(feature1, feature2, k=knn)

        #matches = matcher.knnMatch(feature1.transpose(1, 0), feature2.transpose(1, 0), k=2)
        #matches = flann.knnMatch(feature1, feature2, k=2)

        good = []
        img1_pt = []
        img2_pt = []
        img1_f = []
        img2_f = []
        locations_1_to_use = []
        locations_2_to_use = []


        for n in range(len(matches)):
            first = matches[n][0]

            if matches[n][0].distance <= ratio * matches[n][1].distance:
                good.append([matches[n][0]])
                #img2_pt.append(keyPoint2[first.trainIdx].pt)
                #img1_pt.append(keyPoint1[first.queryIdx].pt)
                p2 = cv2.KeyPoint(keyPoint2[first.trainIdx][0], keyPoint2[first.trainIdx][1], 1)
                p1 = cv2.KeyPoint(keyPoint1[first.queryIdx][0], keyPoint1[first.queryIdx][1], 1)
                #img2_f.append(feature2[first.trainIdx])
                #img1_f.append(feature1[first.queryIdx])
                img1_pt.append([p1.pt[0], p1.pt[1]])
                img2_pt.append([p2.pt[0], p2.pt[1]])
                #locations_1_to_use.append([p1.pt[0], p1.pt[1]])
                #locations_2_to_use.append([p2.pt[0], p2.pt[1]])
        
        print('match num is %d' % len(good))
        


        return img1_pt, img2_pt


start = time.perf_counter()

# read left image
image1 = cv2.imread(imgfile1)
image2 = cv2.imread(imgfile2)

print('read image time is %6.3f' % (time.perf_counter() - start))

start0 = time.perf_counter()

c_pt, sco_right, cf = cnn_feature_extract(image1,  nfeatures = -1)
mpt, sco_left, mf= cnn_feature_extract(image2,  nfeatures = -1)

print('Feature_extract time is %6.3f, left: %6.3f,right %6.3f' % ((time.perf_counter() - start), len(c_pt), len(mpt)))
start = time.perf_counter()


#flann = cv2.FlannBasedMatcher(index_params, search_params)



ratio=0.95

    #自适应阈值
"""
    if n.distance > m.distance + disdif_avg:
        goodMatch.append(m)
        p2 = cv2.KeyPoint(kps_right[m.trainIdx][0],  kps_right[m.trainIdx][1],  1)
        p1 = cv2.KeyPoint(kps_left[m.queryIdx][0], kps_left[m.queryIdx][1], 1)
        locations_1_to_use.append([p1.pt[0], p1.pt[1]])
        locations_2_to_use.append([p2.pt[0], p2.pt[1]])
"""

c_pt,  m_pt = matchRatiodayo(c_pt, cf, mpt, mf, 2, ratio)
                #locations_1_to_use.append([p1.pt[0], p1.pt[1]])
                #locations_2_to_use.append([p2.pt[0], p2.pt[1]])
#goodMatch = sorted(goodMatch, key=lambda x: x.distance)
#print('match num is %d' % len(good))
#locations_1_to_use = np.array(locations_1_to_use)
#locations_2_to_use = np.array(locations_2_to_use)
print(len(c_pt))
print(len(m_pt))

# Perform geometric verification using RANSAC.
#_, inliers = measure.ransac((locations_1_to_use, locations_2_to_use),
 #                         transform.AffineTransform,
  ##                       residual_threshold=_RESIDUAL_THRESHOLD,
    #                      max_trials=1000)


cC_pt, m_pt = ransac(c_pt, m_pt, 2)


print('Found %d inliers' % (len(cC_pt)))

print('Found %d inliers' % (len(m_pt)))

inlier_idxs = np.nonzero(cC_pt)[0]
#最终匹配结果
matches = np.column_stack((inlier_idxs, inlier_idxs))
print('whole time is %6.3f' % (time.perf_counter() - start0))

# Visualize correspondences, and save to file.
#1 绘制匹配连线
locations_1_to_use = np.array(c_pt)
locations_2_to_use = np.array(m_pt)
plt.rcParams['savefig.dpi'] = 100 #图片像素
plt.rcParams['figure.dpi'] = 100 #分辨率
plt.rcParams['figure.figsize'] = (4.0, 3.0) # 设置figure_size尺寸
_, ax = plt.subplots()
plotmatch.plot_matches(
    ax,
    image1,
    image2,
    locations_1_to_use,
    locations_2_to_use,
    np.column_stack((inlier_idxs, inlier_idxs)),
    plot_matche_points = False,
    matchline = True,
    matchlinewidth = 0.3)
ax.axis('off')
ax.set_title('')
plt.savefig('./lashika4.png')
