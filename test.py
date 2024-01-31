import glob
import cv2
from func import AKAZE
import time
from func import *
suc_mum=0
knn = 2
ratio = 0.93# 射影なら0.84
limitpx = 3
ransacnum = 5000
AKAZE_th = -0.000005
sgms = 1.0
from lib.cnn_feature import cnn_feature_extract
import sys
import csv
LOF_ave = 0
from tqdm import tqdm
#rootfolder = "/home/kaman/JAXA_database/"
#rootfolder = "/home/natori21/JAXA_database/"
rootfolder = "/mnt/c/Users/kamanberu88/Desktop/JAXA_database/"
#rootfolder =r"C:\Users\kaman\Desktop\JAXA_database\\"
dir_path = "512/jpg8k/"

cuda = True


noise = 400
start = time.time()
mode = 2
# img1_path=rootfolder+dir_path+str(noise)+"/*"
img1_path = rootfolder + dir_path + str(noise) + "/"
#img2_path = rootfolder + "TCO_CST1_TM_SIM_a7351_i3438_h36000_lanczos3.bmp"
#img2_path = rootfolder + "mapimg\\CST1\\TCO_CST1_TM_SIM_a7351_i3438_h36000_lanczos3.bmp"
img2_path = rootfolder + "mapimg/CST1/TCO_CST1_TM_SIM_a7351_i3438_h36000_lanczos3.bmp"
truepoint_path = rootfolder + dir_path + str(noise) + "/" + "true_point.csv"

# imgファイルの名前読み込み
# f = open(img1_path + "imgfile.txt")
f = open(rootfolder + dir_path + str(noise) + "/" + "imgfile.txt")
data = f.readlines()
f.close()

del f

# 真値の値を持っているファイルの読み込み
f2 = open(truepoint_path)
data2 = f2.readlines()
f2.close()

del f2

Scount = 0


ng_lis = []
ng_lis1 = []
ng_lis2 = []
ng_lis3 = []

f3 = open("s_Matching" + str(noise) + ".csv", "w")
img2 = cv2.imread(img2_path)
#img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#img2=cv2.resize(img2,(1024,1024))
print(img2.shape)
#cv2.imwrite('./aho.jpg',img2)
mpt, sco_left, mf = cnn_feature_extract(img2,  nfeatures = -1)
# print(mpt)  #keypoint
#print(mf)
#with open('./kosu_d2_1result403.csv','w',newline="") as f:
 #       writer = csv.writer(f)
  #      writer.writerow(['d2net_match','d2_inlier ','kyori'])
        #writer.writerow([len(c1_pt),len(c_pt)])
for l in tqdm(range(len(data))):
    st=time.time()
    path1 = img1_path + data[l].rstrip("\n")
    print(path1)
    true_point = getTruePoint(data2)
    img1 = cv2.imread(path1)
    #img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #c_pt, cf = getPairs(img1, fe)
    c_pt_re, sco_right, cf = cnn_feature_extract(img1, nfeatures=-1,)

    phase = "RatioTest"
    label1 = phase
    #c_pt, c_f, m_pt, m_f = matchRatio(c_pt, cf, mpt, mf, knn, ratio)
    c1_pt,c_f,  m_pt,m_f = matchRatio(c_pt_re, cf, mpt, mf, knn, ratio)
    phase = "Ransac"
    label3 = phase
    print(len(c1_pt))

    c_pt, m_pt = ransac(c1_pt, m_pt, mode)
    print(len(c_pt))






    try:
        tform = tformCompute(c_pt, m_pt, mode, num=len(c_pt))
    except IndexError:
        ng_lis2.append(data[l])
        check = False
    else:
        check, d = judge(true_point, tform, limitpx=3)
        if check == False:
            ng_lis3.append(data[l].rstrip("\n"))

        else:
            pass
    # f3.write("\n" + str(d) + "," + str(lof))
    f3.write("\n" + str(d) + ",")
    gl = time.time()
    if d <= limitpx**2:
        suc_mum += 1

    print(suc_mum)

    #with open('./kosu_d2_1result403.csv','a',newline="") as fa:
     #   writer = csv.writer(fa)
        #writer.writerow(['d2net_match','d2_inlier'])
     #   writer.writerow([len(c1_pt),len(c_pt),d])


goal = time.time()
score = goal - start
print(int(score / 3600), "時間", int((score % 3600) / 60), "分", score % 60, "秒")



