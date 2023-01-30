# キャリブレーション関係

import numpy as np
import cv2
import glob
import random

from cvimshow import imshowS
import zed2MyParameters #, zed2_hanaizumi_parameters
from zed2MyParameters import Params

#----
# Windowsのカメラで撮影した映像についてキャリブレーション 
# フォルダ内からランダムにN本の動画を選択し，それでキャリブレーション
VID_FOLDER_PATH = ".\\res\\videos\\calibration\\"; PATTERN_SIZE = (6,9)
#VID_FOLDER_PATH = ".\\res\\videos\\20220912\\"; PATTERN_SIZE = (6,9)
#print(glob.glob(VID_FOLDER_PATH + "*.mp4"))

# 動画から切り出す画像のフレーム値
INIT_FRAME = 10

# ランダム選択
RANDOM_FLAG = False
RANDOM_N = 15

# ----
# 画像を2つに分割
def devideStereo(image):
    """画像を2つに分割
    色情報の変化は無し"""
    w = len(image[0])
    c = int(w/2)
    if image.ndim == 2:
        left = image[:,:c]
        right = image[:,c:]
    if image.ndim == 3:
        left = image[:,:c,:]
        right = image[:,c:,:]
    return left, right

def getChessboardCorners(r_img_l, r_img_r, pattern_size = PATTERN_SIZE, square_size = 10.0):
    if len(r_img_l.shape) > 2: # カラーの場合はグレースケール化
        r_img_l = cv2.cvtColor(r_img_l, cv2.COLOR_BGR2GRAY)
        r_img_r = cv2.cvtColor(r_img_r, cv2.COLOR_BGR2GRAY)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32) #チェスボード（X,Y,Z）座標の指定 (Z=0)
    pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points1 = []
    img_points2 = []

    #コーナー検出
    found_l, corner_l = cv2.findChessboardCorners(r_img_l, pattern_size,None)
    found_r, corner_r = cv2.findChessboardCorners(r_img_r, pattern_size,None)  
    # コーナーがあれば
    if found_l and found_r:
        obj_points.append(pattern_points)

        corners2_l = cv2.cornerSubPix(r_img_l,corner_l,(11,11),(-1,-1),criteria)
        corners2_r = cv2.cornerSubPix(r_img_r,corner_r,(11,11),(-1,-1),criteria)
        img_points1.append(corners2_l)
        img_points2.append(corners2_r)

        # Draw and display the corners
        if False:
            r_img_l = cv2.drawChessboardCorners(r_img_l, pattern_size, corners2_l, found_l)
            r_img_r = cv2.drawChessboardCorners(r_img_r, pattern_size, corners2_r, found_r)

    return obj_points, img_points1, img_points2

# ----
def getFilelist(folderpath=VID_FOLDER_PATH):
    filelist = glob.glob(folderpath + "*.mp4")
    if RANDOM_FLAG:
        filelist = random.sample(filelist, RANDOM_N)
    return filelist

def getCalibMatrix(filelist=getFilelist()):
    print(filelist)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    square_size = 10.0 #

    pattern_points = np.zeros( (np.prod(PATTERN_SIZE), 3), np.float32) #チェスボード（X,Y,Z）座標の指定 (Z=0)
    pattern_points[:,:2] = np.indices(PATTERN_SIZE).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points_l = []
    img_points_r = []

    for i, filename in enumerate(filelist):
        # 画像の取得
        vid = cv2.VideoCapture(filename)
        vid.set(cv2.CAP_PROP_POS_FRAMES, INIT_FRAME)
        ret, frame = vid.read()
        im_l, im_r = devideStereo(frame)
        imageSize = (im_l.shape[1], im_l.shape[0])

        gray_l = cv2.cvtColor(im_l,cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(im_r,cv2.COLOR_BGR2GRAY)
        
        if False:
            imshowS(gray_l, "left", wait=False)

        #コーナー検出
        found_l, corner_l = cv2.findChessboardCorners(gray_l, PATTERN_SIZE,None)
        found_r, corner_r = cv2.findChessboardCorners(gray_r, PATTERN_SIZE,None)  
        # コーナーがあれば
        if found_l and found_r:
            obj_points.append(pattern_points)

            corners2_l = cv2.cornerSubPix(gray_l,corner_l,(11,11),(-1,-1),criteria)
            corners2_r = cv2.cornerSubPix(gray_r,corner_r,(11,11),(-1,-1),criteria)
            img_points_l.append(corners2_l)
            img_points_r.append(corners2_r)

        # Draw and display the corners
        if False:
            r_img_l = cv2.drawChessboardCorners(r_img_l, PATTERN_SIZE, corners2_l, found_l)
            r_img_r = cv2.drawChessboardCorners(r_img_r, PATTERN_SIZE, corners2_r, found_r)

    #print(obj_points)
    lms, K_l, d_l, r, t = cv2.calibrateCamera(obj_points,img_points_l,(im_l.shape[1],im_l.shape[0]),None,None)
    print(f"Calibrate L rms: {lms}")
    rms, K_r, d_r, r, t = cv2.calibrateCamera(obj_points,img_points_r,(im_r.shape[1],im_r.shape[0]),None,None)
    print(f"Calibrate R rms: {rms}")

    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_l, img_points_r, K_l, d_l, K_r, d_r, imageSize)
    print(f"stereoCalibrate rms: {retval}")

    #print(cameraMatrix1,"\n",cameraMatrix2)

    flags=0
    alpha=-1
    newimageSize=imageSize
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, flags, alpha, newimageSize)

    # 平行化変換マップを求める
    m1type = cv2.CV_32FC1
    map1_l, map2_l = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, newimageSize, m1type) #m1type省略不可
    map1_r, map2_r = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, newimageSize, m1type)

    # npzファイルに保存
    np.savez('./src/_soma-stereo-maps.npz', map1_l, map2_l, map1_r, map2_r)

    # Remapしたチェスボードを表示
    stereoRemap(im_l, im_r, zed2MyParameters.maps, resultShow=True)

# ----
# npzファイルのステレオ変換Mapを使って平行化
def stereoRemap(imgl, imgr, maps = zed2MyParameters.MY_GOOD_PARAMS.maps, resultShow=False): # gray scale image
    # npz読み込み
    map1_l, map2_l, map1_r, map2_r = maps()

    # ReMapにより平行化を行う
    rimgl = cv2.remap(imgl, map1_l, map2_l, cv2.INTER_NEAREST) #interpolation省略不可
    rimgr = cv2.remap(imgr, map1_r, map2_r, cv2.INTER_NEAREST)

    if resultShow:
        imshowS(rimgl, "left image", wait=False)
        imshowS(rimgr, "right image", wait=False)

    # cv2.imwrite('./dst/zed2-vid/rectifiedleft.jpg', Re_TgtImg_l)
    # cv2.imwrite('./dst/zed2-vid/rectifiedright.jpg', Re_TgtImg_r)

    return rimgl, rimgr

# ----
# 重ね合わせてみようのコーナー
# def kasane(img_l, img_r):
#     gray_l = cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)
#     gray_r = cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)

#     # Harris角検出
#     fgrayl = np.float32(gray_l)
#     fgrayr = np.float32(gray_r)
#     chimgl = cv2.cornerHarris(fgrayl,11,9,0.1)
#     chimgr = cv2.cornerHarris(fgrayr,11,9,0.1)

#     # しきい値(最大値から10%をしきい値とする)
#     thresholdl = 0.015*chimgl.max()
#     thresholdr = 0.015*chimgr.max()
#     _, binimgl = cv2.threshold(chimgl,thresholdl,255,0)
#     _, binimgr = cv2.threshold(chimgr,thresholdr,255,0)
#     binimgl = np.uint8(binimgl)
#     binimgr = np.uint8(binimgr)

#     # ラベリング
#     retl, labelsl, statsl, centroidsl = cv2.connectedComponentsWithStats(binimgl)
#     retr, labelsr, statsr, centroidsr = cv2.connectedComponentsWithStats(binimgr)
#     #centroids = labeling(binimg, 10)

#     # サブピクセルによる詳細なコーナー位置の検出
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
#     cornersl = cv2.cornerSubPix(fgrayl,np.float32(centroidsl),(5,5),(-1,-1),criteria)
#     cornersr = cv2.cornerSubPix(fgrayr,np.float32(centroidsr),(5,5),(-1,-1),criteria)

#     # 表示データ結合とint変換
#     resl = np.int0(np.hstack((centroidsl,cornersl)))
#     resr = np.int0(np.hstack((centroidsr,cornersr)))

#     resl2=np.array(resl)
#     resr2=(np.array(resr))

#     #dis_res=abs(resr2-resl2)

#     for (x1,y1,x2,y2) in resl:
#         # Harris角検出
#         #cv2.circle(img_left,(x1,y1),5,(0,255,0),-1)

#         # サブピクセル角検出位置
#         cv2.drawMarker(img_l, (x2,y2), (0,255,0), markerType=cv2.MARKER_CROSS, 
#                 markerSize=8, thickness=1, line_type=cv2.LINE_AA)

#     #cv2.imwrite('L_cornerSubPix.png',img_left)

#     for (x1,y1,x2,y2) in resr:
#         pass
#         # Harris角検出
#         # cv2.circle(img_left,(x1,y1),5,(0,0,255),-1)

#         # サブピクセル角検出位置
#         cv2.drawMarker(img_r, (x2,y2), (0,0,255), markerType=cv2.MARKER_CROSS, 
#                 markerSize=8, thickness=1, line_type=cv2.LINE_AA)

#     # cv2.imwrite('./dst/zed2-vid/cornerSubPixLeft.png',img_l)
#     # cv2.imwrite('./dst/zed2-vid/cornerSubPixRight.png',img_r)

#     return

# ----
# 
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def yPosEvaluate(filelist=getFilelist()):
    """
    y座標の差分をとって評価

    Calibration Accuracy Evaluation with Stereo Reconstruction
    Songxiang Gu
    https://drive.google.com/drive/u/0/folders/1_dQ5ZwuGT2sI09v5y-lz1CH4nEqQNSvz 
    """

    fig = plt.figure(figsize=(8,5))
    ax = Axes3D(fig)

    ax.set_title("y Position Difference Between Stereo Images")
    ax.set_xlabel("x Position of Left Image")
    ax.set_ylabel("y Position of Left Image")
    ax.set_zlabel("y Position Difference")

    z_area = [-0.7, 0.7]
    ax.set_zlim3d(z_area)
    ax.set_xlim3d([300, 1800])
    ax.set_ylim3d([200, 1000])

    for i, filename in enumerate(filelist):
        # 画像の取得
        vid = cv2.VideoCapture(filename)
        vid.set(cv2.CAP_PROP_POS_FRAMES, INIT_FRAME)
        ret, frame = vid.read()
        im_l, im_r = devideStereo(frame)
        gray_l = cv2.cvtColor(im_l,cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(im_r,cv2.COLOR_BGR2GRAY)
        
        # 歪み補正
        r_gray_l, r_gray_r = stereoRemap(gray_l, gray_r, maps = zed2MyParameters.maps)
        if True:
            imshowS(r_gray_l, "img_l", wait=False)
            #imshowS(r_gray_r, "img_r", wait=False)

        # コーナー点取得
        _, img_points_l, img_points_r = getChessboardCorners(r_gray_l, r_gray_r)
        #print(len(img_points_l[0])) # 点の数

        # 正常に全ての点が得られなかったときはダメ
        if not img_points_l:
            print("point not found", filename)
            return -100, 100
        elif len(img_points_l[0]) < PATTERN_SIZE[0]*PATTERN_SIZE[1]:
            print("some points not found", filename)
            return -100, 100

        img_points_l_2 = np.array(img_points_l).reshape((PATTERN_SIZE[1], PATTERN_SIZE[0], 2))

        diff = np.array(img_points_l)-np.array(img_points_r)
        # print("img points l: ", img_points_l)
        # print("img points r: ", img_points_r)
        # print("diff:", diff)
        diff = diff.reshape((PATTERN_SIZE[1], PATTERN_SIZE[0], 2))
        #print(diff.shape)

        # 3次元グラフの表示
        #x, y = np.meshgrid(np.arange(PATTERN_SIZE[1]), np.arange(PATTERN_SIZE[0]))
        #print(img_points_l_2[:, 1])
        x = img_points_l_2[:, :, 0]
        y = img_points_l_2[:, :, 1]
        #print(x, "\n", y)
        z = diff[:,:,1]
        #print(z)

        #ax.plot_surface(x, y, z, cmap="bwr", alpha=0.5)
        ax.plot_surface(x, y, z, cmap="bwr", alpha=0.5)

        # y座標の差が-0.5～0.5になればいいんだけどなぁ
        print("z: ",np.min(z), np.max(z))

        #ax.set_zticklabels([-0.5, -0.3, 0.1, 0.3, 0.5])
    
    # 図を表示
    plt.show()

    #return np.min(z), np.max(z)

def main():
    # Get Callibration Matrix
    getCalibMatrix()
    # Value Callibration Result
    yPosEvaluate()
    pass

# def main(): # いい感じの結果が得られるまでループ
#     while True:
#         getCalibMatrix()
#         min, max = yPosEvaluate()
#         if (min > -2) and (max < 2):
#             break

# --------------
# main 実行
if __name__ == "__main__":
    main()
    print("=============================")

# ----
# garbage

    # np.savetxt("K_left.csv", K_l, delimiter =',',fmt="%0.14f") #カメラ行列の保存
    # np.savetxt("d_left.csv", d_l, delimiter =',',fmt="%0.14f") #歪み係数の保存
    # np.savetxt("K_right.csv", K_r, delimiter =',',fmt="%0.14f") #カメラ行列の保存
    # np.savetxt("d_right.csv", d_r, delimiter =',',fmt="%0.14f") #歪み係数の保存

    # np.savetxt("cameraMatrix1.csv", cameraMatrix1, delimiter =',',fmt="%0.14f") #新しいカメラ行列を保存
    # np.savetxt("cameraMatrix2.csv", cameraMatrix2, delimiter =',',fmt="%0.14f") 
    # np.savetxt("distCoeffs1.csv", distCoeffs1, delimiter =',',fmt="%0.14f") #新しい歪み係数を保存
    # np.savetxt("distCoeffs2.csv", distCoeffs2, delimiter =',',fmt="%0.14f")
    # np.savetxt("R.csv", R, delimiter =',',fmt="%0.14f") #カメラ間回転行列の保存
    # np.savetxt("T.csv", T, delimiter =',',fmt="%0.14f") #カメラ間並進ベクトルの保存