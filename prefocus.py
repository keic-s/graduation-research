import re
from turtle import pos
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

# 自作モジュール
from cvimshow import imshowS, grayToColor, imMask, imshowMask, imshowColorShift, imColorShift
from calibration import devideStereo, getChessboardCorners
import calibration
from floorEstimation import floorEstimation, planeZ
# from src.floorEstimation import floorEstimation
# from src.stereo3d import stereo3d
from stereo3d import stereo3d
import zed2MyParameters
import zed2_hanaizumi_parameters

# ----
# 表示速度
DELAY = 1

# 保存
SAVE_FLAG = False

# 動画から読み込む
DEBUG_MODE = True

PARAMS = zed2_hanaizumi_parameters

# 事前固定焦点法
DIST_MAX = 10.0 + random.random()       # 探索距離の幅(m)
DIST_MIN = 0.5 + random.random()/10     # 
FOCUS_LENGTH = 1.07493419e+03           # 焦点距離/画素間隔 (m/(m/画素) = 画素)
BASELINE_LENGTH = 0.12                  # 基線長(m)

# ---
HEIGHT = 1.8; WIDTH = 0.6
POS_UPPER_LEFT = np.array([-1.25, -0.2, None]) # 左上
POS_LOWER_LEFT = np.array([-1.55, 0.7, None]) # 左下
POS_LOWER_RIGHT = np.array([1.62, 0.8, None]) # 右下
POS_UPPER_RIGHT = np.array([0.8, -0.7, None]) # 右上
POS = POS_LOWER_RIGHT

# 2m 離したチェスボード
if False:
    INIT_FRAME = 10
    FILENAME = "./res/videos/20220721/dist2m (1).mp4"
    MAPS = zed2MyParameters.maps
    MASK_AREA = (355, 753, 721, 1270) # ymin, ymax, xmin, xmax

# 4人が動いてる動画
if True:
    INIT_FRAME = 980
    FILENAME = "./res/hana/交差/WIN_20201211_15_20_42_Pro.mp4"
    MAPS = zed2_hanaizumi_parameters.maps
    #MASK_AREA = (215, 510, 537, 704) # 右カメラのマスク

# ----
# 事前固定焦点法
def prefocus(img_l, img_r, mask):
    """事前固定焦点法
    
    img_l:  左カメラの画像
    img_r:  右カメラの画像
    mask:   マスク
    """

    distance_resolution = 0.001    # 距離解像度(m)

    # カラーの場合はグレーへ
    if len(img_l.shape) > 2:
        # Grayscale Images
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # 距離の初期値
    dist_short = DIST_MIN #(m)
    dist_long = DIST_MAX #(m)
    dist_mid = (dist_long - dist_short) / 2.0 #(m)

    # 距離を二分探索
    while True:
        if np.abs(dist_long-dist_short) < distance_resolution:
            print("dist short: ", dist_short, "\ndist long: ", dist_long)
            imshowColorShift(imMask(imwarpDist(img=img_l, distance=dist_mid), mask=mask), img_r, winname="masked", wait=True)
            return dist_mid
        v_short = opticalflow(img_l, img_r, dist_short, mask) # オプティカルフローを用いた位置ずれの評価値
        #v_mid = opticalflow(img_l, img_r, dist_mid, mask)
        v_long = opticalflow(img_l, img_r, dist_long, mask)
        #print(dist_short, ": ", v_short, ", dist_long: ", v_long)
        print(dist_short, dist_mid, dist_long)
        if v_short < v_long: # dist_shortのが位置ずれが小さい(距離あってそうな)場合
            #dist_short = dist_short
            dist_long = dist_mid
            dist_mid = (dist_long + dist_short) / 2. #(m)
        else:# v_short > v_long dist_longのが位置ずれが小さい(距離あってそうな)場合
            dist_short = dist_mid
            #dist_long = dist_long
            dist_mid = (dist_long + dist_short) / 2. #(m)

# ----
# オプティカルフロー計算
def opticalflow(img_l, img_r, dist, mask, params=PARAMS, approach = "farneback"):
    """
    距離を仮定して各画素の移動量dxyから位置ずれの少なさvxyを定義
    vxy = 0       (|dxy|>1)
        = 1-|dxy| (|dxy|<=1)
    位置ずれの大きさの値を導出

    img_l:  左カメラの画像
    img_r:  右カメラの画像
    dist:   深度情報
    mask:   Optical Flow を計算する領域，左カメラのマスク
    ff:     焦点距離
    bb:     基線長
    """

    # 画像をグレースケール化
    if len(img_l.shape) > 2:
        img_l = cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)
        img_r = cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)

    h, w = img_l.shape
    
    # l(赤,左)を変形してr(青に合わせる)
    img_l_2 = imwarpDist(img_l, dist)
    #mask_2 = imwarpDist(mask, dist)

    if approach == "farneback":
        # チェスボード用
        flow = cv2.calcOpticalFlowFarneback(img_r, img_l_2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        #print(flow)
        ev = np.sum(np.linalg.norm(flow, axis=2)*mask) # マスク適用

        # オプティカルフローを可視化したかった
        hsv = np.zeros((h,w,3)).astype(np.float32)
        hsv[...,1] = 255
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2BGR)
        #imshowS(rgb, winname="Optical Flow", wait=False)

        # size = 3 # 正方形サイズ
        # for x in range(size//2, w-size//2):
        #     img_l[x, y, :]
    else:
        # 自作オプティカル
        # 平行化しているので横方向のみの探索でいいとする
        # 微分でいける(先行研究だと前進微分)
        flow = cv2.calcOpticalFlowFarneback(img_r, img_l_2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        ev = np.sum(np.linalg.norm(flow, axis=2)*mask)

        # size = 3 # 正方形サイズ
        # for x in range(size//2, w-size//2):
        #     img_l[x, y, :]

    # 結果表示
    imshowColorShift(img_l_2, img_r, wait=False) # 青が右
    return ev

def imwarpDist(img, distance, params=PARAMS):
    """距離から画像を射影変換
    
    変換行列要検討"""
    h, w = img.shape[:2]
    mat = np.array([
        [1., 0., -params.ff/distance*params.bb],
        [0., 1., 0.],
        [0., 0., 1.]
        ])
    dst = cv2.warpPerspective(img, mat, (w, h))
    return dst

# ----

def pos2mask(img, pos, point, vector, height, width, lr="right"):
    """
    地面に垂直な平面の定義
    マスクを返す

    img: 画像(右カメラ)
    pos: 3次元位置(足元の中心)
    point: 平面上の点
    vector: 法線ベクトル(単位ベクトル)
    height: 人物の高さ
    width: 幅
    """

    h, w = img.shape[:2]

    points = pos2points(img, pos, point, vector, height, width, lr)

    # マスク作成
    pix = 1.
    mask = np.zeros(img.shape[:2])
    cv2.fillConvexPoly(mask, points, color=pix)

    # 描画
    disp = img.copy()
    cv2.polylines(disp, [points], color=(0,0,255), isClosed=True, thickness=10)
    cv2.drawMarker(disp, pos3dToPos2d(pos, w, h)[0], color=(255, 0, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=20,thickness=5)
    imshowS(disp, winname="Mask Area", wait=True)

    # plt.imshow(mask)
    # plt.show()

    return mask

def pos3dToPos2d(pos3d, width, height, lr="right"):
    #print(len(pos3d.shape))
    if len(pos3d.shape) < 2:
        pos3d = np.array([pos3d, ])
    #print(pos3d.shape)
    if lr=="left":
        x = PARAMS.ff*pos3d[:, 0]/pos3d[:, 2]+width/2.
    else:
        x = PARAMS.ff*(pos3d[:, 0]-PARAMS.bb)/pos3d[:, 2]+width/2.
    y = PARAMS.ff*pos3d[:, 1]/pos3d[:, 2]+height/2.
    pos2d = np.stack((x, y),axis=1)
    #pos2d.reshape((pos3d.shape[0],2))
    return pos2d.astype(np.int64)

def pos2points(img, pos, point, vector, height, width, lr="right"):
    """3次元点からバウンディングボックスの4点のカメラ座標を返す"""

    #ベクトルが地面に上向きか？
    vector = -vector
    
    rect_points = np.array([
        [pos[0]-width/2, pos[1], pos[2]],   # 左下
        [pos[0]+width/2, pos[1], pos[2]],   # 右下
        [0., 0., 0.],                       # 右上
        [0., 0., 0.]                        # 左上
    ])
    
    #print("vector ノルム: ", np.linalg.norm(vector))
    rect_points[2] = rect_points[1]+vector*height
    rect_points[3] = [rect_points[2][0]-width, rect_points[2][1], rect_points[2][2]]
    #print(rect_points, rect_points.dtype)
    
    # 3次元位置->カメラ座標
    ff = PARAMS.ff
    tu = PARAMS.tu
    tv = PARAMS.tv
    bb = PARAMS.bb
    h, w = img.shape[:2]

    rect_2d_points = pos3dToPos2d(rect_points, width=w, height=h)
    #print(rect_2d_points)
    points = np.array(rect_2d_points).reshape((-1,1,2)).astype(np.int32)
    
    return points

# ----
# main
def main():
    # 動画を読み込み
    vid = cv2.VideoCapture(FILENAME)

    if vid.isOpened(): #動画を読み込めているとき
        # Video Loaded
        vid.set(cv2.CAP_PROP_POS_FRAMES, INIT_FRAME) # set frame to load
        ret, frame = vid.read() # read
        if not ret: # if not read
            return
        
        # 歪み補正・平行化
        imgl, imgr = devideStereo(frame) # devide
        rimgl, rimgr = calibration.stereoRemap(imgl, imgr, maps=MAPS) # callibrate
        #imshowColorShift(rimgl, rimgr)

        # 地面推定
        plane_p, plane_v = floorEstimation(stereo3d(rimgl, rimgr))

        # マスク作成
        POS[2] = planeZ(POS[0], POS[1], plane_p, plane_v)
        mask = pos2mask(rimgr, POS, plane_p, plane_v, HEIGHT, WIDTH)
        imshowMask(rimgr, mask, winname="Mask", wait=False)
        #mask = np.zeros(rimgl.shape[:2]) #マスク無し
        #mask[MASK_AREA[0]:MASK_AREA[1], MASK_AREA[2]:MASK_AREA[3]] = 1. # make a mask

        #cv2.fillConvexPoly(rimgr, points, color=pix)
        #cv2.polylines(rimgr, [points], color=(255,0,0), isClosed=True, thickness=10)
        # plt.imshow(rimgr)
        # plt.show()

        dist = prefocus(rimgl, rimgr, mask) # pre-focusing method
        print("結果: ", dist)

        RED = (0,0,255)
        BLUE = (255, 0, 0)
        dst = imColorShift(imwarpDist(rimgl, distance=dist), rimgr)
        rect_points = np.array([
                    [611, 498],
                    [736, 498],
                    [651, 234],
                    [468, 234]
        ])
        cv2.polylines(dst, [rect_points], color=BLUE, isClosed=True, thickness=8)
        rect_points = np.array([
                    [508, 700],
                    [646, 700],
                    [499, 503],
                    [289, 503]
        ])
        cv2.polylines(dst, [rect_points], color=BLUE, isClosed=True, thickness=8)
        rect_points = np.array([
            [1231,  721],
            [1367,  721],
            [1597,  538],
            [1391,  538]
        ])
        cv2.polylines(dst, [rect_points], color=RED, isClosed=True, thickness=8)
        rect_points = np.array([
            [1034,  402],
            [1152,  402],
            [1248,  120],
            [1081,  120]
        ])
        cv2.polylines(dst, [rect_points], color=BLUE, isClosed=True, thickness=8)
        imshowS(dst, wait=True)

        print("----\nOptical Flow")
        print("求めた領域: ", opticalflow(rimgl, rimgr, dist, mask))

        # 他の領域それぞれのオプティカルフローを計算
        POS_UPPER_LEFT[2] = planeZ(POS_UPPER_LEFT[0], POS_UPPER_LEFT[1], plane_p, plane_v)
        mask = pos2mask(rimgr, POS_UPPER_LEFT, plane_p, plane_v, HEIGHT, WIDTH)
        print("左上: ", opticalflow(rimgl, rimgr, dist, mask)/np.sum(mask))
        print("左上: ", opticalflow(rimgl, rimgr, dist, mask), np.sum(mask))
        POS_LOWER_LEFT[2] = planeZ(POS_LOWER_LEFT[0], POS_LOWER_LEFT[1], plane_p, plane_v)
        mask = pos2mask(rimgr, POS_LOWER_LEFT, plane_p, plane_v, HEIGHT, WIDTH)
        print("左下: ", opticalflow(rimgl, rimgr, dist, mask)/np.sum(mask))
        print("左下: ", opticalflow(rimgl, rimgr, dist, mask), np.sum(mask))
        POS_LOWER_RIGHT[2] = planeZ(POS_LOWER_RIGHT[0], POS_LOWER_RIGHT[1], plane_p, plane_v)
        mask = pos2mask(rimgr, POS_LOWER_RIGHT, plane_p, plane_v, HEIGHT, WIDTH)
        print("右下: ", opticalflow(rimgl, rimgr, dist, mask)/np.sum(mask))
        print("右下: ", opticalflow(rimgl, rimgr, dist, mask), np.sum(mask))
        POS_UPPER_RIGHT[2] = planeZ(POS_UPPER_RIGHT[0], POS_UPPER_RIGHT[1], plane_p, plane_v)
        mask = pos2mask(rimgr, POS_UPPER_RIGHT, plane_p, plane_v, HEIGHT, WIDTH)
        print("右上: ", opticalflow(rimgl, rimgr, dist, mask)/np.sum(mask))
        print("右上: ", opticalflow(rimgl, rimgr, dist, mask), np.sum(mask))

        # disp = rimgr.copy()
        # h, w = disp.shape[:2]
        # cv2.polylines(disp, [pos2points(disp, POS_UPPER_LEFT, plane_p, plane_v, h, w, lr="right")], color=(0,0,255), isClosed=True, thickness=5)
        # cv2.polylines(disp, [pos2points(disp, POS_LOWER_LEFT, plane_p, plane_v, h, w, lr="right")], color=(0,0,255), isClosed=True, thickness=5)
        # cv2.polylines(disp, [pos2points(disp, POS_LOWER_RIGHT, plane_p, plane_v, h, w, lr="right")], color=(0,0,255), isClosed=True, thickness=5)
        # cv2.polylines(disp, [pos2points(disp, POS_UPPER_RIGHT, plane_p, plane_v, h, w, lr="right")], color=(0,0,255), isClosed=True, thickness=5)
        # imshowS(disp, "Bounding Box")

    else:
        # Video Not Loaded
        print("ビデオのロードに失敗しました")

# --------------
# main 実行
if __name__ == "__main__":
    print("=============================")
    main()
    print("=============================")