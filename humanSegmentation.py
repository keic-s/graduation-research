# ----
# Modules
# - OS -
import os
# - CMath -
from cmath import isnan
from ctypes import pointer
from operator import ne
# - OpenCV -
import cv2
# - MatPlotLib -
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
# - Numpy -
import numpy as np
# - Datetime -
import datetime
# - sklearn -
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.linear_model import LinearRegression
from sklearn import cluster, preprocessing, mixture #機械学習用のライブラリを利用
# - glob -
import glob

# ----
# Modules I Made
# - Utilities -
from util import *
# - Parameters -
import zed2MyParameters
from zed2MyParameters import Params 
# - Video Data Class -
import videodata 
from videodata import Video
# - Stereo Pictures to 3D Depth Data -
from stereo3d import stereo3d, stereoVid3d 
# - Pre-focusing Method -
from prefocus import prefocus 
# - To Show Figures -
from disp import *
# - Calibration -
from calibration import devideStereo, getChessboardCorners 
import calibration
# - Get Floor(Ground) Information -
from floorEstimation import disp3d, floorEstimation, floorEstimationRepeat, floorRemoval, distFromPlane 
# - Animation Save -
from saveToVideo import saveAnime
from tracking import Human, VIDEO

# ----
# 定数
from const import *
SLEEP_TIME = 1

def getRadiusByHuman(human):
    dx = np.nanmax(human.pc[:,0])-np.nanmin(human.pc[:,0])
    dy = np.nanmax(human.pc[:,1])-np.nanmin(human.pc[:,1])
    return (dx + dy) / 4.

def getRadiusByArea(human_list):
    """縦横の長さから半径を出す(円の面積の公式)
    S = pi*r*r"""
    r = 0.
    for human in human_list:
        r += getRadiusByHuman(human)
    r = r / len(human_list) 
    print("r:", r)
    return r

def getRadius(human_list: list[Human]): #  -> (list[(float, float)], float, float)
    """人間のリストから半径(分散)を出す
    return r"""
    #r = getRadiusByVariance(human_list)
    #r = getRadiusByNumPoint(human_list)    $ ダメ
    r = getRadiusByArea(human_list)
    #print(r, ":r")
    return r*RADIUS_ADJUSTMENT      # Test

def setRadius(human_list:list[Human]) -> None:
    if len(human_list) == 0:
        return
    r = getRadius(human_list=human_list)
    global human_r
    human_r = r
    print("humanSegmentation\nhuman_r: ", r)
    for human in human_list:
        human.r = r

# ----
# 図に関する

def disp2d(height, winName="Human Area 2D Plot", lims=None, showFlag=False, saveFlag=True, frame_n=None, video:Video=None):
    """無限遠点 上方から見た点群描画"""
    # 軸の設定
    plt.cla()
    fig = plt.figure(num=winName, figsize=(9,6))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.grid(linewidth=0.2)

    # 描画
    #print(roundInf(ax_2d.get_xlim()))
    if len(height.shape) > 2:
        plot_dst = ax.scatter(np.ravel(height[:, :, 0]), np.ravel(height[:, :, 1]), s=0.05, alpha=1, color="blue")
    else:
        plot_dst = ax.scatter(np.ravel(height[:, 0]), np.ravel(height[:, 1]), s=0.05, alpha=1, color="blue")

    # 軸範囲
    if lims is None:    
        lims = [roundInf(ax.get_xlim()), roundInf(ax.get_ylim())]
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    #ax.invert_yaxis()
    
    # 保存
    if saveFlag:
        if frame_n is None:
            filename = "0"
        else:
            filename = str(frame_n)
        filename = NOW + "_" + filename
        if not os.path.exists("./dst/picture/"+VIDEO.filename+"/"+NOW):
            os.makedirs("./dst/picture/"+VIDEO.filename+"/"+NOW)
        plt.savefig("./dst/picture/"+VIDEO.filename+"/"+NOW+"/"+"2dplot"+filename+".png")   # フレーム番号
        #FIG_2D_LIST.append([plot_dst])

    # 表示
    if showFlag:
        plt.show()
    return lims

def clusterPlot(data, labels, winName="Clustering 2D Plot", lims=None, showFlag=True, waitFlag=False, saveFlag=False, frame_n=None):
    """無限遠点 上方から見た点群描画"""
    # 軸の設定
    plt.cla()
    fig_cl = plt.figure(num="Cluster Plot", figsize=FIG_SIZE)
    ax_cl = fig_cl.add_subplot(111)
    ax_cl.set_aspect('equal')
    ax_cl.set_xlabel("X (m)")
    ax_cl.set_ylabel("Y (m)")
    plt.grid(linewidth=0.2)
    ax_cl.invert_yaxis()

    # DBSCAN用，外れ値を除去
    data_outlier = data[np.where(labels == -1)]
    data[np.where(labels == -1)] = np.nan   
    
    # Plot
    plot_outlier = ax_cl.scatter(data_outlier[:, 0], data_outlier[:, 1], s=0.05, alpha=0.3, c="gray")
    plot_dst = ax_cl.scatter(data[:, 0], data[:, 1], s=0.05, alpha=1, c=labels)
    #print(roundInf(ax_2d.get_xlim()))

    # 表示範囲設定q
    if lims is None:
        lims = [roundInf(ax_cl.get_xlim()), roundInf(ax_cl.get_ylim())]
    ax_cl.set_xlim(lims[0])
    ax_cl.set_ylim(lims[1])

    # 保存
    if saveFlag:
        if frame_n is None:
            filename = "_"
        else:
            filename = str(frame_n)
        filename = NOW + "_" + filename
        if not os.path.exists("./dst/picture/"+VIDEO.filename+"/"+NOW):
            os.mkdir("./dst/picture/"+VIDEO.filename+"/"+NOW)
        plt.savefig("./dst/picture/"+VIDEO.filename+"/"+NOW+"/"+"clustering"+filename+".png")   # フレーム番号
        # plt.savefig("./"+VIDEO.filename+"/"+filename+".png")   # フレーム番号
        #FIG_2D_LIST.append([plot_dst])

    # 表示
    if showFlag:
        plt.show()

    if waitFlag:
        pass
    else:
        time.sleep(SLEEP_TIME)
        plt.close("all")
    return lims

# クラスタリング
def clusterData(data, lims=None, eps=0.05, min_samples=2000):
    labels = dbscan(data, eps, min_samples)
    #labels = meanshift(data)
    #labels = vbgmm(data)
    #n_components = np.unique(labels).size
    #clusterPlot(data=data, labels=labels, showFlag=True, lims=lims) # plot_dst
    return labels

def dbscan(data, eps, min_samples):
    """DBSCAN
    """
    print("----- DBSCAN ------")
    print("ポイントの数", len(data))
    print("eps=", eps, ", mis_samples=", min_samples)
    # 
    dbs = cluster.DBSCAN(eps=eps, min_samples=min_samples)
    dbs.fit(data)
    labels = dbs.labels_
    return labels

# def meanshift(data):
#     """Mean-shift"""
#     print("----- Mean-shift ------")
#     ms = cluster.MeanShift(seeds=data, bandwidth=None)
#     ms.fit(data)
#     labels = ms.labels_
#     return labels

# def vbgmm(data):
#     """Bayesian Gaussian Mixture"""
#     print("----- Bayesian Gaussian Mixture Method ------")
#     print(data)

#     seed = 3
#     max_iter = 1000 
#     times_thsh = 3.0
#     weights_thsh = 0.02
#     n_components = 10
#     while True:
#         if n_components == 1:
#             break
#         vbgm = mixture.BayesianGaussianMixture(n_components=n_components, random_state=seed, max_iter=max_iter)
#         vbgm=vbgm.fit(data)
#         labels=vbgm.predict(data)
#         #print(vbgm.weights_)
        
#         # 尤度グラフ
#         # plt.figure(111)
#         # x_tick =np.array([1,2,3,4,5,6,7,8,9,10])
#         print(vbgm.weights_, "--------------------------\n")
#         # plt.bar(x_tick, vbgm.weights_, width=0.5, tick_label=x_tick)
#         # plt.show()
        
#         print("n_components: ", n_components, "\n倍率: ", np.max(vbgm.weights_)/np.min(vbgm.weights_))
#         if np.max(vbgm.weights_)/np.min(vbgm.weights_) > times_thsh:
#             if np.count_nonzero(vbgm.weights_ < weights_thsh):      # 重みが weights_thsh を下回る分だけクラスター数を削減
#                 n_components -= np.count_nonzero(vbgm.weights_ < weights_thsh)
#             else:
#                 n_components -= 1
#         else:
#             break
#     return labels

# http://neuro-educator.com/ml12/


def humanSegmentation(height:np.array, img_l:np.array=None, frame_n:int=None, lims:np.array=None, contourFlag=True) -> list[list[Human],np.array]:
    """
    高さのデータから人間を一人ひとり分離? 検出?
    opencv findContoursを使う
    
    height: 高さ
    img_l: 画像描画用
    """
    print("----- Human Segmentation -----")

    global HUMAN_ID
    global HUMAN_NUMPOINT

    # _showFlag = True
    # if ~_showFlag:
    #     img_l = None

    if lims is None:
        lims = disp2d(height, winName="Initial 2d Plot", showFlag=False, saveFlag=False)  # 初期状態

    n: int
    humans: list = []   # 人間のリスト
    data = height.copy()

    #disp3d(height, dispAngleMode="height")
    #height, _ = floorRemoval(height, floor_p=(0,0,0), floor_v=(0,0,1),remove="floor")

    # all_human_area_mask = np.zeros_like(mask)
    #imshowS(mask, winname="Human Segmentation Mask")

    # 輪郭検出
    if contourFlag:
    # 小さい輪郭を誤検出として削除する
        mask = np.isnan(height[:, :, 2]).astype(np.float64)
        mask = reverseBW(mask)

        mask_gray = mask*255
        mask_gray = mask_gray.astype(np.uint8)
        contours, hierarchy = cv2.findContours(
            mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        contours = list(filter(lambda x: cv2.contourArea(x) > CONTOURS_THSH, contours))
        print("len(contours): ", len(contours))

        h, w, _ = height.shape
        human_area = np.zeros((h, w),np.uint8)
        cv2.drawContours(human_area, contours, -1, color=255, thickness=-1)
        human_area.astype(np.float64)
        human_mask = human_area/255.

        data[np.where(human_mask==0.)] = np.nan

    # 輪郭を描画する。
    if img_l is not None:
        img_disp = img_l.copy()

    # ----
    # 人物の分離

    if np.count_nonzero(np.logical_not(np.isnan(data[:,:,2])) ) < 1.:
        print("Human Not Found:-(")
        return [], lims

    #print(data[0, 0, :2])
    #print(data.shape)
    h, w, _ = data.shape
    data = data.reshape(h*w, -1)
    data = data[~np.isnan(data).any(axis=1)]
    #print(data[0])
    data_to_clustering = data[:, :2]
    disp2d(data_to_clustering, winName="Disp2d")

    # ----
    # クラスタリング
    # eps_list = [0.1, 0.2, 0.5, 0.8, 1.0, 2.0]
    # min_samples_list = [1000, 5000, 8000, 10000, 15000, 20000]
    # eps_list = [0.2]
    # min_samples_list = [5000, 8000]
    # for eps_ in eps_list:
    #     for min_samples_ in min_samples_list:
    #         data_to_clustering = data[:, :2]
    #         labels = clusterData(data_to_clustering, eps=eps_, min_samples=min_samples_)
    #         clusterPlot(data=data_to_clustering, lims=lims, labels=labels, showFlag=True, frame_n=frame_n) # plot_dst

    # クラスタリング =================
    labels = clusterData(data_to_clustering, eps=EPS, min_samples=MIN_SAMPLES, lims=lims)
    # clusterPlot(data=data_to_clustering, lims=lims, labels=labels, showFlag=True, frame_n=frame_n, saveFlag=False) # plot_dst

    # これの結果を使って

    #print("finish")

    # # 峰の数を使った方がいいよね．
    # # 縦方向に微分？エッジ検出

    humanNumPoint = 0

    for label in np.unique(labels):
        if label == -1:     # ノイズは除去
            continue
        pointCloud = data[np.where(labels==label)]
        human = Human(pos=np.average(data[labels==label], axis=0), pc=pointCloud, id=HUMAN_ID)
        human_r = getRadiusByHuman(human)
        if human_r < HUMAN_R * HUMAN_R_COEF:
            continue
        humans.append(human)
        print(human)
        HUMAN_ID += 1
        humanNumPoint += len(pointCloud)
    
    HUMAN_NUMPOINT = humanNumPoint / 4  # 人数，求める

    # 半径を定める
    # setRadius(humans)

    # print("len(humans): ", len(humans))
    # print(lims)

    return humans, lims      # humans