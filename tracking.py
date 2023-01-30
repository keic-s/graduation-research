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
import time
# - sklearn -
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.linear_model import LinearRegression
# from sklearn import cluster, preprocessing, mixture #機械学習用のライブラリを利用
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
# - Human Segmentation -
from humanSegmentation import *

# ----
# プログラムの流れ
"""
・キャリブレーション，平行化(動画から)
・背景差分
・床推定
・
・検出
・追跡
・結果描画
"""

# -----
# 定数

from const import *

NOW = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# 動画選択
#VIDEO = videodata.MY_SCRFES_2
#VIDEO = videodata.MY_SCRFES_3
# VIDEO = videodata.MY_SCRFES_REFLECT
# VIDEO = videodata.HANA_CROSS_OCCLUDED
# VIDEO = videodata.HANA_CROSS_OCCLUDED_2
VIDEO = videodata.HANA_CROSS
# VIDEO = videodata.HANA_CROSSV2
# VIDEO = videodata.HANA_CROSS2
# VIDEO = videodata.HANA_CROSS3
#VIDEO = videodata.HANA_CROSS4
# VIDEO = videodata.HANA_PARALLEL
#VIDEO = videodata.HANA_PARALLEL_BEHIND
# VIDEO = videodata.HANA_PARALLEL_THREE
# VIDEO = videodata.HANA_CROSS_START

FOLDERNAME = os.path.join(VIDEO.filename, NOW, "Disp*.png")

FRAME_LENGTH = 90

# 人間
class Human:
    """人間クラス"""
    def __init__(self, pos, pc=None, mask=None, id=-1):
        """pos: 3次元位置"""
        self.mask = mask                # 人物領域のマスク
        self.pc = pc                    # 点群，
        self.pc_track = [pc]            # 点群追跡
        self.pos = pos                  # 人物の現在位置(x, y, z?)
        self.track = [pos]              # 追跡ルート
        self.id: int = id               # 人物ごとにID(0, 1, 2, ...)を付与
        self.color = (255, 0, 0)
        self.r: float = HUMAN_R              # 半径
        self.detected_frame_n:int = 0   # 検出されたフレーム
        self.exist = True

    def __str__(self) -> str:
        return "pos: "+str(self.pos)+", "

    def connectToAnotherHuman(self, human):
        return

    def update(self, height, human_list):
        #self.r = r
        newpos, maxArea = self.search(height, human_list)

        # 更新
        self.pos = newpos
        self.track.append(newpos)
        self.pc = height[maxArea]
        
        #deleteArea = self.getWithinCircleArea(pos=newpos, height=height, r=self.r*CORE_RADIUS_COEF)
        #deleteArea = self.getWithinCircleArea(pos=newpos, height=height, r=self.r)
        #height[deleteArea] = np.nan

    def getWithinCircleArea(self, pos:np.array, height:np.array, r:float) -> np.array:
        """posを中心にr以内の点にtrueを返す"""
        h, w = height.shape[:2]
        center = np.zeros((h, w, 2))
        center[:, :, 0] = pos[0]
        center[:, :, 1] = pos[1]
        dist = np.linalg.norm(height[:, :, :2]-center, axis=2)
        #print("dist : ", dist)
        area = dist <= r
        return area
    
    def getEntryDirection(self):
        """一定速度以上であるとき移動している方向を返す"""
        # return None
        velocity_thsh = 0.02
        return None # テスト

        if len(self.track) < 3:    # 推測出来ない場合
            return None

        # 方向予測
        direction_list = np.array([
            [1., 0.],
            [1., 1.],
            [0., 1.],
            [-1., 1.],
            [-1., 0.],
            [-1., -1.],
            [0., -1.],
            [1., -1.]
            ])
        #print(self.track[-1], self.track[-2])
        d = self.track[-1]-self.track[-2]
        #print(d)
        #print(np.linalg.norm(d))

        if np.linalg.norm(d) <= velocity_thsh:  # 一定の速度以下のときは全方向を探索
            return None
        
        theta = np.arctan2(d[1], d[0])+np.pi   # [-pi, pi] -> [0, 2pi]
        idx = int(4*(theta)/np.pi)%len(direction_list)
        direction = direction_list[idx]
        return direction

    def search(self, height:np.array, human_list) -> list[np.array, np.array]:
        """探索"""
        #newPos, maxArea = self.search_around(height)   # 予測して周辺を探索
        newPos, maxArea = self.search_8neighbor(height, human_list) # 移動方向から8近傍で方向を絞って探索
        return newPos, maxArea

    def search_8neighbor(self, height:np.array, human_list) -> list[np.array, np.array]:
        """8近傍で探索
        現在位置の周辺8近傍領域を探索して n*sigmaの範囲にある点数が最も大きいところと結びつけ
        posとエリアを返す"""
        # 初期化
        direction_list = np.array([
            [1., 0.],
            [1., 1.],
            [0., 1.],
            [-1., 1.],
            [-1., 0.],
            [-1., -1.],
            [0., -1.],
            [1., -1.]
            ])
        
        entryDirection = self.getEntryDirection()

        # 進入方向
        if not (entryDirection is None):
            len_ = len(direction_list)

            entryDirection = np.array([1., -1.])
            repeatEntryDirection = np.tile(entryDirection, (len_, 1))
            #print(repeatEntryDirection)

            idx = np.where(np.all(np.isclose(direction_list, repeatEntryDirection), axis=1))[0]
            #print(idx)

            direction_list_to_search = np.delete(direction_list, obj=[(idx-1)%len_, idx%len_, (idx+1)%len_], axis=0)
            #print(direction_list_to_search)
        else:
            direction_list_to_search = direction_list

        # その場の値をデフォルトに
        pos = self.pos
        pos = self.predictNextPoint()     # 移動先を予測
        maxPos = np.array(pos)[:2]
        maxArea = self.getWithinCircleArea(pos=maxPos, height=height, r=self.r)
        maxNum = np.count_nonzero(maxArea)
        
        for dir in direction_list_to_search:
            for dist in DIST_LIST:
                x, y = dir*dist
                currentPos = np.array([self.pos[0] + x, self.pos[1] + y])
                if self.isOverlap(currentPos, human_list):  # 重なってるときはパス
                    continue
                area = self.getWithinCircleArea(pos=currentPos, height=height, r=self.r)
                num = np.count_nonzero(area)
                if num > maxNum:
                    maxNum = num
                    maxPos = currentPos
                    maxArea = area

        return np.array([*maxPos, 0.]), maxArea

    # def search_around(self, height):
    #     """posの周辺領域を探索して n*sigmaの範囲にある点数が最も大きいところと結びつけ
    #     posとエリアを返す
        
    #     X0.03
    #     """
    #     SEARCH_LIST2 = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    #     SEARCH_LIST3 = [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03]

    #     SEARCH_LIST = SEARCH_LIST3

    #     pos = self.predictNextPoint()
        
    #     maxPos = np.array([None, None])
    #     maxNum = -1
    #     maxArea = None
    #     h, w = height.shape[:2]
        
    #     for x in SEARCH_LIST:
    #         for y in SEARCH_LIST: # 周辺 n x n 領域を比較
    #             currentPos = np.array([pos[0] + x, pos[1] + y])
    #             #print("dist : ", dist)
    #             area = Human.getWithinCircleArea(pos=currentPos, height=height, r=self.r)
    #             num = np.count_nonzero(area)
    #             if num > maxNum:
    #                 maxNum = num
    #                 maxPos = currentPos
    #                 maxArea = area
        
    #     return np.array([*maxPos, 0.]), maxArea

    def predictNextPoint(self, numPointsUsedToPredictNext = 3):
        """次の点を予測する"""
        # return None # その場を返してみる
        if numPointsUsedToPredictNext == 1:
            return self.pos
        elif numPointsUsedToPredictNext == 2:
            if len(self.track) <= 1:
                return self.predictNextPoint(1)
            else:
                # 2つ前と1つ前を2対1に外分
                nextPos = self.track[-1]+(self.track[-1]-self.track[-2])#*0.8
                return nextPos
        else:   # 3点以上を使うとき
            if len(self.track) <= 1:    # 次の点が予測できないとき
                return self.predictNextPoint(1)
            elif len(self.track) == 2:
                return self.predictNextPoint(2)
            else: # 3点以上
                return self.predictNextPoint(2)   # 3点以上でも2点で推測
                nextPos = self.pos + (self.track[-1]-self.track[-2])+(self.track[-1]-self.track[-2])-(self.track[-2]-self.track[-3])    # 加速度もふまえて
                return nextPos

    def isOverlap(self, pos, human_list) -> bool:
        for human in human_list:
            if human.id == self.id:
                continue
            if np.linalg.norm(human.pos[:2]-pos) < self.r*CORE_RADIUS_COEF*2.:
                return True
        return False

################################
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
    plot_dst = ax.scatter(np.ravel(height[:, :, 0]), np.ravel(height[:, :, 1]), s=0.05, alpha=1, color="blue")

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

def clusterPlot(data, labels, winName="Clustering 2D Plot", lims=None, showFlag=True, saveFlag=False, frame_n=None):
    """無限遠点 上方から見た点群描画"""
    # 軸の設定
    plt.cla()
    fig_cl = plt.figure(num=winName, figsize=(9,6))
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
    return lims

def dispHumanPos(height, humanList:list[Human], img_l:np.array, winName="2D Points and Pos and Radius", lims=None, showFlag=False, saveFlag=False, frame_n=None):
    """無限遠点 上方から見た点群+人間位置，半径描画"""
    # 準備
    plt.clf()
    fig_dst, axes_dst = plt.subplots(nrows=1, ncols=2, num=winName, figsize=(10,4), tight_layout=True)
    # fig_dst = plt.figure(num=winName, figsize=(9,6))
    # ax_dst[0, 1] = fig_dst.add_subplot(111)
    plt.grid(linewidth=0.2)
    # ax.invert_yaxis()

    img_l_rgb = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)

    # 図左: 画像
    axes_dst[0].axes.xaxis.set_visible(False)    # x軸非表示
    axes_dst[0].axes.yaxis.set_visible(False)    # x軸非表示
    axes_dst[0].imshow(img_l_rgb)  # 画像表示
    # 円筒を描画

    # 図右: 2次元グラフ
    axes_dst[1].scatter(np.ravel(height[:, :, 0]), np.ravel(height[:, :, 1]), s=0.05, alpha=0.3, color="lightgray")       # 点群全て
    for human in humanList:
        if not human.exist:
            pass
        # 色設定
        if False:
            pcColor = COLOR_LIST[human.id]
            trackColor = "black"
            circleColor = "black"
        else:
            pcColor = "gray"
            trackColor = COLOR_LIST[human.id]
            circleColor = "black"
        if True:    # 軌跡も描画
            x = np.array(human.track)[:,0]
            y = np.array(human.track)[:,1]
            axes_dst[1].scatter(x, y, marker=".", s=4., color=trackColor, alpha=0.8)
            axes_dst[1].plot(x, y, marker="", color=trackColor, alpha=0.5)
        axes_dst[1].scatter(np.ravel(human.pc[:, 0]), np.ravel(human.pc[:, 1]), s=0.05, alpha=1., color=pcColor)    # 点群描画
        axes_dst[1].text(human.pos[0]-(human.r), human.pos[1]+(human.r), str(human.id))     # id描画
        # axes_dst[1].text(human.pos[0], human.pos[1]+(human.r)+0.1, f"({human.pos[0]:.2f},{human.pos[1]:.2f})", size=6)     # 位置座標
        axes_dst[1].scatter(human.pos[0], human.pos[1], s=2., alpha=1., color=circleColor, marker="x")                   # 中心描画
        circle = patches.Circle(human.pos, radius=human.r, lw=1.5, fill=False, color=circleColor)    # 円描画
        axes_dst[1].add_patch(circle)

    # if not (frame_n is None):
    #     plt.text(lims[0, 0], lims[0, 1], str(frame_n))
    #print(roundInf(ax_2d.get_xlim()))

    # Axes
    if lims is None:
        lims = [roundInf(axes_dst[1].get_xlim()), roundInf(axes_dst[1].get_ylim())]
    axes_dst[1].set_xlim(lims[0])
    axes_dst[1].set_ylim(lims[1])

    axes_dst[1].set_aspect('equal')
    axes_dst[1].set_xlabel("X (m)")
    axes_dst[1].set_ylabel("Y (m)")

    # Save
    if saveFlag:
        if frame_n is None:
            filename = "0"
        else:
            filename = str(frame_n)
        filename = NOW + "_" + filename
        if not os.path.exists("./dst/picture/"+VIDEO.filename+"/"+NOW):
            os.makedirs("./dst/picture/"+VIDEO.filename+"/"+NOW)
        plt.savefig("./dst/picture/"+VIDEO.filename+"/"+NOW+"/"+"DispHuman"+filename+".png")   # フレーム番号
        #FIG_2D_LIST.append([plot_dst])

    # Display
    if showFlag:
        plt.show()

    return lims

def disp2dtrack(human_list:list[Human], winName="Human 2D Track Plot", lims=None, showFlag=True, saveFlag=True, frame_n=None):
    """無限遠点 上方から見た追跡の軌跡描画"""
    # 軸設定
    plt.cla()
    fig_track = plt.figure(num=winName, figsize=FIG_SIZE)
    ax_track = fig_track.add_subplot(111)
    plt.grid(linewidth=0.2)

    # 描画
    for human in human_list:
        x = np.array(human.track)[:,0]
        y = np.array(human.track)[:,1]
        # print("x: ", x, "\ny: ", y)
        
        # 色設定
        if False:
            pcColor = COLOR_LIST[human.id]
            trackColor = "black"
            circleColor = "black"
        else:
            pcColor = "gray"
            trackColor = COLOR_LIST[human.id]
            circleColor = "black"

        # 点群描画
        ax_track.scatter(np.ravel(human.pc[:, 0]), np.ravel(human.pc[:, 1]), s=0.05, alpha=0.1, color=pcColor)
        ax_track.scatter(human.pos[0], human.pos[1], s=2., alpha=1., color=circleColor, marker="x")                   # 中心描画
        circle = patches.Circle(human.pos, radius=human.r, lw=1.5, fill=False, color=circleColor)    # 円描画
        ax_track.add_patch(circle)
        plt.text(human.pos[0]-human.r, human.pos[1]+human.r, str(human.id))
        # plt.text(human.pos[0]-human.r, human.pos[1]+human.r, "id:"+str(human.id))     # id描画

        # 追跡軌跡描画
        if False:   # マーカーの透明度を変える
            if False:
                for i in range(len(x)):
                    p = 0.9
                    alpha = (float(i+1.)/len(x))*p+(1.-p)
                    ax_track.scatter(x[i], y[i], marker=".", s=4., color=trackColor, alpha=alpha)
            else:
                color = np.linspace(0.1, 1.0, len(x))
                ax_track.scatter(x, y, marker=".", s=4., color=color, alpha=0.8, cmap="Greys")
        else:
            ax_track.scatter(x, y, marker=".", s=4., color=trackColor, alpha=0.8)
            pass
        ax_track.plot(x, y, marker="", color=trackColor, alpha=0.5)
        
    #print(roundInf(ax_2d.get_xlim()))

    # 軸設定
    if lims is None:
        lims = [roundInf(ax_track.get_xlim()), roundInf(ax_track.get_ylim())]
    ax_track.set_xlim(lims[0])
    ax_track.set_ylim(lims[1])

    ax_track.set_aspect('equal')
    ax_track.set_xlabel("X (m)")
    ax_track.set_ylabel("Y (m)")
    
    # 保存
    if saveFlag:
        if frame_n is None:
            filename = "0"
        else:
            filename = str(frame_n)
        filename = NOW + "_" + filename
        if not os.path.exists("./dst/picture/"+VIDEO.filename+"/"+NOW):
            os.makedirs("./dst/picture/"+VIDEO.filename+"/"+NOW)
        plt.savefig("./dst/picture/"+VIDEO.filename+"/"+NOW+"/"+"2dTrackPlot"+filename+".png")   # フレーム番号
        #FIG_2D_LIST.append([plot_dst])
    
    # 表示
    if showFlag:
        plt.show()
    return lims

################################

def setRadius(human_list:list[Human]) -> None:
    r = getRadius(human_list=human_list)
    human_r = r
    for human in human_list:
        human.r = r

def updateHumanList(human_list:list[Human], height:np.array, lims:np.array, n_frame:int, mask:np.array):
    print("Before Update: ", SW.lap())  # updateにかかる時間

    # 初期化
    height_to_newly_detect = height.copy()
    #human_list.sort(key= lambda human:human.pos[1])        # 手前からでソート？
    # dispHeight(height_to_newly_detect[:, :, 2])

    # ループ
    for human in human_list:
        if not human.exist:
            pass
        human.update(height, human_list)
        if (human.pos[0] < lims[0][0]) or (human.pos[0] > lims[0][1]) \
            or (human.pos[1] < lims[1][0]) or (human.pos[1] > lims[1][1]): 
            human.exist = False     # 画面外へ出た人間を消す
        if len(human.pc) < HUMAN_MINPOINT_COEF*HUMAN_NUMPOINT: 
            human.exist = False     # 画面外へ出た人間を消す
        area = human.getWithinCircleArea(pos=human.pos, height=height_to_newly_detect, r=human.r)
        height_to_newly_detect[area, :] = np.nan

    # dispHeight(height_to_newly_detect[:, :, 2], winName="Human Removed")
    
    numPointsOfHuman = 50
    if n_frame % 100 == 0:    # 5フレームに一回新規検出
        print(f"frame {n_frame}: Newly Detection Processing...")
        # height_to_newly_detect[np.where(mask.T < 0.1), :] = np.nan    # 動いてないところにマスク
        # dispHeight(height_to_newly_detect[:, :, 2], winName="Masked Height")
        if np.count_nonzero(~np.isnan(height[:, :, 2])) > numPointsOfHuman: #if height_to_track にポイントが残ってたら
            # Detect Newly
            #disp3d(height, winTitle="Newly Detection", dispAngleMode="height")
            print("Newly Detecting is Running...\n", np.count_nonzero(~np.isnan(height[:, :, 2])), "Points Found")
            newHumanList = detectNewHuman(height_to_newly_detect, lims=lims)
            # disp3d(height_to_newly_detect)
            if len(newHumanList) >= 1:
                human_list.extend(newHumanList)

    print("After Update: ", SW.lap())   # updateにかかる時間

def detectNewHuman(height, lims) -> list[Human]:
    newHumanList, _ = humanSegmentation(height, lims=lims, contourFlag=False)
    return newHumanList

def _3dPointTo2d(point, floor_p, floor_v):
    p = point+floor_p
    _2dpos_u = p[0]/p[2]
    _2dpos_v = p[1]/p[2]
    return (_2dpos_u, _2dpos_v)

def frameDiff(prevframe, curframe, nextframe, openingFlag):
    """3枚の画像から動体を検出
    リアルタイムでの利用を考慮すると、nextframeが現在のフレーム
    マスクは0.0 or 1.0"""

    if len(prevframe.shape) > 2:
        # グレースケール化
        prevframe = cv2.cvtColor(prevframe,cv2.COLOR_BGR2GRAY)
        curframe = cv2.cvtColor(curframe,cv2.COLOR_BGR2GRAY)
        nextframe = cv2.cvtColor(nextframe,cv2.COLOR_BGR2GRAY)
    #imshowS(curgray)

    # 前フレームと現フレーム、現フレームと次フレームの差をとる
    prevmask = cv2.absdiff(prevframe, curframe)
    nextmask = cv2.absdiff(curframe, nextframe)

    mask = cv2.bitwise_and(prevmask, nextmask)
    mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
    
    # 画像のモルフォロジー処理
    if openingFlag:
        n = 3
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel(n))
        n = 5
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel(n))
        #n = 1
        #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel(n))
    
    mask = mask/255.0

    return mask

def getRadius(human_list: list[Human]): #  -> (list[(float, float)], float, float)
    """人間のリストから半径(分散)を出す
    return r"""
    #r = getRadiusByVariance(human_list)
    #r = getRadiusByNumPoint(human_list)    $ ダメ
    r = getRadiusByArea(human_list)
    #print(r, ":r")
    return r*RADIUS_ADJUSTMENT      # Test

def getRadiusByVariance(human_list:list[Human]) -> float:
    """距離の平均を使って半径を出す"""
    r = 0.
    #print("len(human_list): ", len(human_list))
    for human in human_list:
        human_v = human.pc.reshape(-1, 3) # [x y z]が要素数ぶんだけ連なる形へ変形
        human_v = human_v[~np.isnan(human_v).any(axis=1)]
        human_xy = human_v[:, :2]

        # 平均を求める
        center = np.average(human_xy, axis=0)
        #print("center", center)
        #center_list.append(center)
        # 全分散を求める
        dist = np.power(np.linalg.norm(human_xy-center, axis=1), 2)
        #print("dist : ", dist)
        s = np.average(dist)
        r += s
        #human.r = s
        #covmatrix = np.cov(human_xy.T)
    #print(covmatrix)
    r /= float(len(human_list))
    #print(r, ":r^2")
    r = np.sqrt(r)
    return r*2. # 95%区間

# def getRadiusByNumPoint(human_list):
#     """画素数から半径を出す(円の面積の公式)
#     S = pi*r*r"""
#     S = 0.
#     for human in human_list:
#         S += np.count_nonzero(np.logical_not(np.isnan(human.pc[:,2])))
#         print("S:", S)
#     r = np.sqrt(S/np.pi/len(human_list))
#     print("r:", r)
#     return r

def getRadiusByArea(human_list):
    """画素数から半径を出す(円の面積の公式)
    S = pi*r*r"""
    r = 0.
    for human in human_list:
        dx = np.nanmax(human.pc[:,0])-np.nanmin(human.pc[:,0])
        dy = np.nanmax(human.pc[:,1])-np.nanmin(human.pc[:,1])
        r += dx + dy
    r = r / 4. / len(human_list) 
    print("r:", r)
    return r

# -------------------------------------------------------------------------------------
# 関数
def tracking(video: Video, save_flag=False):
    """ビデオから読み込み、追跡"""

    vid = video.vid

    # パラメータ
    w = 1080 # frameS表示用サイズ

    # fig_2d = plt.figure(num="2D Plot", figsize=(9,6))
    # ax_2d = fig_2d.add_subplot(111)
    # ax_2d.set_xlabel("X (m)")
    # ax_2d.set_ylabel("Y (m)")
    # ax_2d.invert_yaxis()
    # plt.grid(linewidth=0.2)
    #ax_2d.invert_yaxis()
    fig_2d_list = []
    #fig_3d = plt.figure(num="3D Plot", figsize=(9,6))
    #ax_3d = fig_3d.add_subplot(111)
    #fig_3d_list = []

    # ----
    # ループ前準備(地面推定，人物初期化)
    preLoopFrameN = 10
    depth_vector = None
    preLoopDispFlag = True
    
    current_rimgl, current_rimgr = None, None
    next_rimgl, next_rimgr = None, None

    for frame_n in range(preLoopFrameN):
        print("Pre Loop Processing... (", frame_n+1, "/", preLoopFrameN, ")")

        ret, frame = vid.read()
        if not ret:
            break

        # 真ん中がマスク除去対象
        current_rimgl, current_rimgr = next_rimgl, next_rimgr

        # キャリブレーション，平行化
        imgl, imgr = devideStereo(frame)
        next_rimgl, next_rimgr = calibration.stereoRemap(imgl, imgr, video.params.maps)
        
        # imshowPlt(next_rimgl, winName="Image Left")

        # 深度情報取得
        if current_rimgl is None:
            continue

        depth = stereo3d(current_rimgl, current_rimgr, video.params)
        print(f"重心: {np.nanmean(depth[:, :, 0])}, {np.nanmean(depth[:, :, 1])}, {np.nanmean(depth[:, :, 2])}")
        # disp3d(depth)

        # オクルージョン発生時頭と肩の位置 ----
        # imshowPlt(next_rimgl[300:700, 1000:1200], winName="Image Left")
        # imshowPlt(depth[300:700, 1000:1200, 2], winName="Image Left")
        # dispScatter(depth)
        # disp3d(depth[300:700, 1000:1200])
        # disp3d(depth[300:700, 700:1200])
        # ----

        # (0, 0)はどこ？？ ----
        h, w = depth.shape[:2]
        # print("Test\n", depth[int(h/2)-1, int(w/2)-1, :2], depth[int(h/2)-1, int(w/2), :2], "\n", \
        #     depth[int(h/2), int(w/2)-1, :2], depth[int(h/2), int(w/2), :2])
        # print("Test\n", depth[int(h/2), int(w/2), :2], depth[int(h/2), int(w/2)+1, :2], "\n", \
        #     depth[int(h/2)+1, int(w/2)+1, :2], depth[int(h/2)+1, int(w/2)+1, :2])
        #print("Test2\n", depth[0, 0, :2])
        # fig_2d = plt.figure(num="dawfaafw")
        # ax_2d = fig_2d.add_subplot(111)
        # ax_2d.plot()
        # abs_ = np.abs(depth[:, :, 0])+np.abs(depth[:, :, 1])
        # idx = np.unravel_index(np.nanargmin(abs_), depth.shape[:2])
        # print("abs min", idx, " value: ", abs_[idx], idx[0]*2, idx[1]*2)
        # print("Test\n", depth[idx[0], idx[1], :2], depth[idx[0], idx[1]+1, :2], "\n", \
        #     depth[idx[0]+1, idx[1], :2], depth[idx[0]+1, idx[1]+1, :2])
        # print("Test\n", depth[idx[0]-1, idx[1]-1, :2], depth[idx[0]-1, idx[1], :2], depth[idx[0]-1, idx[1]+1, :2], "\n", \
        #     depth[idx[0], idx[1]-1, :2], depth[idx[0], idx[1], :2], depth[idx[0], idx[1]+1, :2], "\n", \
        #     depth[idx[0]+1, idx[1]-1, :2], depth[idx[0]+1, idx[1], :2], depth[idx[0]+1, idx[1]+1, :2])
        # ----

        try:
            prevframe = curframe
        except NameError:
            prevframe = None
        try:
            curframe = nextframe
        except NameError:
            curframe = None

        nextframe = next_rimgl

        if (prevframe is None) or (curframe is None):
            continue

        mask = frameDiff(prevframe, curframe, nextframe, openingFlag=False)
        # imshowS(mask)   # Debug

        depth[np.where(mask > 0)] = np.nan  # マスクが0でないところ、動体領域のところにnanを代入
        #imshowMask(curframe, reverseBW(mask), wait=True, saveFlag=True)
        #disp3d(depth, winTitle="Masked Depth")
        depth_vector_frame = depth.reshape(-1, 3) # [x y z]が要素数ぶんだけ連なる形へ変形
        depth_vector_frame = depth_vector_frame[~np.isnan(depth_vector_frame).any(axis=1)]

        if depth_vector is None:
            depth_vector = depth_vector_frame
        else:
            depth_vector = np.concatenate([depth_vector, depth_vector_frame])

    # if preLoopDispFlag:
    #     # フレーム間差分のマスクを取ったdepth
    #     fig_3d = plt.figure(num="Depth Masked Scatter", figsize=(9,6))
    #     ax_3d = fig_3d.add_subplot(111, projection='3d')
    #     depth_vector_disp = np.random.choice(depth_vector, size=1000, replace=False, axis=0)
    #     ax_3d.scatter(depth_vector_disp[:, 0], depth_vector[:, 1], depth_vector[:, 2], s=0.05, alpha=1, c="blue")
    #     plt.show()

    #imshowS(curframe, "Pre Loop Frame", wait=True, saveFlag=True)

    # 地面推定
    #print(depth_vector)
    floor_p, floor_v = floorEstimationRepeat(depth_vector)
    #disp3d(depth=depth, floor_p=floor_p, floor_v=floor_v, winTitle="Floor Estimation")

    # 高さ
    depth = stereo3d(current_rimgl, current_rimgr, video.params)
    # 表示用
    depth_disp = depth
    # depth_disp, mask = floorRemoval(depth, floor_p, floor_v, remove="human", correction=0.2)
    # imshowMask(current_rimgl, mask)
    # depth_disp[np.where(mask==0)] = np.nan
    #depth_disp, mask = floorRemoval(depth, floor_p, floor_v, remove="floor", correction=0.2)
    # disp3d(depth_disp, floor_p=floor_p, floor_v=floor_v, winTitle="Floor", video=VIDEO)
    # depth_disp[:50, :, :] = np.nan  # 表示用, 上の除去
    # depth_disp[800:, :, :] = np.nan  # 表示用, 下の除去
    # depth_disp[:, :340, :] = np.nan  # 左の除去
    # depth_disp[:, 1650:, :] = np.nan  # 表示用, 右の除去
    # disp3d(depth_disp, floor_p=floor_p, floor_v=floor_v, winTitle="Floor Display", video=VIDEO)
    # disp3d(depth_disp, winTitle="3D Point Cloud Display", video=VIDEO)

    #disp3d(distFromPlane(depth, floor_p=floor_p, floor_v=floor_v), winTitle="Height", video=VIDEO, dispAngleMode="height")

    # mask -> floorRemoval -> HeightRotate
    mask = frameDiff(prevframe, curframe, nextframe, openingFlag=True)
    #imshowMask(curframe, mask)
    depth[np.where(mask == 0)] = np.nan  # マスクが0のところ、動体領域でないところにnanを代入
    # disp3d(depth, winTitle="Masked Depth", video=VIDEO)

    # 地面除去
    depth2, mask = floorRemoval(depth, floor_p, floor_v, remove="floor", correction=0.05)
    # disp3d(depth2, winTitle="Ground Removal", video=VIDEO)

    height = distFromPlane(depth2, floor_p, floor_v)
    # disp3d(height, winTitle="Height", video=VIDEO, dispAngleMode="height")
    # disp2d(height, showFlag=True, saveFlag=False)

    # ----
    # 初期設定
    #imshowS(curframe)
    human_list: list[Human]
    human_list, lims = humanSegmentation(height, img_l=current_rimgl, frame_n=frame_n)
    #dispHumanPos(height=height, humanList=human_list, winName="Before Loop")

    # ループシステム選択
    frame_n = 0
    frameLength = FRAME_LENGTH

    while True:
        frame_n += 1
        if not LOOP_MODE_WHILE_TRUE:
            if frameLength <= frame_n+1:
                break
    # for frame_n in range(1, frameLength+1): 
        # メインループ
        print("\n Processing... [", int(frame_n/(frameLength/10.))*"#",int((frameLength-frame_n)/(frameLength/10.))*" ", "] (", frame_n, "/", frameLength, ")\n")
        SW.lap()

        ret, frame = vid.read()
        if not ret:
            break

        # 真ん中がマスク除去対象
        current_rimgl, current_rimgr = next_rimgl, next_rimgr

        # キャリブレーション，平行化
        imgl, imgr = devideStereo(frame)
        next_rimgl, next_rimgr = calibration.stereoRemap(imgl, imgr, video.params.maps)
        #imshowS(next_rimgl, winname="Image Left", wait=True, saveFlag=False)

        if False: # 図
            fig_2_getPos = plt.figure(num="frame", figsize=(9,6))
            ax_2_getPos = fig_2_getPos.add_subplot(111)
            ax_2_getPos.imshow(cv2.cvtColor(rimgl, cv2.COLOR_BGR2RGB))
            #plt.show()

        # 深度情報取得
        if current_rimgl is None:
            continue

        bg_mask = BG_MODEL.apply(current_rimgl)
        imshowMask(bg_mask)

        depth = stereo3d(current_rimgl, current_rimgr, video.params)
        #disp3d(depth, video=video, winTitle="Depth 3D Data")

        try:
            prevframe = curframe
        except NameError:
            prevframe = None
        try:
            curframe = nextframe
        except NameError:
            curframe = None

        nextframe = next_rimgl

        if (prevframe is None) or (curframe is None):
            continue
        
        # ----
        # フレーム間差分
        mask = frameDiff(prevframe, curframe, nextframe, openingFlag=True)

        #disp3d(depth, video=video, floor_p=floor_p, floor_v=floor_v, winTitle="Depth 3D before FrameDiff Mask")
        #depth[np.where(mask == 0)] = np.nan  # マスクが0のところ、動体領域でないところにnanを代入      #testing
        #disp3d(depth, video=video, floor_p=floor_p, floor_v=floor_v, winTitle="Depth 3D With FrameDiff Mask")

        # ----
        # 地面除去
        #depth2, mask = floorRemoval(depth, floor_p, floor_v, remove="floor", correction=0.1)
        depth2, mask = floorRemoval(depth, floor_p, floor_v, remove="floor", correction=0.3)    # マスクないなら
        #disp3d(depth2, video=video, floor_p=floor_p, floor_v=floor_v, winTitle="Depth 3D After Floor Removal")

        # ----
        # カメラからの距離を地面からの高さへ変換
        height = distFromPlane(depth2, floor_p, floor_v)
        # dispHeight(height[:, :, 2])
        # disp3d(height, video=video, winTitle="Height 3D Data", dispAngleMode="height")
        
        #disp2d(height, showFlag=True, saveFlag=False)
        # _2dplot = ax_2d.scatter(np.ravel(height[:, :, 0]), np.ravel(height[:, :, 1]), s=0.05, alpha=1, color="blue")
        # plt.show()
        #fig_2d_list.append([_2dplot])
        #scatter = ax_3d.scatter(height[:, :, 0], height[:, :, 1], height[:, :, 2], s=0.2, alpha=1)
        #surface = disp3d(height, video=video, winTitle="Height", showFlag=False, dispAngleMode="height")
        #plt.show()

        if False:    # 尾を引いているのはどこ？
            h, w = rimgl.shape[:2]
            testmask = np.zeros((h, w))
            # 尾の部分
            area = (np.where(height[:,:,0] > -2.) \
                and np.where(height[:,:,0] < -1.5) \
                and np.where(height[:,:,1] > 0.3) \
                and np.where(height[:,:,1] < 1.))
            # 人物部分
            area = np.where(height[:,:,0] > -1.8) \
                and np.where(height[:,:,0] < -1.) \
                and np.where(height[:,:,1] > -0.2) \
                and np.where(height[:,:,1] < 0.5) 
            testmask[area] = 1
            print(testmask.shape) # -> (1080, 1920)
            #print(np.sum(testmask))
            imshowS(grayToColor(testmask)*imgl, winname="Test Mask")
            #imshowS(grayToColor(testmask)*imgl, winname="Test Mask")

        #fig_3d_list.append([surface])
        #imshowS(reverseBW(mask), winname="Mask", wait=False)


        if False:
            for human_new in human_list_new:
                #print(human_new.pos)
                fig_frame = plt.figure(num="Frame Human Position", figsize=(9,6))
                ax_frame = fig_frame.add_subplot(111)
                #ax_frame.invert_yaxis()
                plt.grid(linewidth=0.2)
                ax_frame.plot(human_new.pos[0], human_new.pos[1], marker=".", color="blue", markersize=2)

        # if len(human_list)==0:    # 最初のフレーム
        #     human_list = humanSegmentation(height, img_l=current_rimgl, frame_n=frame_n, lims=lims)
        #     continue
        
        # ----
        # 追跡プロセス

        if len(human_list) == 0:    # リストが空のときは毎回新規検出
            human_list, lims = humanSegmentation(height, img_l=current_rimgl, frame_n=frame_n)
        else:
            updateHumanList(human_list, height, lims, frame_n, mask)

        dispHumanPos(height, human_list, img_l=current_rimgl, lims=lims, saveFlag=True, showFlag=False, frame_n=frame_n)
        # disp2dtrack(human_list=human_list, winName="Processing Frame "+str(frame_n), lims=lims, frame_n=frame_n, showFlag=False)

        # human_list_associate = human_list.copy()
        # human_list_new_associate = human_list_new.copy()

        #imshowS(rimgl)

        # if save_flag:
        #     writer.write(disp) # 画像を1フレーム分として書き込み

        if cv2.waitKey(DELAY) & 0xFF == ord('q'):
            print("Loop Quited")
            break

    # アニメーション保存
    # ani = animation.ArtistAnimation(fig_2d, fig_2d_list)
    # ani.save("./dst/tracking/2d_plot"+ datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".gif") # Anacondaではこっちを使う
    #ani.save("./dst/tracking/2d_plot"+ datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".gif", writer="imagemagick")
    # ani_3d = animation.ArtistAnimation(fig_3d, fig_3d_list)
    # ani_3d.save("./dst/tracking/3d_plot"+ datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".gif", writer="imagemagick")
    saveAnime(foldername=FOLDERNAME, now=NOW)

    # # 追跡結果描画
    human_list: list[Human]
    disp2dtrack(human_list, lims=lims, saveFlag=True, showFlag=True)
    
    # for i, human in enumerate(human_list):
    #     print(human_list)
    #     plotSingleTrack(human, color=colorList[i])

    # フレームごとに描画したやつ
    # ax_frame.set_xlim(*x_disp_area)
    # ax_frame.set_ylim(*y_disp_area)

    plt.show()

    return

# ----
# main
def main():
    # world = World()
    # world.start()

    video = VIDEO
    tracking(video, SAVE_FLAG)

    print("Track Completed!")

# --------------
# main 実行
if __name__ == "__main__":
    main()