#################### モジュール #######################
from math import floor
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import datetime
import random
# ----
# 自作
#import zed2_hanaizumi_parameters as zed2
from disp import imshowS, resizeWithAspectRatio, grayToColor, imshowMask, reverseBW
from calibration import devideStereo, stereoRemap
from stereo3d import stereo3d, sgbm, disp2depth, MAX_SEARCH
# パラメータ関係
from zed2MyParameters import Params
# ビデオデータ関係
import videodata
from videodata import Video

#################### モード選択いろいろ #######################
# ----
# 動画から読み込むモード
VID_LOAD_MODE = True 
VIDEO = videodata.HANA_CROSS
VIDEO = videodata.HANA_PARALLEL
PARAMS = VIDEO.params

# ----
# 動画へ保存するモード
VID_SAVE_MODE = False
# ----
# 地面切り出しモード
TRIM_DEBUG_MODE = False
if TRIM_DEBUG_MODE:
    FLOOR_AREA = [180,900]
# ----
# 自動領域設定
AUTO_TRIM_MODE = True
# ----
# 床面除去
FLOOR_REMOVE_MODE = True

#################### 定数 #######################
DELAY = 1 # waitkey
CAM_ID = 1 # カメラID

#################### 関数 #######################
def floorEstimationRepeat(distance, img_l=None, n=3):
    """繰り返し処理"""
    #correction = [1.0, 0.5, 0.2, 0.1]
    correction = [3.0, 2.5, 2.0, 1.5, 1.0, 0.8, 0.5]
    floor_p, floor_v = floorEstimation(distance)
    #disp3d(distance, floor_p, floor_v, winTitle="処理前")
    for i in range(len(correction)):
        floor_p, floor_v = floorEstimation(distance)
        #print(i, "回目: ", floor_p, floor_v)
        
        distance, mask = floorRemoval(
            distance,
            floor_p,
            floor_v,
            remove="human",
            correction=correction[i]
            )
        
        # マスク
        #imshowS((grayToColor(mask)*img_l).astype(np.uint8), winname="Mask Removed", wait=False)
        
        if img_l:
            mask_left = (grayToColor(reverseBW(mask))*img_l).astype(np.uint8)
            mask_left[:, :MAX_SEARCH] = 0
            mask_left[:, img_l.shape[1]-MAX_SEARCH:] = 0
            #imshowS(mask_left, winname="Mask Left", wait=False)

        #disp3d(distance, floor_p, floor_v, winTitle=str(i+1)+"回目")

    # disp3d(distance, floor_p=floor_p, floor_v=floor_v, winTitle="Floor Estimation Result")

    return floor_p, floor_v

def floorEstimation(distance):
    """距離から地面の点と法線ベクトルを求める

    distance: 距離
    共分散行列(covariance matrix)
    """
    distance_v = distance.reshape(-1, 3) # [x y z]が要素数ぶんだけ連なる形へ変形
    distance_v = distance_v[~np.isnan(distance_v).any(axis=1)]
    covmatrix = np.cov(distance_v.T)
    #print(covmatrix)

    # 固有値, 固有ベクトルを求める
    eig = np.linalg.eig(covmatrix)[0]
    eigvec = np.linalg.eig(covmatrix)[1]
    #print("固有値: ", eig, "\n固有ベクトル: ", eigvec)

    floor_p = np.average(distance_v, axis=0)
    floor_v = eigvec[:,2]

    if floor_v[2] > 0:    # 法線ベクトルが負の方向(地面からカメラの方向)を向くように統一
        floor_v = -floor_v

    #print("通る点: ", floor_p, "\n法線ベクトル: ", floor_v)

    return floor_p, floor_v

def floorRemoval(depth, floor_p, floor_v, remove="floor", tall=3.0, correction=0.1, img_l=None):
    """距離と地面の平面からremove(地面or人間)を除去

    depth: 深度情報
    floor_p: 地面上のある点
    floor_v: 地面の法線ベクトル
    remove: 除去対象 floor / human / noise
    correction: 除去時地面をどれだけ浮かせるかのパラメータ
    """

    dstDepth = depth.copy()
    dstDepth = dstDepth.reshape(-1, 3) # [x y z]が要素数ぶんだけ連なる形へ変形
    #disp3d(depth, floor_p, floor_v, winTitle="Before Floor Removal 3D Disp")

    if len(depth.shape) > 2:
        h, w = depth.shape[:2]
        _len = h*w
    else:
        _len, _ = dstDepth.shape
    
    #print("dstDepth.shape: ", dstDepth.shape)

    derived_floor = planeZ(dstDepth[:,0], dstDepth[:,1], floor_p, floor_v)
    fill = np.nan # 埋めるやつ
    mask = np.zeros(_len).astype(np.float64) # 除去対象を除いた領域のマスク
    #mask = mask.reshape(h*w)
    maskfill = 1.0
    mask[np.isnan(dstDepth[:,2])] = maskfill # 既にnanのところ

    # 収縮しようとしてる
    mask = mask * 255
    mask = mask.astype(np.uint8)
    n = 5
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel(n))
    mask = mask / 255.
    # imshowMask(mask)

    if remove == "human": # 床面から一定の距離の領域を残す
        mask[np.where(np.abs(dstDepth[:,2] - derived_floor) >= correction)] = maskfill
        dstDepth[np.abs(dstDepth[:,2] - derived_floor) >= correction, :] = fill # 床面から距離のある点を除去
        #imshowS(mask, winname="Mask Default", wait=False)
    elif remove == "floor": # 床を除去，人間を残す
        mask[np.where(dstDepth[:,2] >= derived_floor-correction)] = maskfill
        mask[np.where(dstDepth[:,2] <= derived_floor-tall)] = maskfill
        dstDepth[dstDepth[:,2] >= derived_floor-correction, :] = fill # 床面とその下を除去
        dstDepth[dstDepth[:,2] <= derived_floor-tall, :] = fill # 人間より高いところを除去
    elif remove == "noise": # 床面も人も残しノイズを除去
        mask[np.where(dstDepth[:,2] >= derived_floor+correction)] = maskfill
        mask[np.where(dstDepth[:,2] <= derived_floor-tall)] = maskfill
        dstDepth[dstDepth[:,2] >= derived_floor+correction, :] = fill # 床面より下を除去
        dstDepth[dstDepth[:,2] <= derived_floor-tall, :] = fill # 人間より高いところを除去
    else: # remove == "noise"
        pass

    if len(depth.shape) > 2:
        dstDepth = dstDepth.reshape(h, w, 3)
        mask = mask.reshape(h, w)
    else:
        dstDepth = dstDepth.reshape(-1, 3)
        mask = mask.reshape(_len)

    #print(np.min(mask), np.max(mask))
    #mask = mask.astype("uint8")
    #imshowS(reverseBW(mask), winname="Floor Remove Mask", wait=True) #除去するマスクの表示
    # if not (img_l is None):
    #     imshowS((grayToColor(mask)*img_l).astype(np.uint8), winname="Mask Removed", wait=False)
        
    #     mask_left = (grayToColor(-(mask-1.0))*img_l).astype(np.uint8)
    #     mask_left[:, :MAX_SEARCH] = 0
    #     mask_left[:, img_l.shape[1]-MAX_SEARCH:] = 0
    #     imshowS(mask_left, winname="Mask Left", wait=False)
    
    #disp3d(depth, floor_p, floor_v, winTitle="After Floor Removal 3D Disp")

    return dstDepth, mask

def floorErode(height: np.array) -> np.array:
    """Height あるいは Depth を収縮
    nanの位置を用いて"""

    return

def disp3d(depth, video:Video=None, floor_p=None, floor_v=None, winTitle="3D Data", dispAngleMode="distance", showFlag = True):
    """3d図描画

    depth: 深度
    floor_p: 地面の点
    floor_v: 地面のベクトル
    """

    fig = plt.figure(num=winTitle, figsize=(9,6))
    ax = fig.add_subplot(111, projection='3d')
    #ax.set_title("Floor Estimation 3Ddst")
    # 軸ラベル設定
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    #ax.view_init(elev=-145, azim=-125)
    # 描画視点設定
    if dispAngleMode == "distance":
        ax.view_init(elev=-140, azim=-125)
    elif dispAngleMode == "height":
        ax.view_init(elev=40, azim=-60)
    elif dispAngleMode == "x-z":
        ax.view_init(elev=0, azim=-90)
    elif dispAngleMode == "y-z":
        ax.view_init(elev=0, azim=0)
    else:
        ax.view_init(elev=-140, azim=-125)
    #ax.view_init(elev=-120, azim=-90)
   
    xx = depth[:,:,0]
    yy = depth[:,:,1]
    zz = depth[:,:,2]

    # ステレオマッチングによる深度データ
    #surface = ax.plot_surface(xx, yy, zz, color="cyan", alpha=0.8, cmap="Grays")
    if dispAngleMode == "height":
        surface = ax.plot_surface(xx, yy, zz, alpha=0.8, cmap="Greys")
    elif dispAngleMode == "distance":
        surface = ax.plot_surface(xx, yy, zz, alpha=0.8, cmap="Greys_r")
    elif not (floor_v is None):
        surface = ax.plot_surface(xx, yy, zz, color="blue", alpha=0.8)
    else: #dispAngleMode == "distance":
        surface = ax.plot_surface(xx, yy, zz, alpha=0.8, cmap="Greys_r")
    #plt.colorbar()

    # 図表示範囲設定
    if AUTO_TRIM_MODE: # 自動表示範囲
        x_area = [np.nanmin(xx), np.nanmax(xx)]
        y_area = [np.nanmin(yy), np.nanmax(yy)]
        z_area = [np.nanmin(zz), np.nanmax(zz)]
        # print("x: ",np.min(xx), np.max(xx), 
        #     "\ny: ",np.min(yy), np.max(yy),
        #     "\nz: ",np.min(zz), np.max(zz))
    else:
        x_area = video.xArea
        y_area = video.yArea
        z_area = video.zArea

    ax.set_xlim3d(x_area)
    ax.set_ylim3d(y_area)
    ax.set_zlim3d(z_area)

    if not (floor_v is None):
        # floor_v が入力されたとき
        # 法線ベクトルから求めた平面を描画
        dstPlaneX, dstPlaneY = np.meshgrid(
            np.linspace(*x_area, 50),
            np.linspace(*y_area, 50)
            )
        dstPlaneZ = planeZ(dstPlaneX, dstPlaneY, floor_p, floor_v)
        ax.plot_surface(dstPlaneX, dstPlaneY, dstPlaneZ, color="red", alpha=0.2)

        if False:
            normal_v = np.tile(floor_p, (2, 1))
            #print(normal_v)
            normal_v[1] += floor_v*5
            line = art3d.Line3D(normal_v[:, 0], normal_v[:, 1], normal_v[:, 2], color="blue")
            ax.add_line(line)
        #ax.plot(normal_v[:, 0], normal_v[:, 1], normal_v[:, 2], marker="o-")

    if showFlag:
        plt.show()

    return surface

    # 描画
    if False:
        fig.add_axes(ax)
        fig.canvas.draw()
        buffer = np.array(fig.canvas.renderer.buffer_rgba())
        buffer = cv2.cvtColor(buffer, cv2.COLOR_RGBA2RGB)
        buffer = (buffer*255).astype(np.uint8)
        plt.show()
        return buffer

def distFromPlane(point, floor_p, floor_v):
    """平面からの距離と，平面上に下した垂線との交点

    point: 点(3次元)
    floor_p: 地面の点
    floor_v: 地面の法線ベクトル

    floor_v dot (point + k floor_v - floor_p) = 0
    """

    _dispProcess = False

    h, w = point.shape[:2]
    height = point.reshape((h*w, -1))   # Reshape

    z = planeZ(0., 0., floor_p=floor_p, floor_v=floor_v)
    origin = np.array([0., 0., z])

    # height = height - floor_p   # Move 2 Origin Corrdinates
    height = height - origin   # Move 2 Origin Corrdinates
    if _dispProcess:    # Show Process Result
        disp3d(height.reshape(point.shape), winTitle="Move 2 Origin Corrdinates")

    def rotate(mat, axis, degree):
        """
        axis 0 -> yz
        axis 1 -> xz
        axis 2 -> xy
        """
        if axis == 0:
            rotate_mat = np.array([
                [1., 0., 0.],
                [0., np.cos(degree), np.sin(degree)],
                [0., -np.sin(degree), np.cos(degree)]
                ])
        elif axis == 1:
            rotate_mat = np.array([
                [np.cos(degree), 0., np.sin(degree)],
                [0., 1., 0.],
                [-np.sin(degree), 0., np.cos(degree)]
                ])
        else:
            rotate_mat = np.array([
                [np.cos(degree), np.sin(degree), 0.],
                [-np.sin(degree), np.cos(degree), 0.],
                [0., 0., 1.]
                ])
        return np.dot(mat, rotate_mat)
    
    #np.arctan(floor_v[2]/floor_v[1])
    height = rotate(height, axis=0, degree=np.pi+np.pi/2-np.arctan(floor_v[2]/floor_v[1])) # Roll(x軸基準)
    if _dispProcess:    # Show Process Result
        disp3d(height.reshape(point.shape), winTitle="Rotate 1: Roll", dispAngleMode="height")
    height = rotate(height, axis=1, degree=np.pi/2-np.arctan(floor_v[2]/floor_v[0])) # Pitch(y軸基準) arctan(f2/f0)
    if _dispProcess:    # Show Process Result
        disp3d(height.reshape(point.shape), winTitle="Rotate 2: Pitch", dispAngleMode="height")
    # height = rotate(height, axis=2, degree=np.pi/2-np.arctan(floor_v[1]/floor_v[0])) # Yaw

    # 上下左右反転
    #height[:, 0] = - height[:, 0]
    #height[:, 1] = - height[:, 1]
    # 高さを反転
    #height[:, 2] = - height[:, 2]

    #print(height, "\n-------")
    height = height.reshape(point.shape)
    
    return height

def planeZ(x, y, floor_p, floor_v):
    """平面のz座標

    x: x座標
    y: y座標
    floor_p: 地面の点
    floor_v: 地面の法線ベクトル
    """
    z = floor_p[2] - (floor_v[0] * (x - floor_p[0]) + floor_v[1] * (y - floor_p[1])) / floor_v[2]
    return z

def removePlane(dist):
    """地面を曲面で近似"""
    return 

# ---- ラベル関係
def kernel(n):
    return np.ones((n,n),np.uint8)

# ラベルテーブルの情報を元に入力画像に色をつける
def put_color_to_objects(src_img, label_table):
    dst = src_img.copy()
    for label in range(label_table.max()+1):
        label_group_index = np.where(label_table == label)
        dst[label_group_index] = random.sample(range(255), k=3)
    return dst

def floorEstimationVid(video: Video, floorEstimationMode="start", save_flag=VID_SAVE_MODE):
    """地面の法線ベクトル，平面の式推定

    vid: ビデオクラス
    floorEstimationMode: 床推定をいつ行うか
        always: 毎時
        start: 開始時のみ
    save_flag: セーブフラグ
    """

    # パラメータ
    DISP_WIDTH = 480 # frameS表示用サイズ

    # 地面推定
    video.vid.set(cv2.CAP_PROP_POS_FRAMES, 980)
    #print("setできないね")
    ret, frame = video.read()
    img_l, img_r = devideStereo(frame)
    r_img_l, r_img_r = stereoRemap(img_l, img_r, video.params.maps)
    disparity = sgbm(r_img_l, r_img_r) # 視差
    if TRIM_DEBUG_MODE: # トリミングするとき
        distance = disp2depth(disparity[FLOOR_AREA[0]:FLOOR_AREA[1], :], video.params)
        imshowS(r_img_l[FLOOR_AREA[0]:FLOOR_AREA[1], :], wait=True)
    else: # トリミングしないとき
        distance = disp2depth(disparity, video.params)
    floor_p, floor_v = floorEstimationRepeat(distance, img_l=r_img_l)
    #distFromPlane(distance, floor_p, floor_v)

    # 動画保存の設定
    # if save_flag:
    #     testS = resizeWithAspectRatio(r_img_l, width = DISP_WIDTH)
    #     h, w = testS.shape[:2]
    #     now = datetime.datetime.now()
    #     fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
    #     outFilename = "./dst/PARAMC_ADJUST_VID_"+ now.strftime('%Y%m%d_%H%M%S') + ".mp4"
    #     writer = cv2.VideoWriter(
    #         outFilename,
    #         fmt,
    #         int(vid.get(cv2.CAP_PROP_FPS)),
    #         (w*2,h*2)
    #         ) # ライター作成
    
    # メインループ
    while True:
        ret, frame = video.read()
        if not ret:
            break

        img_l, img_r = devideStereo(frame)
        r_img_l, r_img_r = stereoRemap(img_l, img_r, video.params.maps)
        distance = stereo3d(r_img_l, r_img_r, video.params)

        distance90degree = distFromPlane(distance, floor_p, floor_v)
        floorRemoval(distance90degree, 
            np.array([0, 0, 0]), 
            np.array([0, 0, 1]),
            remove="noise")
        disp3d(distance90degree, winTitle="90-degree angle distance")
        
        distance, _ = floorRemoval(distance, floor_p, floor_v, remove="noise", correction=0.1)
        #disp3d(distance, floor_p, floor_v, winTitle="Distance")

        #imshowS(r_img_l, wait=False)
        if floorEstimationMode == "always":
            floor_p, floor_v = floorEstimationRepeat(distance)

        if False: # 距離の二次元Map
            #plt.imshow(depth[:,:,2], cmap='gray') # index 2 は距離
            plt.figure(figsize=(9,4))
            plt.imshow(distance[:,:,2]) # index 2 は距離
            plt.colorbar()
            plt.title('Distance Map')
            plt.show()

        if True: # 3次元グラフ
            #disp3d(distance, floor_p, floor_v, winTitle="Main Loop Distance")
            if FLOOR_REMOVE_MODE:
                #floor_p, floor_v = floorEstimationRepeat(distance)
                dst, mask_005 = floorRemoval(distance, floor_p, floor_v, remove="floor", correction=0.05)
                dst, mask_01 = floorRemoval(distance, floor_p, floor_v, remove="floor", correction=0.1)
                dst, mask_02 = floorRemoval(distance, floor_p, floor_v, remove="floor", correction=0.2)
            #disp3d(dst, floor_p, floor_v, winTitle="Floor Removed")
            #imshowS(figure)
            mask_005 = cv2.morphologyEx(mask_005, cv2.MORPH_OPEN, kernel(9))
            mask_01 = cv2.morphologyEx(mask_01, cv2.MORPH_OPEN, kernel(9))
            #mask_02 = cv2.morphologyEx(mask_02, cv2.MORPH_OPEN, kernel(9))
            #imshowS(mask_02, winname="0.2 m", wait=True)
            #print(mask.shape, mask.dtype)
            #retval, labels, stats, centroids = cv2.connectedComponentsWithStats(reverseBW(mask).astype(np.uint8))
            #dst = put_color_to_objects(r_img_l, labels)
            #dst = r_img_l*grayToColor(mask).astype(np.uint8)
            #print(np.min(dst), np.max(dst))
            #imshowS(reverseBW(mask_02), "Masked 0.2m", wait=False)
            #imshowS(r_img_l*grayToColor(reverseBW(mask_02)).astype(np.uint8), "Masked 0.2m", wait=False)
            #imshowS(put_color_to_objects(r_img_l, labels), "Labeling", wait=False)

        # ----
        # スクリーン表示用に画像の縮小
        #dst_disp_s = resizeWithAspectRatio(distance, width=w)]
        w = 480
        frameS = resizeWithAspectRatio(r_img_l, width=w)
        mask_005S = resizeWithAspectRatio(reverseBW(mask_005), width=w)
        mask_02S = resizeWithAspectRatio(reverseBW(mask_02), width=w)
        mask_01S = resizeWithAspectRatio(reverseBW(mask_01), width=w)
        
        frame_h = frameS.shape[0]
        frame_w = frameS.shape[1]

        #disp = dst_disp_s
        
        disp = np.tile(np.zeros_like(frameS),(2,2,1))

        disp[:frame_h, :frame_w, :] = frameS # (1, 1)
        disp[:frame_h, frame_w:, :] = grayToColor(mask_005S*255) # (1, 2)
        disp[frame_h:, :frame_w, :] = grayToColor(mask_01S*255) # (2, 1)
        disp[frame_h:, frame_w:, :] = grayToColor(mask_02S*255) # (2, 1)

        #cv2.imshow("Floor Removal", disp)

        # if save_flag:
        #     writer.write(disp) # 画像を1フレーム分として書き込み

        if cv2.waitKey(DELAY) & 0xFF == ord('p'):
            break

    # if save_flag:
    #     writer.release() # ファイルを閉じる

def main(video = VIDEO):
    floorEstimationVid(video, save_flag=VID_SAVE_MODE)

# --------------
# main 実行
if __name__ == "__main__":
    main()
    print("=============================")