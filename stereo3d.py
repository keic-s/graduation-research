from ast import Pass
import cv2
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
#from scipy.ndimage import morphology as mor
import datetime

#from src.main import SAVE_FLAG

import zed2MyParameters
from zed2MyParameters import Params
#import zed2_hanaizumi_parameters
from calibration import devideStereo
import calibration
from cvimshow import imshowS, resizeWithAspectRatio

# ----
# 表示速度
DELAY = 1

# 保存
SAVE_FLAG = False

# 動画から読み込むモード
DEBUG_MODE = True

FILENAME = "./res/hana/交差/WIN_20201211_15_20_42_Pro.mp4"; INIT_FRAME = 980; PARAMS = zed2MyParameters.HANA_PARAMS # 4人が交差
#filename = "./res/hana/上下/WIN_20201211_15_14_20_Pro.mp4" # 4人並列
#filename = "./res/videos/20220706/WIN_20220706_17_43_53_Pro.mp4"; INIT_FRAME = 1200 # 管理棟横駐輪場 チェスボード
#filename = "./res/videos/20220706/WIN_20220706_17_52_02_Pro.mp4" # 管理棟横駐輪場 ちょっと近い
#filename = "./res/videos/20220721/dist2m (1).mp4"; INIT_FRAME = 20 # 2m離したチェスボード
#INIT_FRAME = 200

# カメラのとき
CAM_ID = 1

# StereoBMの探索領域
MAX_SEARCH = 120

# ガウシアンフィルタのパラメータ
GAUSSIAN_KSIZE = 1
GAUSSIAN_SIGMA = 1.

# 図の表示
FIG_DISPLAY = False
if FIG_DISPLAY:
    FIG = plt.figure(figsize=(9,6))
    AX = FIG.add_subplot(111, projection='3d')
    AX.set_title("Stereo 3D 3Ddst")
    AX.set_xlabel("X (m)")
    AX.set_ylabel("Y (m)")
    AX.set_zlabel("Z (m)")
    AX.view_init(elev=-150, azim=-130)

def stereo3d(imgl, imgr, params):
    """2枚の画像から深度情報を取得"""
    rimg1 = imgl.copy()
    rimg2 = imgr.copy()
    ksize, sigmaX = (GAUSSIAN_KSIZE, GAUSSIAN_KSIZE), GAUSSIAN_SIGMA
    rimg1 = cv2.GaussianBlur(rimg1, ksize, sigmaX)
    rimg2 = cv2.GaussianBlur(rimg2, ksize, sigmaX)
    if False:
        imshowS(rimg1, winname="Left Image Gaussian Blured", wait=False)
    disp = sgbm(rimg1, rimg2)
    depth = disp2depth(disp, params)
    return depth

def sgbm(rimg1, rimg2, max_search=MAX_SEARCH):
    """run SGM stereo matching with weighted least squares filtering
    #print('Running SGBM stereo matcher...')
    """
    if len(rimg1.shape) > 2:
        rimg1 = cv2.cvtColor(rimg1, cv2.COLOR_BGR2GRAY)
        rimg2 = cv2.cvtColor(rimg2, cv2.COLOR_BGR2GRAY)
    maxd = max_search #220 #DMAX_SEARCH
    #print('MAXD = ', maxd)
    window_size = 5
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-maxd,
        numDisparities=maxd * 2,
        blockSize=9,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    lmbda = 8000
    sigma = 1.5
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(rimg1, rimg2)
    dispr = right_matcher.compute(rimg2, rimg1)
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    disparity = wls_filter.filter(displ, rimg1, None, dispr) / 16.0 # StereoBMによって16倍された値が出る
    
    # print(disparity.dtype)

    # 画像を図示
    if False:
        plt.figure(figsize=(9,6))
        plt.imshow(rimg1, cmap="gray")
        plt.title('Left')
        # plt.show()
        plt.figure(figsize=(9,6))
        plt.imshow(rimg2, cmap="gray")
        plt.title('Right')
        # plt.show()

    # 視差を図示
    if False:
        plt.figure(figsize=(9,4))
        plt.imshow(disparity)
        plt.colorbar()
        plt.title('disparity')
        #plt.show()

    return disparity

def disp2depth(disp, param: Params, trim=False):
    """視差から深度を求める
    
    disp: 視差
    trim: キリトリフラグ
    """
    h, w = disp.shape

    if trim:
        disp = disp[:, MAX_SEARCH:w-MAX_SEARCH]
        w = w - 2*MAX_SEARCH

    #print(disp.min())
    disp[np.where(disp==disp.min())] = np.nan # 最小値はマッチングミス？(要考察)
    #print(disp.min())
    bb = PARAMS.bb
    tu = PARAMS.tu
    tv = PARAMS.tv
    ff = PARAMS.ff

    x,y = np.meshgrid(np.arange(w),np.arange(h))
    #depth = np.zeros((h,w,3)).asarray(float32)
    depth = np.zeros((h,w,3))
    xx = bb * (x - tu) / disp
    yy = bb * (y - tv) / disp
    zz = bb * ff / disp

    distrange = [1, 15]
    fill = np.nan
    xx[np.where(zz < distrange[0])] = fill
    xx[np.where(zz > distrange[1])] = fill
    yy[np.where(zz < distrange[0])] = fill
    yy[np.where(zz > distrange[1])] = fill
    zz[np.where(zz < distrange[0])] = fill
    zz[np.where(zz > distrange[1])] = fill

    depth[:,:,0] = xx
    depth[:,:,1] = yy
    depth[:,:,2] = zz

    if False: # データのヒストグラム
        dist_hist_data = dist.flatten()
        plt.figure(figsize=(9,4))
        plt.hist(dist_hist_data, range = (distrange[0], distrange[1]))

    if False:
        #plt.imshow(depth[:,:,2], cmap='gray') # index 2 は距離
        plt.figure(figsize=(9,4))
        plt.imshow(depth[:,:,2]) # index 2 は距離
        plt.colorbar()
        plt.title('Distance Map')
        plt.show()

    if False: # 3次元グラフ
        #print(depth.shape)
        h, w = depth.shape[:2]

        xx = depth[:,MAX_SEARCH:w-MAX_SEARCH,0]#.reshape(-1); #xx = np.filp(xx, 0)
        yy = depth[:,MAX_SEARCH:w-MAX_SEARCH,1]#.reshape(-1); #yy = np.flipud(yy)
        zz = depth[:,MAX_SEARCH:w-MAX_SEARCH,2]; #zz = np.flip(zz, 2)
        #print("xx: ", xx, "\nyy: ", yy)

        # print("x: ",np.nanmin(xx), np.nanmax(xx), 
        #     "\ny: ",np.nanmin(yy), np.nanmax(yy),
        #     "\nz: ",np.nanmin(zz), np.nanmax(zz))
        x_area = [np.nanmin(xx), np.nanmax(xx)]
        y_area = [np.nanmin(yy), np.nanmax(yy)]
        z_area = [np.nanmin(zz), np.nanmax(zz)]

        AX.plot_surface(xx,yy,zz,cmap="winter")

        AX.set_xlim3d(x_area)
        AX.set_ylim3d(y_area)
        AX.set_zlim3d(z_area)

        #AX.scatter(0, 0, 0, s=10, c="red")

        if False:
        # 距離z=2.015の位置の平面
            floor_p = np.array([0., 0., 2.015])
            floor_v = np.array([0., 0., 1.])
            dstPlaneX,dstPlaneY = np.meshgrid(
                np.linspace(*x_area, 50),
                np.linspace(*y_area, 50)
                )
            dstPlaneZ = floor_p[2] - (floor_v[0] * (dstPlaneX - floor_p[0]) + floor_v[1] * (dstPlaneY - floor_p[1])) / floor_v[2]
            ax.plot_surface(dstPlaneX, dstPlaneY, dstPlaneZ, color="red", alpha=0.5)

        plt.pause(0.001)
        #plt.clf()

    return depth

def stereoVid3d(filename, init_frame, save_flag=False):
    """ステレオ映像から

    filename: 動画のファイル
    init_frame: 初期フレーム
    save_flag: 保存
    """

    vid = cv2.VideoCapture(filename)

    if not vid.isOpened(): #動画を読み込めていないとき終了
        # Video Not Loaded
        print("ビデオのロードに失敗しました")
        return

    # パラメータ
    w = 1080 # frameS表示用サイズ

    if DEBUG_MODE:
        vid.set(cv2.CAP_PROP_POS_FRAMES, init_frame)

    if save_flag:
        now = datetime.datetime.now()
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
        outFilename = "./dst/stereo3d/STEREO3D_VID_"+ now.strftime('%Y%m%d_%H%M%S') +".mp4"
        writer = cv2.VideoWriter(
            outFilename,
            fmt,
            int(vid.get(cv2.CAP_PROP_FPS)),
            (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            ) # ライター作成

    while(vid.isOpened()):
        ret, frame = vid.read()
        if not ret:
            break
        
        imgl, imgr = devideStereo(frame)
        rimgl, rimgr = calibration.stereoRemap(imgl, imgr, maps=PARAMS.maps)

        dstDisp = disp2depth(sgbm(rimgl, rimgr))

        # ----
        # スクリーン表示用に画像の縮小
        dstDispS = resizeWithAspectRatio(dstDisp, width=w)
        #frameS = resizeWithAspectRatio(frame, width=w)
        
        #frame_h = frameS.shape[0]
        #frame_w = frameS.shape[1]

        disp = dstDispS
        
        #disp = np.tile(np.zeros_like(frameS),(2,1,1))

        # disp[:frame_h, :, :] = grayToColor(lapS)
        # disp[frame_h:, :, :] = grayToColor(maskOpenS)

        #cv2.imshow("stereo3d", disp)

        if save_flag:
            writer.write(disp) # 画像を1フレーム分として書き込み

        if cv2.waitKey(DELAY) & 0xFF == ord('q'):
            break

    if save_flag:
        writer.release() # ファイルを閉じる

def main():
    filename = FILENAME

    if not DEBUG_MODE:
    # Webカメラを読み込み
        filename = CAM_ID

    stereoVid3d(filename, INIT_FRAME, SAVE_FLAG)

# --------------
# main 実行
if __name__ == "__main__":
    print("=============================")
    main()
    print("=============================")