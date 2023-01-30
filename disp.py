

from ctypes import pointer
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
import datetime
import os

# 自作モジュール
#from cvimshow import imshowS, resizeWithAspectRatio
from util import *

# ----
# 定数
WINNAME_DEFAULT = "Image"
WIDTH_DEFAULT = 640

# ----
# 画像表示全般
def imshowS(img, winName=WINNAME_DEFAULT, width=WIDTH_DEFAULT, wait=True, saveFlag=False, now=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')):
    """アスペクト比を変えずリサイズして画像を表示"""
    imgS = resizeWithAspectRatio(img, width=width)
    cv2.imshow(winName, imgS)
    if saveFlag:
        cv2.imwrite('./dst/frame/'+now+".jpg", imgS)
        print("Image Saved!")
    if wait:
        cv2.waitKey(0)
    else:
        cv2.waitKey(10)

def imshowPlt(img, winName=WINNAME_DEFAULT, waitFlag=True):
    """アスペクト比を変えずリサイズして画像を表示"""
    # print(img.shape, len(img.shape))
    if len(img.shape) == 3:
        imgDisp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        imgDisp = img

    fig_img_plt = plt.figure(num=winName, figsize=(10,6))
    ax_img_plt = fig_img_plt.add_subplot(111)
    ax_img_plt.imshow(imgDisp)

    if waitFlag:
        cv2.waitKey(0)
    else:
        cv2.waitKey(10)

def dispHeight(height, winName="Height", waitFlag=True):
    """高さを表示"""
    # print(img.shape, len(img.shape))
    fig_img_plt = plt.figure(num=winName, figsize=(10,6))
    ax_img_plt = fig_img_plt.add_subplot(111)
    ax_img_plt.imshow(height)
    plt.show()

    if waitFlag:
        cv2.waitKey(0)
    else:
        cv2.waitKey(10)

def resizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """アスペクト比を変えずに画像をリサイズ"""
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

# ----
# マスク関連
def imMask(img, mask):
    mask_dst = mask.copy()
    if len(img.shape) > 2:
        mask_dst = grayToColor(mask_dst)
    dst = img*mask_dst.astype(np.uint8)
    return dst

def imshowMask(img, mask, winname="Mask", width=WIDTH_DEFAULT, wait=True, saveFlag=False, now=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')):
    if np.max(mask) > 1.0:
        mask[np.where(mask > 0)] = mask[np.where(mask > 0)]/np.max(mask)
    disp = imMask(img, mask)
    if saveFlag:
        cv2.imwrite('./dst/mask/'+now+".jpg", disp)
        print("Image Saved!")
    imshowS(disp, winname, width, wait)

def reverseBW(mask):
    return -(mask-1.0)

# ----
# グレースケール画像関連
def grayToColor(grayImg):
    """グレー画像をカラーに変換する"""
    colorImg = np.zeros((grayImg.shape[0], grayImg.shape[1], 3))
    for i in range(3):
        colorImg[:, :, i] = grayImg
    return colorImg

# ----
# ステレオ画像関連
def imColorShift(img_l, img_r):
    """色ずれ画像"""
    # カラーの場合はグレーへ
    if len(img_l.shape) > 2:
        # Grayscale Images
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # 色ずれ画像表示
    dst = np.zeros((*img_l.shape, 3))
    dst[:, :, 0] = img_l/256
    dst[:, :, 1] = img_l/256
    dst[:, :, 2] = img_r/256

    return dst

def imshowColorShift(img_l, img_r, winname="Color Shift", width=WIDTH_DEFAULT, wait=True):
    """色ずれ画像表示"""
    dst = imColorShift(img_l, img_r)
    imshowS(dst, winname, width=width, wait=wait)
    return

# def disp3d(_3dData, ax = None, dispMode = "Depth", winName="3D Disp"):
#     """3次元描画
#     DispMode = Depth , Height"""
#     return

# 3次元scatter
def dispScatter(depth) -> None:
    fig_test = plt.figure(num="Disp Scatter", figsize=(6, 4))
    ax_test = fig_test.add_subplot(111, projection='3d')
    ax_test.scatter(depth[:, :, 0].reshape(-1, 1), 
            depth[:, :, 1].reshape(-1, 1), 
            depth[:, :, 2].reshape(-1, 1),
            s=0.01, alpha=0.1
            )
    ax_test.set_xlabel("X (m)"); ax_test.set_ylabel("Y (m)"); ax_test.set_zlabel("Z (m)")
    ax_test.view_init(elev=-140, azim=-125)
    plt.show()