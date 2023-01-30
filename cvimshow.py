#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 画像表示に関する関数

import cv2
import matplotlib.pyplot as plt
import numpy as np

import datetime

# ----
# 定数
WINNAME_DEFAULT = "Image"
WIDTH_DEFAULT = 640

# ----
# 画像表示全般
def imshowS(img, winname=WINNAME_DEFAULT, width=WIDTH_DEFAULT, wait=True, saveFlag=False, now=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')):
    """アスペクト比を変えずリサイズして画像を表示"""
    imgS = resizeWithAspectRatio(img, width=width)
    cv2.imshow(winname, imgS)
    if saveFlag:
        cv2.imwrite('./dst/frame/'+now+".jpg", imgS)
        print("Image Saved!")
    if wait:
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