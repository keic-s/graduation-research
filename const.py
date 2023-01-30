# 定数・パラメータをまとめたファイル

import os
import datetime
import videodata
import cv2
from util import *

# -----
# 定数

DELAY = 1
CAM_ID = 1

NOW = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

SAVE_FLAG = True # True: Save to folder "dst"
DEBUG_MODE = True # True: Load from video

# クラスタリング パラメータ
if False:    # MY PARAM
    EPS = 0.1
    MIN_SAMPLES = 800
    CONTOURS_THSH = 500
else:   # HANA
    EPS = 0.08
    MIN_SAMPLES = 800
    CONTOURS_THSH = 1000

    # EPS = 0.05
    # MIN_SAMPLES = 600
    # CONTOURS_THSH = 1000

COLOR_LIST = ["r", "g", "b", "c", "m", "y", "k", "orange", "purple", "yellowgreen", "darkblue", "skyblue"]

DIST_LIST = [0.01, 0.02, 0.03, 0.04, 0.05]
#DIST_LIST = [0.01, 0.02, 0.03, 0.04]

LOOP_MODE_WHILE_TRUE = False

SW = StopWatch()

HUMAN_ID = 0

HUMAN_R = 0.3 # [m]
HUMAN_R_COEF = 0.7 
CORE_RADIUS_COEF = 0.7 # 半径のこの倍のエリアが重ならないようにする
RADIUS_ADJUSTMENT = 1.0

HUMAN_MINPOINT_COEF = 0.2 # 点数の半分になったとき削除
HUMAN_NUMPOINT = 0  # 定数だが後に更新

# FIG_SIZE = (4,3)
FIG_SIZE = (10,6)

BG_MODEL = cv2.createBackgroundSubtractorMOG2(history=600, varThreshold=20, detectShadows=True)