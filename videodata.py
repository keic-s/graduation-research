# ビデオデータ管理

import numpy as np
import cv2
from matplotlib import pyplot as plt
import datetime

# --
# from disp import *
import zed2_hanaizumi_parameters, zed2MyParameters
from zed2MyParameters import Params
from cvimshow import imshowS

class Video:
    """動画クラス"""
    loadFoldername: str = None          # フォルダーの名前
    filename: str = None                # ファイルの名前
    fileExtension: str = None           # ファイルの拡張子
    loadFilename: str = None            # 読み込むファイルの名前(loadFoldername + filename + fileExtension)
    saveFilename: str = None
    params: Params = None
    width = 0
    height = 0
    xArea = [0, 0]
    yArea = [0, 0]
    zArea = [0, 0]
    vid: cv2.VideoCapture = None

    def __init__(self, loadFoldername: str, filename: str, fileExtension: str, params: Params, initFrame, xArea=None, yArea=None, zArea=None): # saveFilename="./dst/"
        self.filename = filename
        self.loadFilename = loadFoldername+filename+fileExtension
        self.vid = cv2.VideoCapture(self.loadFilename)
        # Video Not Loaded
        #print("ビデオのロードに失敗しました")
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # now = datetime.datetime.now()
        # fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
        # self.saveFilename = saveFilename + now.strftime('%Y%m%d_%H%M%S') + ".mp4"
        # self.writer = cv2.VideoWriter(
        #     self.filename,
        #     fmt,
        #     int(self.vid.get(cv2.CAP_PROP_FPS)),
        #     (self.width,self.height)
        #     ) # ライター作成
        self.params = params
        self.initFrame = initFrame
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, initFrame)
        self.xArea = xArea
        self.yArea = yArea
        self.zArea = zArea

    def setFrame(self, n):
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, n)
    
    def read(self):
        return self.vid.read()

    def setting(self):
        """設定"""
        return

    def play(self, frame:int =None, wait=False):
        if frame is None:
            while True:
                ret, frame = self.vid.read()
                imshowS(frame, winname="Play", wait=wait)
            return
        for n in range(frame):
            ret, frame = self.vid.read()
            imshowS(frame, winname="Play", wait=wait)

    # def save(self, frame):
    #     """フレーム保存"""
    #     self.writer.write(frame)

    def close(self):
        """動画閉じる"""
        self.writer.release()

# ----
# 4人が交差
HANA_CROSS = Video(
    loadFoldername="./res/hana/交差/",
    filename="WIN_20201211_15_20_42_Pro",
    fileExtension=".mp4",
    params=zed2MyParameters.HANA_PARAMS, 
    initFrame=980
    )

HANA_CROSSV2 = Video(
    loadFoldername="./res/hana/交差/",
    filename="WIN_20201211_15_20_42_Pro",
    fileExtension=".mp4",
    params=zed2MyParameters.HANA_PARAMS, 
    initFrame=990
    )

HANA_CROSS2 = Video(
    loadFoldername="./res/hana/交差/",
    filename="WIN_20201211_15_20_42_Pro",
    fileExtension=".mp4",
    params=zed2MyParameters.HANA_PARAMS, 
    initFrame=30*(60+54)
    )

HANA_CROSS3 = Video(
    loadFoldername="./res/hana/交差/",
    filename="WIN_20201211_15_23_54_Pro",
    fileExtension=".mp4",
    params=zed2MyParameters.HANA_PARAMS, 
    initFrame=30*(6)+50
    )

    

HANA_CROSS_OCCLUDED = Video(        # オクルージョン発生の瞬間
    loadFoldername="./res/hana/交差/",
    filename="WIN_20201211_15_20_42_Pro",
    fileExtension=".mp4",
    params=zed2MyParameters.HANA_PARAMS, 
    initFrame=1005
    )

HANA_CROSS_OCCLUDED_2 = Video(        # オクルージョン発生の瞬間
    loadFoldername="./res/hana/交差/",
    filename="WIN_20201211_15_20_42_Pro",
    fileExtension=".mp4",
    params=zed2MyParameters.HANA_PARAMS, 
    initFrame=1022
    )

HANA_CROSS_START = Video(  
    loadFoldername="./res/hana/交差/",
    filename="WIN_20201211_15_20_42_Pro",
    fileExtension=".mp4",
    params=zed2MyParameters.HANA_PARAMS, 
    initFrame=30*(30)
    )

HANA_PARALLEL = Video(
    loadFoldername="./res/hana/上下/",
    filename="WIN_20201211_15_14_20_Pro",
    fileExtension=".mp4",
    params=zed2MyParameters.HANA_PARAMS, 
    initFrame=30*10, 
    # xArea=[-5, 5], 
    # yArea=[-4, 0], 
    # zArea=[3, 7]
    )

HANA_TWO = Video(
    loadFoldername="./res/hana/上下/",
    filename="WIN_20201211_15_11_06_Pro",
    fileExtension=".mp4",
    params=zed2MyParameters.HANA_PARAMS, 
    initFrame=30*11, 
    # xArea=[-5, 5], 
    # yArea=[-4, 0], 
    # zArea=[3, 7]
    )

HANA_PARALLEL_BEHIND = Video(
    loadFoldername="./res/hana/上下/",
    filename="WIN_20201211_15_11_06_Pro",
    fileExtension=".mp4",
    params=zed2MyParameters.HANA_PARAMS, 
    initFrame=30*(60+26), 
    # xArea=[-5, 5], 
    # yArea=[-4, 0], 
    # zArea=[3, 7]
    )

HANA_PARALLEL_THREE = Video(
    loadFoldername="./res/hana/上下/",
    filename="WIN_20201211_15_11_06_Pro",
    fileExtension=".mp4",
    params=zed2MyParameters.HANA_PARAMS, 
    initFrame=30*(60*2+16), 
    # xArea=[-5, 5], 
    # yArea=[-4, 0], 
    # zArea=[3, 7]
    )

HANA_SOLO = Video(
    loadFoldername="./res/hana/上下/",
    filename="WIN_20201211_15_14_20_Pro",
    fileExtension=".mp4",
    params=zed2MyParameters.HANA_PARAMS, 
    initFrame=30*14, 
    # xArea=[-5, 5], 
    # yArea=[-4, 0], 
    # zArea=[3, 7]
    )



# ----
# 学祭データ
MY_SCRFES_1 = Video(
    loadFoldername="./res/videos/20221103/",
    filename="WIN_20221103_13_49_37_Pro",
    fileExtension=".mp4",
    params=zed2MyParameters.MY_GOOD_PARAMS, 
    initFrame=120,
    # xArea=[-5, 5],
    # yArea=[-4, 2],
    # zArea=[3, 7]
)

MY_SCRFES_2 = Video(
    loadFoldername="./res/videos/20221103/",
    filename="WIN_20221103_14_10_19_Pro",
    fileExtension=".mp4",
    params=zed2MyParameters.MY_GOOD_PARAMS, 
    initFrame=150*30,
    # xArea=[-5, 5],
    # yArea=[-4, 2],
    # zArea=[3, 7]
)

MY_SCRFES_3 = Video(
    loadFoldername="./res/videos/20221103/",
    filename="WIN_20221103_13_53_25_Pro",
    fileExtension=".mp4",
    params=zed2MyParameters.MY_GOOD_PARAMS, 
    initFrame=1,
    # xArea=[-5, 5],
    # yArea=[-4, 2],
    # zArea=[3, 7]
)

MY_SCRFES_4 = Video(
    loadFoldername="./res/videos/20221103/",
    filename="WIN_20221103_13_42_17_Pro",
    fileExtension=".mp4",
    params=zed2MyParameters.MY_GOOD_PARAMS, 
    initFrame=73*30,
    # xArea=[-5, 5],
    # yArea=[-4, 2],
    # zArea=[3, 7]
)

MY_SCRFES_5 = Video(
    loadFoldername="./res/videos/20221103/",
    filename="WIN_20221103_13_42_17_Pro",
    fileExtension=".mp4",
    params=zed2MyParameters.MY_GOOD_PARAMS, 
    initFrame=73*30,
    # xArea=[-5, 5],
    # yArea=[-4, 2],
    # zArea=[3, 7]
)

MY_SCRFES_REFLECT = Video(  # 反射してる
    loadFoldername="./res/videos/20221103/",
    filename="WIN_20221103_14_15_47_Pro",
    fileExtension=".mp4",
    params=zed2MyParameters.MY_GOOD_PARAMS, 
    initFrame=1,
    # xArea=[-5, 5],
    # yArea=[-4, 2],
    # zArea=[3, 7]
)

MY_DEC_THREE = Video(
    loadFoldername="./res/videos/20221222/",
    filename="WIN_20221222_12_31_48_Pro",
    fileExtension=".mp4",
    params=zed2MyParameters.MY_GOOD_PARAMS, 
    initFrame=30*(60*3), 
    # xArea=[-5, 5], 
    # yArea=[-4, 0], 
    # zArea=[3, 7]
    )

MY_DEC_TEST = Video(
    loadFoldername="./res/videos/20230112/",
    filename="DetectTest1",
    fileExtension=".mp4",
    params=zed2MyParameters.MY_GOOD_PARAMS, 
    initFrame=0, 
    # xArea=[-5, 5], 
    # yArea=[-4, 0], 
    # zArea=[3, 7]
    )

# video_dict = {
#     "hanacross": HANA_CROSS,
#     "hanaparalell": HANA_PARALLEL,
#     "mychessboard": MY_CHESSBOARD,
#     "myexperiment": MY_EXPERIMENT
# }

# def getVideo(filename):
#     return video_dict[filename]

if False: # テスト用
    vid = MY_MOVE
    width = 980
    delay = 1

    while(True):
        ret, frame = vid.read()

        if not ret:
            break
    
        # ----
        # q
        dstDispS = resizeWithAspectRatio(frame, width=width)
        disp = dstDispS
        cv2.imshow("stereo3d", disp)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

def main(video = HANA_CROSS_OCCLUDED):
    video.play(frame=30, wait=True)

# --------------
# main 実行
if __name__ == "__main__":
    main()