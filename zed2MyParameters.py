import numpy as np

class Params:
    path: str
    ff: float
    tu: float
    tv: float
    bb: float

    def __init__(self,
            path:str, 
            ff: float, 
            tu: float, tv, bb):
        self.path = path
        self.ff = ff
        self.tu = tu
        self.tv = tv
        self.bb = bb

    def maps(self):
        maps = np.load(self.path)
        return maps['arr_0'], maps['arr_1'], maps['arr_2'], maps['arr_3']

    # def two2three(self, posel, poser):
    #     posexyz = posel.copy()
    #     posexyz[:,:,0] = self.bb * (posel[:,:,0] - self.tu) / (posel[:,:,0] - poser[:,:,0]) #x
    #     posexyz[:,:,1] = self.bb * (posel[:,:,1] - self.tv) / (posel[:,:,0] - poser[:,:,0]) #y
    #     posexyz[:,:,2] = self.bb * self.ff / (posel[:,:,0] - poser[:,:,0]) #z
    #     return posexyz
    
# ----
# インスタンス
MY_GOOD_PARAMS = Params(
    path = "./src/_soma-stereo-maps-iikannji.npz",
    ff = 1.07493419e+03, 
    tu = 9.63787597e+02, 
    tv = 5.45712005e+02, 
    bb = 0.12
)

MY_PARAMS = Params(
    path = "./src/_soma-stereo-maps.npz",
    ff = 1.07493419e+03, 
    tu = 9.63787597e+02, 
    tv = 5.45712005e+02, 
    bb = 0.12
)

HANA_PARAMS = Params(
    path = "./src/_maps_stereo_pair.npz",
    ff = 1.06731341e+03,
    tu = 9.54766167e+02,
    tv = 5.85289013e+02,
    bb = 0.12                               # baseline length (m)
)

#map_path = "./src/_soma-stereo-maps.npz"
# map_path = "./src/_soma-stereo-maps-iikannji.npz"

# ff = 1.07493419e+03     # 焦点距離 (pixel？)
# tu = 9.63787597e+02     #-1.00723653e+03     # 主点補正 縦
# tv = 5.45712005e+02     #-5.55301683e+02     # 横
# bb = 0.12               # 基線長(m)

""" 20220728
[[1.07493419e+03 0.00000000e+00 9.63787597e+02]
 [0.00000000e+00 1.07435144e+03 5.45712005e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
 [[1.07436046e+03 0.00000000e+00 1.00723653e+03]
 [0.00000000e+00 1.07444350e+03 5.55301683e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
"""

# def getMat():
#     mat_l = np.array([[1.07493419e+03, 0.00000000e+00, 9.63787597e+02],
#     [0.00000000e+00, 1.07435144e+03, 5.45712005e+02],
#     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
#     mat_r = np.array([[1.07436046e+03, 0.00000000e+00, 1.00723653e+03],
#     [0.00000000e+00, 1.07444350e+03, 5.55301683e+02],
#     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
#     return mat_l, mat_r

# def two2three(posel, poser, ff=ff, bb=bb, tu=tu, tv=tv):
#     posexyz = posel.copy()
#     posexyz[:,:,0] = bb * (posel[:,:,0] - tu) / (posel[:,:,0] - poser[:,:,0])
#     posexyz[:,:,1] = bb * (posel[:,:,1] - tv) / (posel[:,:,0] - poser[:,:,0])
#     posexyz[:,:,2] = bb * ff / (posel[:,:,0] - poser[:,:,0])
#     return posexyz

# def maps():
#     maps = np.load(map_path)
#     return maps['arr_0'], maps['arr_1'], maps['arr_2'], maps['arr_3']

#print(maps)