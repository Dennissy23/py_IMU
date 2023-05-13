import numpy as np

class Attitude:
    def __init__(self):
        self.qbn = np.zeros(4)
        self.cbn = np.zeros((3,3))
        self.euler = np.zeros(3)

class PVA:
    def __init__(self):
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.att = Attitude()

class ImuError:
    def __init__(self):
        self.gyrbias = np.zeros(3)
        self.accbias = np.zeros(3)
        self.gyrscale = np.zeros(3)
        self.accscale=  np.zeros(3)

class NavState:
    def __init__(self):
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.euler = np.zeros(3)
        self.imuerror = ImuError()

class ImuNoise:
    def __init__(self):
        self.gyr_arw = np.zeros(3)
        self.acc_vrw = np.zeros(3)
        self.gyrbias_std = np.zeros(3)
        self.accbias_std = np.zeros(3)
        self.gyrscale_std = np.zeros(3)
        self.accscale_std = np.zeros(3)
        self.corr_time = 0.0

class GINSOptions:
    def __init__(self):
        ##  初始状态和状态标准差
        self.initstate = NavState()
        self.initstate_std = NavState()
        ##  IMU噪声参数
        self.imunoise = ImuNoise()
        ##  安装参数
        self.antlever = np.zeros(3)