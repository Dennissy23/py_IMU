import numpy as np

class GNSS:
    def __init__(self):
        self.time = 0.0
        self.blh = np.zeros(3)
        self.std = np.zeros(3)
        self.isvalid = False

class IMU:
    def __init__(self):
        self.time = 0.0
        self.dt = 0.0
        self.dtheta = np.zeros(3)
        self.dvel = np.zeros(3)
        self.odovel = 0.0

class Pose:
    def __init__(self):
        self.R = np.eye(3)
        self.t = np.zeros(3)