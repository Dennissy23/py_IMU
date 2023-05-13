import numpy as np
import math
from scipy.spatial.transform import Rotation
import warnings

class Rotation_my:
    @staticmethod
    def matrix2quaternion(matrix):
        matrix = np.transpose(matrix)
        
        quat = Rotation.from_matrix(matrix).as_quat()
        return quat
    
    @staticmethod
    def quaternion2matrix(quat):
        matrix = Rotation.from_quat(quat).as_matrix()
        
        
        return matrix
    
    @staticmethod
    def matrix2euler(dcm: np.ndarray) -> np.ndarray:
        euler = np.zeros(3)

        euler[1] = math.atan(-dcm[2, 0]/math.sqrt(dcm[2, 1] ** 2 + dcm[2, 2] ** 2))

        if dcm[2, 0] <= -0.999:
            euler[0] = 0
            euler[2] = math.atan2((dcm[1, 2] - dcm[0, 1]), (dcm[0, 2] + dcm[1, 1]))
            warnings.warn("[WARNING] Rotation::matrix2euler: Singular Euler Angle! Set the roll angle to 0!")
        elif dcm[2, 0] >= 0.999:
            euler[0] = 0
            euler[2] = math.pi + math.atan2((dcm[1, 2] + dcm[0, 1]), (dcm[0, 2] - dcm[1, 1]))
            warnings.warn("[WARNING] Rotation::matrix2euler: Singular Euler Angle! Set the roll angle to 0!")
        else:
            euler[0] = math.atan2(dcm[2, 1], dcm[2, 2])
            euler[2] = math.atan2(dcm[1, 0], dcm[0, 0])
        # heading 0~2PI
        if euler[2] < 0:
            euler[2] = math.pi * 2 + euler[2]
        
        return euler
    
    @staticmethod   
    def quaternion2euler(quat):
        matrix = Rotation.from_quat(quat).as_matrix()
        return Rotation_my.matrix2euler(matrix)
    
    @staticmethod 
    def rotvec2quaternion(rotvec):
        return Rotation.from_rotvec(rotvec).as_quat()
    
    @staticmethod 
    def quaternion2vector(quaternion):
        rotation = Rotation.from_quat(quaternion)
        vector = rotation.as_rotvec()
        return vector
    
    @staticmethod   
    def euler2matrix(euler):
        rotation = Rotation.from_euler('ZYX', euler, degrees=False)
        ma = rotation.as_matrix()
        ma = np.flip(ma, axis=(0,1))
        return ma.T
    
    @staticmethod   
    def euler2quaternion(euler):
        rotation = Rotation.from_euler('ZYX', euler, degrees=False)
        quaternion = rotation.as_quat()
        return quaternion
    
    @staticmethod   
    def skewSymmetric(vector):
        mat = np.array([[0, -vector[2], vector[1]],
                        [vector[2], 0, -vector[0]],
                        [-vector[1], vector[0], 0]] )
        
        return -mat.T
    
    @staticmethod 
    def quaternionleft(q):
        ans = np.zeros((4, 4))
        ans[0, 0]     = q[3]
        ans[0, 1:4]   = -q[0:3].transpose()
        ans[1:4, 0]   = q[0:3]
        ans[1:4, 1:4] = q[3] * np.eye(3) + Rotation_my.skewSymmetric(q[0:3])
        return ans
    
    @staticmethod 
    def quaternionright(p):
        ans = np.zeros((4, 4))
        ans[0, 0]     = p[3]
        ans[0, 1:4]   = -p[0:3].transpose()
        ans[1:4, 0]   = p[0:3]
        ans[1:4, 1:4] = p[3] * np.eye(3) - Rotation_my.skewSymmetric(p[0:3])
        return ans