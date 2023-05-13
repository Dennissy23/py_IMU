import numpy as np
import math
from scipy.spatial.transform import Rotation
import types_my as ty

class Earth:
    WGS84_WIE = 7.2921151467E-5   # 地球自转角速度
    WGS84_F = 0.0033528106647474805   # 扁率
    WGS84_RA = 6378137.0000000000   # 长半轴a
    WGS84_RB = 6356752.3142451793   # 短半轴b
    WGS84_GM0 = 398600441800000.00   # 地球引力常数
    WGS84_E1 = 0.0066943799901413156   # 第一偏心率平方
    WGS84_E2 = 0.0067394967422764341   # 第二偏心率平方

    # 正常重力计算
    @staticmethod
    def gravity(blh:np.ndarray) -> float:
        sin2 = math.sin(blh[0])**2
        return 9.7803267715 * (1 + 0.0052790414 * sin2 + 0.0000232718 * sin2**2) + \
               blh[2] * (0.0000000043977311 * sin2 - 0.0000030876910891) + \
               0.0000000000007211 * blh[2]**2
    
    # 计算子午圈半径和卯酉圈半径
    @staticmethod
    def meridianPrimeVerticalRadius(lat:float) -> np.ndarray:
        tmp = math.sin(lat)**2
        tmp = 1 - Earth.WGS84_E1 * tmp
        sqrttmp = math.sqrt(tmp)
        return np.array([Earth.WGS84_RA * (1 - Earth.WGS84_E1) / (sqrttmp * tmp), Earth.WGS84_RA / sqrttmp])
    
    # 计算卯酉圈半径
    @staticmethod
    def RN(lat:float) -> float:
        sinlat = math.sin(lat)
        return Earth.WGS84_RA / math.sqrt(1.0 - Earth.WGS84_E1 * sinlat * sinlat)
    
    # n系(导航坐标系)到e系(地心地固坐标系)转换矩阵
    @staticmethod
    def cne(blh:np.ndarray) -> np.ndarray:
        sinlat = math.sin(blh[0])
        sinlon = math.sin(blh[1])
        coslat = math.cos(blh[0])
        coslon = math.cos(blh[1])

        dcm = np.zeros((3,3))
        dcm[0, 0] = -sinlat * coslon
        dcm[0, 1] = -sinlon
        dcm[0, 2] = -coslat * coslon

        dcm[1, 0] = -sinlat * sinlon
        dcm[1, 1] = coslon
        dcm[1, 2] = -coslat * sinlon

        dcm[2, 0] = coslat
        dcm[2, 1] = 0
        dcm[2, 2] = -sinlat
        return dcm
    
    # n系(导航坐标系)到e系(地心地固坐标系)转换四元数
    @staticmethod
    def qne(blh:np.ndarray) -> np.ndarray:
        coslon = math.cos(blh[1] * 0.5)
        sinlon = math.sin(blh[1] * 0.5)
        coslat = math.cos(-math.pi * 0.25 - blh[0] * 0.5)
        sinlat = math.sin(-math.pi * 0.25 - blh[0] * 0.5)

        quat = np.array([-sinlat * sinlon,sinlat * coslon,coslat * sinlon,coslat * coslon])

        return quat #[x y x w]
    
    # 从n系到e系转换四元数得到纬度和经度
    @staticmethod
    def blh(qne:np.ndarray, height:float) -> np.ndarray:
        
        return np.array([-2 * math.atan(qne[1] / qne[3]) - math.pi * 0.5, 2 * math.atan2(qne[2], qne[3]), height], dtype=float)
    
    # 地理坐标(纬度、经度和高程)转地心地固坐标
    @staticmethod
    def blh2ecef(blh:np.ndarray) -> np.ndarray:
        coslat = math.cos(blh[0])
        sinlat = math.sin(blh[0])
        coslon = math.cos(blh[1])
        sinlon = math.sin(blh[1])

        rn = Earth.RN(blh[0])
        rnh = rn + blh[2]

        x = rnh * coslat * coslon
        y = rnh * coslat * sinlon
        z = (rnh - rn * Earth.WGS84_E1) * sinlat

        return np.array([x, y, z])
    
    # 地心地固坐标转地理坐标
    @staticmethod
    def ecef2blh(ecef:np.ndarray) -> np.ndarray:
        p = math.sqrt(ecef[0] ** 2 + ecef[1] ** 2)
        rn = 0
        lat = 0
        lon = 0
        h = 0
        h2 = 0

        # 初始状态
        lat = math.atan(ecef[2] / (p * 1.0 - Earth.WGS84_E1))
        lon = 2.0 * math.atan2(ecef[1], ecef[0] + p)

        while abs(h - h2) > 1.0e-4:
            h2 = h
            rn = Earth.RN(lat)
            h = p / math.cos(lat) - rn
            lat = math.atan(ecef[2] / (p * (1.0 - Earth.WGS84_E1 * rn / (rn + h))))

        return np.array([lat, lon, h])
    
    # n系相对位置转地理坐标相对位置
    @staticmethod
    def DRi(blh:np.ndarray) -> np.ndarray:
        dri = np.zeros((3,3))
        rmn = Earth.meridianPrimeVerticalRadius(blh[0])
        dri[0,0] = 1.0 / (rmn[0] + blh[2])
        dri[1,1] = 1.0 / ((rmn[1] + blh[2]) * np.cos(blh[0]))
        dri[2,2] = -1
        return dri

    # 地理坐标相对位置转n系相对位置
    @staticmethod
    def DR(blh:np.ndarray) -> np.ndarray:
        dr = np.zeros((3, 3))
        rmn = Earth.meridianPrimeVerticalRadius(blh[0])
        dr[0, 0] = rmn[0] + blh[2]
        dr[1, 1] = (rmn[1] + blh[2]) * np.cos(blh[0])
        dr[2, 2] = -1
        return dr
    
    # 局部坐标(在origin处展开)转地理坐标
    @staticmethod
    def local2global(origin:np.ndarray, local:np.ndarray) -> np.ndarray:
        ecef0 = Earth.blh2ecef(origin)
        cn0e = Earth.cne(origin)

        ecef1 = ecef0 + cn0e @ local
        blh1 = Earth.ecef2blh(ecef1)

        return blh1
    
    # 地理坐标转局部坐标(在origin处展开)
    @staticmethod
    def global2local(origin:np.ndarray, global_pos:np.ndarray) -> np.ndarray:
        ecef0 = Earth.blh2ecef(origin)
        cn0e = Earth.cne(origin)
        ecef1 = Earth.blh2ecef(global_pos)
        return np.matmul(cn0e.T, (ecef1 - ecef0))
    
    
    @staticmethod
    def local2global_P(origin:np.ndarray, local:ty.Pose) -> ty.Pose:
        global_pose = ty.Pose()

        ecef0 = Earth.blh2ecef(origin)
        cn0e = Earth.cne(origin)

        ecef1 = ecef0 + cn0e @ local.t
        blh1 = Earth.ecef2blh(ecef1)
        cn1e = Earth.cne(blh1)

        global_pose.t = blh1
        global_pose.R = cn1e.T @ cn0e @ local.R

        return global_pose
    
    @staticmethod
    def global2local_P(origin:np.ndarray, global_pose:ty.Pose) -> ty.Pose:
        local_pose = ty.Pose()

        ecef0 = Earth.blh2ecef(origin)
        cn0e = Earth.cne(origin)

        ecef1 = Earth.blh2ecef(global_pose.t)
        cn1e = Earth.cne(global_pose.t)

        local_pose.t = cn0e.T @ (ecef1 - ecef0)
        local_pose.R = cn0e.T @ cn1e @ global_pose.R

        return local_pose
    

    # 地球自转角速度投影到e系
    @staticmethod
    def iewe() -> np.ndarray:
        return {0, 0, Earth.WGS84_WIE}
    
    # 地球自转角速度投影到n系
    @staticmethod
    def iewn(lat:float) -> np.ndarray:
        return np.array([Earth.WGS84_WIE * np.cos(lat), 0, -Earth.WGS84_WIE * np.sin(lat)])
    
    @staticmethod
    def iewn_at(origin:float, local:float) -> np.ndarray:
        global_pos = Earth.local2global(origin, local) 
        return Earth.iewn(global_pos[0])
    
    # n系相对于e系转动角速度投影到n系
    @staticmethod
    def enwn(rmn:np.ndarray, blh:np.ndarray, vel:np.ndarray) -> np.ndarray:
        return np.array([vel[1] / (rmn[1] + blh[2]), -vel[0] / (rmn[0] + blh[2]), -vel[1] * np.tan(blh[0]) / (rmn[1] + blh[2])])
    
    @staticmethod
    def enwn_at(origin:np.ndarray, local:np.ndarray, vel:np.ndarray) -> np.ndarray:
        global_pos = Earth.local2global(origin, local) 
        rmn = Earth.meridianPrimeVerticalRadius(global_pos[0]) 
        return Earth.enwn(rmn, global_pos, vel)
    
    