import types_my as ty
import kf_gins_types as kf
from rotation import Rotation_my as ro
from insmech import INSMech as INS
from earth import Earth
from enum import IntEnum
import numpy as np 
import os
from scipy.spatial.transform import Rotation
from mpmath import mp, matrix

class StateID(IntEnum):
    P_ID = 0
    V_ID = 3
    PHI_ID = 6
    BG_ID = 9
    BA_ID = 12
    SG_ID = 15
    SA_ID = 18

class NoiseID(IntEnum):
    VRW_ID = 0
    ARW_ID = 3
    BGSTD_ID = 6
    BASTD_ID = 9
    SGSTD_ID = 12
    SASTD_ID = 15
    
class GIEngine:
    ##  更新时间对齐误差，IMU状态和观测信息误差小于它则认为两者对齐
    TIME_ALIGN_ERR = 0.001
    ## Kalman滤波相关
    RANK = 21
    NOISERANK = 18

    def __init__(self):
        self.options_ = kf.GINSOptions()
        self.timestamp_ = 0.0
        ##   IMU和GNSS原始数据
        self.imupre_ = ty.IMU()
        self.imucur_ = ty.IMU()
        self.gnssdata_ = ty.GNSS()
        ##  IMU状态（位置、速度、姿态和IMU误差）
        self.pvacur_ = kf.PVA()
        self.pvapre_ = kf.PVA()
        self.imuerror_ = kf.ImuError()
        ## Kalman滤波相关
        self.Cov_ = np.zeros((GIEngine.RANK,GIEngine.RANK))
        self.Qc_ = np.zeros((GIEngine.NOISERANK,GIEngine.NOISERANK))
        self.dx_ = np.zeros((GIEngine.RANK,1))
    
    def GIFunction(self,options:kf.GINSOptions):
        self.options_ = options
        self.timestamp_ = 0.0
        imunoise  = self.options_.imunoise
        ## 初始化系统噪声阵
        self.Qc_[NoiseID.ARW_ID:NoiseID.ARW_ID+3, NoiseID.ARW_ID:NoiseID.ARW_ID+3] = np.diag(np.square(imunoise.gyr_arw))
        self.Qc_[NoiseID.VRW_ID:NoiseID.VRW_ID+3, NoiseID.VRW_ID:NoiseID.VRW_ID+3] = np.diag(np.square(imunoise.acc_vrw))
        self.Qc_[NoiseID.BGSTD_ID:NoiseID.BGSTD_ID+3, NoiseID.BGSTD_ID:NoiseID.BGSTD_ID+3] = 2/imunoise.corr_time * np.diag(np.square(imunoise.gyrbias_std))
        self.Qc_[NoiseID.BASTD_ID:NoiseID.BASTD_ID+3, NoiseID.BASTD_ID:NoiseID.BASTD_ID+3] = 2/imunoise.corr_time * np.diag(np.square(imunoise.accbias_std))
        self.Qc_[NoiseID.SGSTD_ID:NoiseID.SGSTD_ID+3, NoiseID.SGSTD_ID:NoiseID.SGSTD_ID+3] = 2/imunoise.corr_time * np.diag(np.square(imunoise.gyrscale_std))
        self.Qc_[NoiseID.SASTD_ID:NoiseID.SASTD_ID+3, NoiseID.SASTD_ID:NoiseID.SASTD_ID+3] = 2/imunoise.corr_time * np.diag(np.square(imunoise.accscale_std))
        ## 设置系统状态(位置、速度、姿态和IMU误差)初值和初始协方差
        initstate = self.options_.initstate
        initstate_std = self.options_.initstate_std
        self.initialize(initstate, initstate_std)
        

    def initialize(self,initstate:kf.NavState,initstate_std:kf.NavState):
        # 初始化位置、速度、姿态
        self.pvacur_.pos= initstate.pos
        self.pvacur_.vel= initstate.vel
        self.pvacur_.att.euler= initstate.euler
        self.pvacur_.att.cbn = ro.euler2matrix( initstate.euler)
        self.pvacur_.att.qbn = ro.euler2quaternion( np.flip(initstate.euler, axis=0))
        # 初始化IMU误差
        self.imuerror_ = initstate.imuerror
        # 给上一时刻状态赋同样的初值
        p = self.pvacur_
        self.pvapre_ = p
        # 初始化协方差
        imuerror_std = initstate_std.imuerror
        self.Cov_[StateID.P_ID:StateID.P_ID+3,StateID.P_ID:StateID.P_ID+3]= \
        np.diag(np.square(initstate_std.pos))
        self.Cov_[StateID.V_ID:StateID.V_ID+3,StateID.V_ID:StateID.V_ID+3]= \
        np.diag(np.square(initstate_std.vel))
        self.Cov_[StateID.PHI_ID:StateID.PHI_ID+3,StateID.PHI_ID:StateID.PHI_ID+3]= \
        np.diag(np.square(initstate_std.euler))
        self.Cov_[StateID.BG_ID:StateID.BG_ID+3,StateID.BG_ID:StateID.BG_ID+3]= \
        np.diag(np.square(imuerror_std.gyrbias))
        self.Cov_[StateID.BA_ID:StateID.BA_ID+3,StateID.BA_ID:StateID.BA_ID+3]= \
        np.diag(np.square(imuerror_std.accbias))
        self.Cov_[StateID.SG_ID:StateID.SG_ID+3,StateID.SG_ID:StateID.SG_ID+3]= \
        np.diag(np.square(imuerror_std.gyrscale))
        self.Cov_[StateID.SA_ID:StateID.SA_ID+3,StateID.SA_ID:StateID.SA_ID+3]= \
        np.diag(np.square(imuerror_std.accscale))

    def newImuProcess(self):
        ## 当前IMU时间作为系统当前状态时间
        time = self.imucur_.time
        self.timestamp_ = time
        ## 如果GNSS有效，则将更新时间设置为GNSS时间
        if self.gnssdata_.isvalid:
            updatetime = self.gnssdata_.time
        else:
            updatetime = -1
        ## 判断是否需要进行GNSS更新
        imupre_ = self.imupre_
        imucur_ = self.imucur_
        gnssdata_ = self.gnssdata_
        pvacur_ = self.pvacur_
        
        res = self.isToUpdate(imupre_.time, imucur_.time, updatetime)
    
        if res == 0:
            ## 只传播导航状态
            self.insPropagation(imupre_, imucur_)
        elif res == 1:
            ## GNSS数据靠近上一历元，先对上一历元进行GNSS更新
            self.gnssUpdate(gnssdata_)
            self.stateFeedback()
            self.pvapre_ = self.pvacur_
            self.insPropagation(imupre_, imucur_)
        elif res == 2:
            ## GNSS数据靠近当前历元，先对当前IMU进行状态传播
            self.insPropagation(imupre_, imucur_)
            self.gnssUpdate(gnssdata_)
            self.stateFeedback()
        else:
            ## GNSS数据在两个IMU数据之间(不靠近任何一个), 将当前IMU内插到整秒时刻
            midimu = ty.IMU
            imucur_, midimu = GIEngine.imuInterpolate(imupre_, imucur_, updatetime, midimu)
            ## 对前一半IMU进行状态传播
            self.insPropagation(imupre_, midimu)
            ## 整秒时刻进行GNSS更新，并反馈系统状态
            self.gnssUpdate(gnssdata_)
            self.stateFeedback()
            ## 对后一半IMU进行状态传播
            self.pvapre_ = self.pvacur_
            self.insPropagation(midimu, imucur_)

        ## 检查协方差矩阵对角线元素
        self.checkCov()

        ## 更新上一时刻的状态和IMU数据
        self.pvapre_ = self.pvacur_
        self.imupre_ = self.imucur_

    def checkCov(self):
        for i in range(GIEngine.RANK):
            if self.Cov_[i, i] < 0:
                print(f"Covariance is negative at {self.timestamp_:.10f} !")
                exit(os.EXIT_FAILURE)

    def imuCompensate(self,imu:ty.IMU) -> ty.IMU :
        ## 补偿IMU零偏误差
        imu.dtheta -= self.imuerror_.gyrbias * imu.dt
        imu.dvel -= self.imuerror_.accbias * imu.dt
        ## 补偿IMU比例因子误差
        one = np.ones(3)
        gyrscale = one + self.imuerror_.gyrscale
        accscale = one + self.imuerror_.accscale
        imu.dtheta = np.multiply(imu.dtheta,np.reciprocal(gyrscale))
        imu.dvel = np.multiply(imu.dvel,np.reciprocal(accscale))
        return imu
    
    def insPropagation(self,imupre:ty.IMU,imucur:ty.IMU):
        ## 对当前IMU数据(imucur)补偿误差, 上一IMU数据(imupre)已经补偿过了
        imucur = self.imuCompensate(imucur)
        ## IMU状态更新(机械编排算法)
        pvapre_ = self.pvapre_
        self.pvacur_ = INS.insMech(pvapre_, imupre, imucur)
        
        
        
        ## 系统噪声传播，姿态误差采用phi角误差模型
        ## 初始化Phi阵(状态转移矩阵)，F阵，Qd阵(传播噪声阵)，G阵(噪声驱动阵)
        # Phi = np.identity(self.Cov_.shape[0])
        # F = np.zeros_like(self.Cov_)
        # Qd = np.zeros_like(self.Cov_)
        Phi = np.identity(21)
        F = np.zeros((21,21))
        Qd = np.zeros((21,21))
        G = np.zeros((GIEngine.RANK, GIEngine.NOISERANK))
        ## 使用上一历元状态计算状态转移矩阵
        rmrn = Earth.meridianPrimeVerticalRadius(pvapre_.pos[0])
        gravity = Earth.gravity(pvapre_.pos)
        wie_n = np.array([Earth.WGS84_WIE * np.cos(pvapre_.pos[0]), 0, -Earth.WGS84_WIE * np.sin(pvapre_.pos[0])])
        wen_n = np.array([pvapre_.vel[1] / (rmrn[1] + pvapre_.pos[2]), -pvapre_.vel[0] / (rmrn[0] + pvapre_.pos[2]),-pvapre_.vel[1] * np.tan(pvapre_.pos[0]) / (rmrn[1] + pvapre_.pos[2])])
        rmh   = rmrn[0] + pvapre_.pos[2]
        rnh   = rmrn[1] + pvapre_.pos[2]
        accel = imucur.dvel / imucur.dt
        omega = imucur.dtheta / imucur.dt

        ## 位置误差
        temp = np.zeros((3,3))
        temp[0,0] = -pvapre_.vel[2] / rmh
        temp[0,2] =  pvapre_.vel[0] / rmh
        temp[1,0] = pvapre_.vel[1] * np.tan(pvapre_.pos[0]) / rnh
        temp[1,1] = -(pvapre_.vel[2] + pvapre_.vel[0] * np.tan(pvapre_.pos[0])) / rnh
        temp[1,2] = pvapre_.vel[1] / rnh
        F[StateID.P_ID:StateID.P_ID+3,StateID.P_ID:StateID.P_ID+3] = temp
        F[StateID.P_ID:StateID.P_ID+3,StateID.V_ID:StateID.V_ID+3] = np.identity(3)

        ## 速度误差
        temp = np.zeros((3,3))
        temp[0, 0] = -2 * pvapre_.vel[1] * Earth.WGS84_WIE * np.cos(pvapre_.pos[0]) / rmh -np.power(pvapre_.vel[1], 2) / rmh / rnh / np.power(np.cos(pvapre_.pos[0]), 2)
        temp[0, 2] = pvapre_.vel[0] * pvapre_.vel[2] / rmh / rmh - np.power(pvapre_.vel[1], 2) * np.tan(pvapre_.pos[0]) / rnh / rnh
        temp[1, 0] = 2 * Earth.WGS84_WIE * (pvapre_.vel[0] * np.cos(pvapre_.pos[0]) - pvapre_.vel[2] * np.sin(pvapre_.pos[0])) / rmh + pvapre_.vel[0] * pvapre_.vel[1] / rmh / rnh / np.power(np.cos(pvapre_.pos[0]), 2)
        temp[1, 2] = (pvapre_.vel[1] * pvapre_.vel[2] + pvapre_.vel[0] * pvapre_.vel[1] * np.tan(pvapre_.pos[0])) / rnh / rnh
        temp[2, 0] = 2 * Earth.WGS84_WIE * pvapre_.vel[1] * np.sin(pvapre_.pos[0]) / rmh
        temp[2, 2] = -np.power(pvapre_.vel[1], 2) / rnh / rnh - np.power(pvapre_.vel[0], 2) / rmh / rmh +2 * gravity / (np.sqrt(rmrn[0] * rmrn[1]) + pvapre_.pos[2])
        F[StateID.V_ID:StateID.V_ID+3,StateID.P_ID:StateID.P_ID+3] = temp
        temp = np.zeros((3,3))
        temp[0, 0] = pvapre_.vel[2] / rmh
        temp[0, 1] = -2 * (Earth.WGS84_WIE * np.sin(pvapre_.pos[0]) + pvapre_.vel[1] * np.tan(pvapre_.pos[0]) / rnh)
        temp[0, 2] = pvapre_.vel[0] / rmh
        temp[1, 0] = 2 * Earth.WGS84_WIE * np.sin(pvapre_.pos[0]) + pvapre_.vel[1] * np.tan(pvapre_.pos[0]) / rnh
        temp[1, 1] = (pvapre_.vel[2] + pvapre_.vel[0] * np.tan(pvapre_.pos[0])) / rnh
        temp[1, 2] = 2 * Earth.WGS84_WIE * np.cos(pvapre_.pos[0]) + pvapre_.vel[1] / rnh
        temp[2, 0] = -2 * pvapre_.vel[0] / rmh
        temp[2, 1] = -2 * (Earth.WGS84_WIE * np.cos(pvapre_.pos[0]) + pvapre_.vel[1] / rnh)
        F[StateID.V_ID:StateID.V_ID+3,StateID.V_ID:StateID.V_ID+3] = temp
        F[StateID.V_ID:StateID.V_ID+3,StateID.PHI_ID:StateID.PHI_ID+3] = ro.skewSymmetric(pvapre_.att.cbn @ accel)
        F[StateID.V_ID:StateID.V_ID+3,StateID.BA_ID:StateID.BA_ID+3] = pvapre_.att.cbn
        F[StateID.V_ID:StateID.V_ID+3,StateID.SA_ID:StateID.SA_ID+3] = pvapre_.att.cbn @ np.diag(accel)

        ## 姿态误差
        temp = np.zeros((3,3))
        temp[0, 0] = -Earth.WGS84_WIE * np.sin(pvapre_.pos[0]) / rmh
        temp[0, 2] = pvapre_.vel[1] / rnh / rnh
        temp[1, 2] = -pvapre_.vel[0] / rmh / rmh
        temp[2, 0] = -Earth.WGS84_WIE * np.cos(pvapre_.pos[0]) / rmh - pvapre_.vel[1] / rmh / rnh / np.power(np.cos(pvapre_.pos[0]), 2)
        temp[2, 2] = -pvapre_.vel[1] * np.tan(pvapre_.pos[0]) / rnh / rnh
        F[StateID.PHI_ID:StateID.PHI_ID+3,StateID.P_ID:StateID.P_ID+3] = temp
        temp = np.zeros((3,3))
        temp[0, 1] = 1 / rnh
        temp[1, 0] = -1 / rmh
        temp[2, 1] = -np.tan(pvapre_.pos[0]) / rnh
        F[StateID.PHI_ID:StateID.PHI_ID+3,StateID.V_ID:StateID.V_ID+3] = temp
        F[StateID.PHI_ID:StateID.PHI_ID+3,StateID.PHI_ID:StateID.PHI_ID+3] = -ro.skewSymmetric(wie_n + wen_n)
        F[StateID.PHI_ID:StateID.PHI_ID+3,StateID.BG_ID:StateID.BG_ID+3] = -pvapre_.att.cbn
        F[StateID.PHI_ID:StateID.PHI_ID+3,StateID.SG_ID:StateID.SG_ID+3] = -pvapre_.att.cbn @ np.diag(omega)

        ## IMU零偏误差和比例因子误差，建模成一阶高斯-马尔科夫过程
        F[StateID.BG_ID:StateID.BG_ID+3,StateID.BG_ID:StateID.BG_ID+3] = -1 / self.options_.imunoise.corr_time * np.identity(3)
        F[StateID.BA_ID:StateID.BA_ID+3,StateID.BA_ID:StateID.BA_ID+3] = -1 / self.options_.imunoise.corr_time * np.identity(3)
        F[StateID.SG_ID:StateID.SG_ID+3,StateID.SG_ID:StateID.SG_ID+3] = -1 / self.options_.imunoise.corr_time * np.identity(3)
        F[StateID.SA_ID:StateID.SA_ID+3,StateID.SA_ID:StateID.SA_ID+3] = -1 / self.options_.imunoise.corr_time * np.identity(3)

        ## 系统噪声驱动矩阵
        G[StateID.V_ID:StateID.V_ID+3,NoiseID.VRW_ID:NoiseID.VRW_ID+3] =  pvapre_.att.cbn
        G[StateID.PHI_ID:StateID.PHI_ID+3,NoiseID.ARW_ID:NoiseID.ARW_ID+3] =  pvapre_.att.cbn
        G[StateID.BG_ID:StateID.BG_ID+3,NoiseID.BGSTD_ID:NoiseID.BGSTD_ID+3] = np.identity(3)
        G[StateID.BA_ID:StateID.BA_ID+3,NoiseID.BASTD_ID:NoiseID.BASTD_ID+3] = np.identity(3)
        G[StateID.SG_ID:StateID.SG_ID+3,NoiseID.SGSTD_ID:NoiseID.SGSTD_ID+3] = np.identity(3)
        G[StateID.SA_ID:StateID.SA_ID+3,NoiseID.SASTD_ID:NoiseID.SASTD_ID+3] = np.identity(3)

        ## 状态转移矩阵
        Phi = Phi + F * imucur.dt
        ## 计算系统传播噪声
        # for i in range(21):
        #     for j in range(21):
        #         print(Phi[i][j], end=" ")
        #     print()
        # print()
        Qd = G @ self.Qc_ @ G.T * imucur.dt
        
        Qd = (Phi @ Qd @ Phi.T + Qd) / 2
        
        # print(Phi[3,7])
        ## EKF预测传播系统协方差和系统误差状态
        self.EKFPredict(Phi, Qd)

    def gnssUpdate(self,gnssdata:ty.GNSS):
        ## IMU位置转到GNSS天线相位中心位置
        Dr_inv = Earth.DRi(self.pvacur_.pos)
        Dr = Earth.DR(self.pvacur_.pos)
        antenna_pos = self.pvacur_.pos + Dr_inv @ self. pvacur_.att.cbn @ self.options_.antlever
        ## GNSS位置测量新息
        dz = Dr @ (antenna_pos - gnssdata.blh)
        ## 构造GNSS位置观测矩阵
        H_gnsspos = np.zeros((3, self.Cov_.shape[0]))
        H_gnsspos[0:3,StateID.P_ID:StateID.P_ID+3] = np.identity(3)
        H_gnsspos[0:3,StateID.PHI_ID:StateID.PHI_ID+3] = ro.skewSymmetric(self.pvacur_.att.cbn @ self.options_.antlever)
        ## 位置观测噪声阵
        R_gnsspos = np.diag(np.multiply(gnssdata.std, gnssdata.std))
        ## EKF更新协方差和误差状态
        dz = dz.reshape(3, 1)
        self.EKFUpdate(dz, H_gnsspos, R_gnsspos)
        ## GNSS更新之后设置为不可用
        self.gnssdata_.isvalid = False
          
    def isToUpdate(self,imutime1:float,imutime2:float,updatetime:float) -> int:
        if np.abs(imutime1 - updatetime) < GIEngine.TIME_ALIGN_ERR :
            ## 更新时间靠近imutime1
            return 1
        elif np.abs(imutime2 - updatetime) <= GIEngine.TIME_ALIGN_ERR :
            ## 更新时间靠近imutime2
            return 2
        elif imutime1 < updatetime and updatetime < imutime2 :
            ## 更新时间在imutime1和imutime2之间, 但不靠近任何一个
            return 3
        else:
            ## 更新时间不在imutimt1和imutime2之间，且不靠近任何一个
            return 0
        
    def EKFPredict(self,Phi,Qd):
        # assert Phi.shape[0] == self.Cov_.shape[0]
        # assert Qd.shape[0] == self.Cov_.shape[0]

        
        ## 传播系统协方差和误差状态
       
        self.Cov_ = Phi @ self.Cov_ @ Phi.transpose() + Qd
        self.dx_  = Phi @ self.dx_
        

    def EKFUpdate(self,dz:np.ndarray,H:np.ndarray,R:np.ndarray):
        assert  H.shape[1] == self.Cov_.shape[0]
        assert  dz.shape[0] == H.shape[0]
        assert  dz.shape[0] == R.shape[0]
        # assert  dz.shape[1] == 1

        ## 计算Kalman增益
        temp =  H @ self.Cov_ @ H.transpose() + R
        K =  self.Cov_ @ H.transpose() @ np.linalg.inv(temp)

        ## 更新系统误差状态和协方差
        I = np.identity(self.Cov_.shape[0])
        I = I - K @ H
        ## 如果每次更新后都进行状态反馈，则更新前dx_一直为0，下式可以简化为：dx_ = K * dz
        self.dx_  = self.dx_ + K @ (dz - H @ self.dx_)
        self.Cov_ = I @ self.Cov_ @ I.transpose() + K @ R @ K.transpose()

    def stateFeedback(self):
        ## 位置误差反馈
        delta_r = np.concatenate(self.dx_[StateID.P_ID:StateID.P_ID+3,0:1])
        Dr_inv = Earth.DRi(self.pvacur_.pos)
        self.pvacur_.pos -= Dr_inv @ delta_r
        ## 速度误差反馈
        vectemp = np.concatenate(self.dx_[StateID.V_ID:StateID.V_ID+3,0:1])
        self.pvacur_.vel -= vectemp
        ## 姿态误差反馈
        vectemp = np.concatenate(self.dx_[StateID.PHI_ID:StateID.PHI_ID+3,0:1])
        qpn = ro.rotvec2quaternion(vectemp)
        qqq = Rotation.from_quat(qpn) * Rotation.from_quat(self.pvacur_.att.qbn)
        self.pvacur_.att.qbn = qqq.as_quat()
        self.pvacur_.att.cbn = ro.quaternion2matrix(self.pvacur_.att.qbn)
        self.pvacur_.att.euler = ro.matrix2euler(self.pvacur_.att.cbn)
        ## IMU零偏误差反馈
        vectemp = np.concatenate(self.dx_[StateID.BG_ID:StateID.BG_ID+3,0:1])
        self.imuerror_.gyrbias += vectemp
        vectemp = np.concatenate(self.dx_[StateID.BA_ID:StateID.BA_ID+3,0:1])
        self.imuerror_.accbias += vectemp
        ## IMU比例因子误差反馈
        vectemp = np.concatenate(self.dx_[StateID.SG_ID:StateID.SG_ID+3,0:1])
        self.imuerror_.gyrscale += vectemp
        vectemp = np.concatenate(self.dx_[StateID.SA_ID:StateID.SA_ID+3,0:1])
        self.imuerror_.accscale += vectemp
        ## 误差状态反馈到系统状态后,将误差状态清零
        self.dx_[:, :] = 0

    def addImuData(self,imu:ty.IMU, compensate = False ):
        i = self.imucur_
        self.imupre_ = i
        self.imucur_ = imu
        if compensate:
            self.imucur_ = self.imuCompensate(imu)

    def addGnssData(self,gnss:ty.GNSS):
        self.gnssdata_ = gnss
        self.gnssdata_.isvalid = True
    
    @staticmethod
    def imuInterpolate(imu1:ty.IMU,imu2:ty.IMU,timestamp:float,midimu:ty.IMU,):
        if imu1.time > timestamp or imu2.time < timestamp:
            return
        lamda = (timestamp - imu1.time) / (imu2.time - imu1.time)
        midimu.time   = timestamp
        midimu.dtheta = imu2.dtheta * lamda
        midimu.dvel   = imu2.dvel * lamda
        midimu.dt     = timestamp - imu1.time
        imu2.dtheta = imu2.dtheta - midimu.dtheta
        imu2.dvel   = imu2.dvel - midimu.dvel
        imu2.dt     = imu2.dt - midimu.dt
        return imu2,midimu

    def timestamp(self) -> float:
        return self.timestamp_
    
    def getNavState(self) ->  kf.NavState:
        state = kf.NavState()
        state.pos = self.pvacur_.pos
        state.vel  = self.pvacur_.vel
        state.euler  = self.pvacur_.att.euler
        state.imuerror = self.imuerror_
        return state
    
    def getCovariance(self) -> np.ndarray:
        return self.Cov_