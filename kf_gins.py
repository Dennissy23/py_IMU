from angle import Angle
import gi_engine as gi
import numpy as np
import kf_gins_types as ky
import types_my as ty
import sys

class IMU_O:
    # IMU和GNSS数据文件路径
    imupath = r"C:\\Users\\Administrator\\Desktop\\WSN+IMU定位\\数据\\测试集\\IMU测试集.csv"
    gnsspath =  r"C:\\Users\\Administrator\\Desktop\\WSN+IMU定位\\数据\\测试集\\间隔5秒_前10秒连续\\GNSS间隔5秒测试集.csv"
    # IMU文件列数 (只会用前7列IMU数据)
    imudatalen = 7
    # IMU原始数据频率
    # IMU datarate [Hz]
    imudatarate = 200
    # 处理时间段，结束时间设置为-1时则处理至IMU文件结束
    # processing interval[s]
    starttime = 357485.0
    endtime = -1
    # 初始状态
    # 初始位置, 纬度 经度 高程
    # initial position, latitude, longitude, altitude. [deg, deg, m]
    initpos = np.array([  30.4604514341, 114.4719753313, 22.93866 ])
    # 初始速度, 北向速度, 东向速度, 垂向速度
    # initial velocity, speed in the directions of north, east and down. [m/s, m/s, m/s]
    initvel = np.array([ 0.114, -7.18422, 0.00324 ])
    # 初始姿态, 欧拉角(ZYX旋转顺序), 横滚, 俯仰, 航向
    # initial attitude, euler angle(ZYX rotation), roll, pitch, yaw [deg, deg, deg]
    initatt = np.array([ 0.33721 ,  -0.44062 ,  271.39921 ])
    # 初始IMU零偏和比例因子, IMU的三个轴(前、右、下)
    # initial IMU bias and scale error, three axes of the imu (forward, right and down)
    initgyrbias = np.array([ 0.0, 0.0, 0.0])    # [deg/h]
    initaccbias = np.array([ 0.0, 0.0, 0.0 ])    # [mGal]
    initgyrscale = np.array([ 0.0, 0.0, 0.0 ])   # [ppm]
    initaccscale = np.array([ 0.0, 0.0, 0.0 ])   # [ppm]
    # 初始状态标准差
    # initial state std
    # 初始位置标准差, 导航坐标系下 北向, 东向和垂向
    # initial position std, north, east and down in n-frame. [m, m, m]
    initposstd = np.array([ 0.005, 0.004, 0.008 ])
    # 初始速度标准差, 导航坐标系下北向、东向和垂向速度
    # initial velocity std, north, east and down speed in n-frame. [m/s, m/s, m/s]
    initvelstd = np.array([ 0.003, 0.004, 0.004 ])
    # 初始姿态标准差, 横滚、俯仰、航向角标准差
    # initial attitude std, roll, pitch and yaw std. [deg, deg, deg]
    initattstd = np.array([ 0.003, 0.003, 0.023 ])
    # IMU噪声建模参数, IMU的三个轴
    arw = np.array([0.2, 0.2, 0.2])          # [deg/sqrt(hr)]
    vrw = np.array([0.1, 0.1, 0.1])         # [m/s/sqrt(hr)]
    gbstd = np.array([20.0, 20.0, 20.0])        # [deg/hr]
    abstd = np.array([100.0, 100.0, 100.0])     # [mGal]
    gsstd = np.array([300.0, 300.0, 300.0])  # [ppm]
    asstd = np.array([300.0, 300.0, 300.0])  # [ppm]
    corrtime = 1.0                    # [hr]
    # 天线杆臂, IMU坐标系前右下方向
    # antenna lever, forward, right and down in the imu frame. [m]
    antlever =  np.array([ 0.045, 0.46, -0.238 ])

def LoadOptions():
    ## 读取初始位置(纬度 经度 高程)、(北向速度 东向速度 垂向速度)、姿态(欧拉角，ZYX旋转顺序, 横滚角、俯仰角、航向角)
    options = ky.GINSOptions()
    options.initstate.pos = IMU_O.initpos * Angle.D2R
    options.initstate.vel = IMU_O.initvel
    options.initstate.euler = IMU_O.initatt * Angle.D2R
    options.initstate.pos[2] *= Angle.R2D

    ## 读取IMU误差初始值(零偏和比例因子)
    options.initstate.imuerror.gyrbias = IMU_O.initgyrbias * Angle.D2R/3600.0
    options.initstate.imuerror.accbias = IMU_O.initaccbias * 1e-5
    options.initstate.imuerror.gyrscale = IMU_O.initgyrscale * 1e-6
    options.initstate.imuerror.accscale = IMU_O.initaccscale * 1e-6

    ## 读取初始位置、速度、姿态(欧拉角)的标准差
    options.initstate_std.pos = IMU_O.initposstd
    options.initstate_std.vel = IMU_O.initvelstd
    options.initstate_std.euler = IMU_O.initattstd * Angle.D2R

    ## 读取IMU噪声参数
    options.imunoise.gyr_arw = IMU_O.arw
    options.imunoise.acc_vrw = IMU_O.vrw
    options.imunoise.gyrbias_std = IMU_O.gbstd
    options.imunoise.accbias_std = IMU_O.abstd
    options.imunoise.gyrscale_std = IMU_O.gsstd
    options.imunoise.accscale_std = IMU_O.asstd
    options.imunoise.corr_time = IMU_O.corrtime

    ## 读取IMU误差初始标准差,如果配置文件中没有设置，则采用IMU噪声参数中的零偏和比例因子的标准差
    options.initstate_std.imuerror.gyrbias = IMU_O.gbstd * Angle.D2R / 3600.0
    options.initstate_std.imuerror.accbias = IMU_O.abstd * 1e-5
    options.initstate_std.imuerror.gyrscale = IMU_O.gsstd * 1e-6
    options.initstate_std.imuerror.accscale = IMU_O.asstd * 1e-6

    ## IMU噪声参数转换为标准单位
    options.imunoise.gyr_arw *= (Angle.D2R / 60.0)
    options.imunoise.acc_vrw /= 60.0
    options.imunoise.gyrbias_std *= (Angle.D2R / 3600.0)
    options.imunoise.accbias_std *= 1e-5
    options.imunoise.gyrscale_std *= 1e-6
    options.imunoise.accscale_std *= 1e-6
    options.imunoise.corr_time *= 3600

    ## GNSS天线杆臂, GNSS天线相位中心在IMU坐标系下位置
    options.antlever = IMU_O.antlever
    return options

def imuload(data_,rate,pre_time):
    dt_ = 1.0 / rate
    imu_ = ty.IMU()
    imu_.time = data_[0]
    imu_.dtheta = np.array(data_[1:4])
    imu_.dvel = np.array(data_[4:7])
    dt = imu_.time - pre_time
    pre_time = imu_.time
    if dt < 0.1:
        imu_.dt = dt
    else:
        imu_.dt = dt_
    return imu_,pre_time

def gnssload(data_):
    gnss_ = ty.GNSS()
    gnss_.time = data_[0]
    gnss_.blh = np.array(data_[1:4])
    gnss_.std = np.array(data_[4:7])
    gnss_.blh[0] *= Angle.D2R
    gnss_.blh[1] *= Angle.D2R
    return(gnss_)

def align(imu_data,gnss_data,starttime):
    imu_cur = ty.IMU()
    gnss = ty.GNSS()
    imu_index = 0
    gnss_index = 0
    pre_time = starttime
    p_t = 0
    for index,row in enumerate(imu_data):
        imu_cur,p_t = imuload(row,imudatarate,pre_time)
        imu_index = index
        if row[0] > starttime:
            break  
    for index,row in enumerate(gnss_data):
        gnss = gnssload(row)
        gnss_index = index
        if row[0] > starttime:
            break
    return imu_cur,gnss,imu_index,gnss_index,p_t


nav_result = np.empty((0, 10))
options = LoadOptions()

giengine = gi.GIEngine()
giengine.GIFunction(options)

imudatarate = IMU_O.imudatarate
starttime = IMU_O.starttime
endtime = IMU_O.endtime
pre_time = starttime
imu_data = np.genfromtxt(IMU_O.imupath,delimiter=',')
gnss_data = np.genfromtxt(IMU_O.gnsspath,delimiter=',')
if endtime < 0 :
    endtime = gnss_data[-1, 0]

imu_cur,gnss,is_index,gs_index,pre_time = align(imu_data,gnss_data,starttime)

giengine.addImuData(imu_cur, True)
giengine.addGnssData(gnss)
for row in imu_data[is_index+1:]:
    
    if gnss.time < imu_cur.time and gnss.time+1!= endtime:
        gnss = gnssload(gnss_data[gs_index])
        gs_index += 1
        giengine.addGnssData(gnss)
    
    imu_cur,pre_time = imuload(row,imudatarate,pre_time)
    if imu_cur.time > endtime:
        break
    giengine.addImuData(imu_cur)

    giengine.newImuProcess()

    timestamp = giengine.timestamp()
    navstate  = giengine.getNavState()
    # cov       = giengine.getCovariance()
    result = np.array([np.round(timestamp,9),np.round(navstate.pos[1]* Angle.R2D,9),np.round(navstate.pos[0]* Angle.R2D,9),  np.round(navstate.pos[2],9),np.round(navstate.vel[0],9), np.round(navstate.vel[1],9), np.round(navstate.vel[2],9),np.round(navstate.euler[0]* Angle.R2D,9),np.round(navstate.euler[1]* Angle.R2D,9),np.round(navstate.euler[2]* Angle.R2D,9)])

    nav_result = np.vstack((nav_result, result))

    sys.stdout.write('\r' + str(timestamp))
    sys.stdout.flush()
np.savetxt(r"C:\\Users\\Administrator\\Desktop\\WSN+IMU定位\\数据\\测试集\\间隔5秒_前10秒连续\\result.csv", nav_result, delimiter=",",fmt="%6f")    





