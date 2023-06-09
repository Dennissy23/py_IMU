from angle import Angle
import gi_engine as gi
import numpy as np
import kf_gins_types as ky
import types_my as ty
import sys
import yaml

with open('kf-gins.yaml', 'r',encoding='utf-8') as file:
        config = yaml.safe_load(file)
        
class IMU_O:
    
        
    # IMU和GNSS数据文件路径
    imupath = r"C:\\Users\\Administrator\\Desktop\\WSN+IMU定位\\代码\\data processing\\IMU_2023-06-01 2046改.csv"
    gnsspath =  r"C:\\Users\\Administrator\\Desktop\\WSN+IMU定位\\代码\\data processing\\RTK_2023-06-01 2046改2.csv"
    # IMU文件列数 (只会用前7列IMU数据)
    imudatalen = 7
    # IMU原始数据频率
    # IMU datarate [Hz]
    imudatarate = 200
    # 处理时间段，结束时间设置为-1时则处理至IMU文件结束
    # processing interval[s]
    starttime = 1685623563.0
    endtime = -1
    # 初始状态
    # 初始位置, 纬度 经度 高程
    # initial position, latitude, longitude, altitude. [deg, deg, m]
    initpos = np.array([  23.1579337, 113.3437332, 22.9325 ])
    # 初始速度, 北向速度, 东向速度, 垂向速度
    # initial velocity, speed in the directions of north, east and down. [m/s, m/s, m/s]
    initvel = np.array([ 0.0, 0.0, 0.0 ])
    # 初始姿态, 欧拉角(ZYX旋转顺序), 横滚, 俯仰, 航向
    # initial attitude, euler angle(ZYX rotation), roll, pitch, yaw [deg, deg, deg]
    initatt = np.array([ 1.384866476, -1.366858721 ,  265.3240585 ])
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
    arw = np.array([0.3, 0.3, 0.3])          # [deg/sqrt(hr)]
    vrw = np.array([0.04, 0.04, 0.04])         # [m/s/sqrt(hr)]
    gbstd = np.array([15.0, 15.0, 15.0])        # [deg/hr]
    abstd = np.array([100.0, 100.0, 100.0])     # [mGal]
    gsstd = np.array([550.0, 550.0, 550.0])  # [ppm]
    asstd = np.array([300.0, 300.0, 300.0])  # [ppm]
    corrtime = 1.0                    # [hr]
    # 天线杆臂, IMU坐标系前右下方向
    # antenna lever, forward, right and down in the imu frame. [m]
    antlever =  np.array([ 0,0,0 ])

def LoadOptions():
    ## 读取初始位置(纬度 经度 高程)、(北向速度 东向速度 垂向速度)、姿态(欧拉角，ZYX旋转顺序, 横滚角、俯仰角、航向角)
    options = ky.GINSOptions()
    options.initstate.pos = np.array(config['initpos']) * Angle.D2R
    options.initstate.vel = np.array(config['initvel'])
    options.initstate.euler = np.array(config['initatt']) * Angle.D2R
    options.initstate.pos[2] *= Angle.R2D

    ## 读取IMU误差初始值(零偏和比例因子)
    options.initstate.imuerror.gyrbias = np.array(config['initgyrbias']) * Angle.D2R/3600.0
    options.initstate.imuerror.accbias = np.array(config['initaccbias']) * 1e-5
    options.initstate.imuerror.gyrscale = np.array(config['initgyrscale']) * 1e-6
    options.initstate.imuerror.accscale = np.array(config['initaccscale']) * 1e-6

    ## 读取初始位置、速度、姿态(欧拉角)的标准差
    options.initstate_std.pos = np.array(config['initposstd'])
    options.initstate_std.vel = np.array(config['initvelstd'])
    options.initstate_std.euler = np.array(config['initattstd']) * Angle.D2R

    ## 读取IMU噪声参数
    options.imunoise.gyr_arw = np.array(config['arw'])
    options.imunoise.acc_vrw = np.array(config['vrw'])
    options.imunoise.gyrbias_std = np.array(config['gbstd'])
    options.imunoise.accbias_std = np.array(config['abstd'])
    options.imunoise.gyrscale_std = np.array(config['gsstd'])
    options.imunoise.accscale_std = np.array(config['asstd'])
    options.imunoise.corr_time = config['corrtime']

    ## 读取IMU误差初始标准差,如果配置文件中没有设置，则采用IMU噪声参数中的零偏和比例因子的标准差
    options.initstate_std.imuerror.gyrbias = np.array(config['gbstd']) * Angle.D2R / 3600.0
    options.initstate_std.imuerror.accbias = np.array(config['abstd']) * 1e-5
    options.initstate_std.imuerror.gyrscale = np.array(config['gsstd']) * 1e-6
    options.initstate_std.imuerror.accscale = np.array(config['asstd']) * 1e-6

    ## IMU噪声参数转换为标准单位
    options.imunoise.gyr_arw *= (Angle.D2R / 60.0)
    options.imunoise.acc_vrw /= 60.0
    options.imunoise.gyrbias_std *= (Angle.D2R / 3600.0)
    options.imunoise.accbias_std *= 1e-5
    options.imunoise.gyrscale_std *= 1e-6
    options.imunoise.accscale_std *= 1e-6
    options.imunoise.corr_time *= 3600

    ## GNSS天线杆臂, GNSS天线相位中心在IMU坐标系下位置
    options.antlever = np.array(config['antlever'])
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
error_result = np.empty((0, 13))
options = LoadOptions()

giengine = gi.GIEngine()
giengine.GIFunction(options)

imudatarate = config['imudatarate']
starttime = config['starttime']
endtime = config['endtime']
pre_time = starttime
imu_data = np.genfromtxt(config['imupath'],delimiter=',')
gnss_data = np.genfromtxt(config['gnsspath'],delimiter=',')
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
    imuerr = navstate.imuerror
    # cov       = giengine.getCovariance()
    result1 = np.array([np.round(timestamp,9),np.round(navstate.pos[1]* Angle.R2D,9),np.round(navstate.pos[0]* Angle.R2D,9),  np.round(navstate.pos[2],9),np.round(navstate.vel[0],9), np.round(navstate.vel[1],9), np.round(navstate.vel[2],9),np.round(navstate.euler[0]* Angle.R2D,9),np.round(navstate.euler[1]* Angle.R2D,9),np.round(navstate.euler[2]* Angle.R2D,9)])
    result2 = np.array([np.round(timestamp,9),np.round(imuerr.gyrbias[0]* Angle.R2D*3600,9),np.round(imuerr.gyrbias[1]* Angle.R2D*3600,9),  np.round(imuerr.gyrbias[2]* Angle.R2D*3600,9),np.round(imuerr.accbias[0]* 1e5,9), np.round(imuerr.accbias[1]* 1e5,9), np.round(imuerr.accbias[2]* 1e5,9),np.round(imuerr.gyrscale[0] * 1e6,9),np.round(imuerr.gyrscale[1] * 1e6,9),np.round(imuerr.gyrscale[2] * 1e6,9),np.round(imuerr.accscale[0] * 1e6,9),np.round(imuerr.accscale[1] * 1e6,9),np.round(imuerr.accscale[2] * 1e6,9)])
    nav_result = np.vstack((nav_result, result1))
    error_result = np.vstack((error_result, result2))

    sys.stdout.write('\r' + str(timestamp))
    sys.stdout.flush()
np.savetxt(config['outputpath_nav'], nav_result, delimiter=",",fmt="%6f")    
np.savetxt(config['outputpath_error'], error_result, delimiter=",",fmt="%6f")    





