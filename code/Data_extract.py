import os
from scipy import interpolate
import numpy as np
import pandas as pd
from scipy.io import loadmat
import h5py

#计算航向角
def calculate_heading_np(v_lat, v_lon):
    # 计算航向角
    theta_radians = np.arctan2(v_lon, v_lat)
    theta_degrees = np.degrees(theta_radians)

    # 确保航向角为正
    theta_degrees = np.where(theta_degrees < 0, theta_degrees + 360, theta_degrees)

    return theta_degrees

def preproceing2(index=2,base_path=None,save_path=None):
    """
    :param index: 原始数据的航次（weather、voyage、machine），需要进行提取的数据
    :param base_path: 原始数据的根目录
    :param save_path:保存total_data的目录
    :return:
    """
    weather_data = h5py.File(os.path.join(base_path, f'WeatherData_No.{index}.mat'))  # 10个类型数据，目前仅需要用这个即可
    # 航行数据
    voyage_data = h5py.File(os.path.join(base_path, f'VoyageData_No.{index}.mat'))  # 2个类型数据
    # 机舱数据
    machine_data = loadmat(os.path.join(base_path, f'MachineData_No.{index}.mat'))  # 22个类型数据

    # 重新构建csv数据文件格式
    # 天气环境因素数据
    new_weather_data = pd.DataFrame(weather_data['Weather_data']['data'][:].T,
                                    columns=['海洋表面温度', '海平面大气压', '海平面10m处经度方向风场分量',
                                             '海平面10m处纬度方向风场分量', '海平面2m处空气温度',
                                             '海平面2处露点温度', '海平面粗糙度', '联合风浪和涌浪的特征波高',
                                             '平均波浪方向', '平均波浪周期'])

    # 航行数据
    new_voyage_data = pd.DataFrame({'航速': voyage_data['Voyage_data']['speed'][:].reshape(-1),
                                    '航向': voyage_data['Voyage_data']['course'][:].reshape(-1)})

    # 机舱数据
    new_machine_data = pd.DataFrame(machine_data['Machine'][0][0][-1],
                                    columns=['采样时间', '冷却海水温度', '主机燃油进机温度', '主机1号涡轮增压器转速',
                                             '主机2号涡轮增压器转速', '轴转速', '轴功率'
                                        , '主、辅机燃油流量', '主机扫气压力', '主机扫气箱温度', '主机1号气缸排气温度',
                                             '主机2号气缸排气温度', '主机3号气缸排气温度', '主机4号气缸排气温度',
                                             '主机5号气缸排气温度', '主机6号气缸排气温度',
                                             '主机1号涡轮增压器废气进口温度', '主机2号涡轮增压器废气进口温度',
                                             '主机1号涡轮增压器废气出口温度', '主机2号涡轮增压器废气出口温度',
                                             '辅机燃油流量', '辅机燃油回油流量'])
    # 保存csv文件数据
    total_data = pd.concat([new_voyage_data, new_weather_data, new_machine_data], axis=1)  # 合并数据
    total_data.to_csv(os.path.join(save_path, f'total_data{index}.csv'), index=False, encoding='gbk')

def preproceing3(index=3,base_path=None,save_path=None):
    """
    :param index: 原始数据的航次（weather、voyage、machine）
    :param base_path: 原始数据的根目录
    :param save_path:保存total_data的目录
    :return:
    """
    # 天气环境因素数据
    weather_data = loadmat(os.path.join(base_path,f'WeatherData_No.{index}.mat'))  # 10个类型数据，目前仅需要用这个即可
    # 航行数据
    voyage_data = loadmat(os.path.join(base_path,f'VoyageData_No.{index}.mat'))  # 2个类型数据
    # 机舱数据
    machine_data = loadmat(os.path.join(base_path,f'MachineData_No.{index}.mat'))  # 22个类型数据

    # 重新构建csv数据文件格式
    # 天气环境因素数据
    new_weather_data = pd.DataFrame(weather_data['Weather'][0][0][0],
                                    columns=['海洋表面温度', '海平面大气压', '海平面10m处经度方向风场分量',
                                             '海平面10m处纬度方向风场分量', '海平面2m处空气温度',
                                             '海平面2处露点温度', '海平面粗糙度', '联合风浪和涌浪的特征波高',
                                             '平均波浪方向', '平均波浪周期'])
    # 航行数据
    new_voyage_data = pd.DataFrame({'航速': voyage_data['Voyage_data'][0][0][0].flatten().tolist(),
                                    '航向': voyage_data['Voyage_data'][0][0][1].flatten().tolist()})
    # 机舱数据
    new_machine_data = pd.DataFrame(machine_data['Machine'][0][0][-1],
                                    columns=['采样时间', '冷却海水温度', '主机燃油进机温度', '主机1号涡轮增压器转速',
                                             '主机2号涡轮增压器转速', '轴转速', '轴功率'
                                        , '主、辅机燃油流量', '主机扫气压力', '主机扫气箱温度', '主机1号气缸排气温度',
                                             '主机2号气缸排气温度', '主机3号气缸排气温度', '主机4号气缸排气温度',
                                             '主机5号气缸排气温度', '主机6号气缸排气温度',
                                             '主机1号涡轮增压器废气进口温度', '主机2号涡轮增压器废气进口温度',
                                             '主机1号涡轮增压器废气出口温度', '主机2号涡轮增压器废气出口温度',
                                             '辅机燃油流量', '辅机燃油回油流量'])
    # 保存csv文件数据
    total_data = pd.concat([new_voyage_data, new_weather_data, new_machine_data], axis=1)  # 合并数据

    total_data.to_csv(os.path.join(save_path,f'total_data{index}.csv'), index=False, encoding='gbk')


def liner_interpolation(start_index,end_index,original_data,target_num,num_feature=1):
    """
    :param start_index: 原始数据开始标签（天气和航行）
    :param end_index: 原始数据结束标签（天气和航行）
    :param original_data: 原始天气或航行数据序列（从开始索引到结束）
    :param target_num: 目标插值的数量（要插值与机舱的数据量相等）
    :param num_feature: 特征数量
    :return:
    """
    original_data=original_data[start_index:end_index+1]
    if num_feature==1:
        # 创建原始数据索引
        original_indices = np.arange(start_index, end_index + 1)  #天气和航行数据索引列
        # 生成新的目标索引数组（生成类似机舱索引列）
        target_indices = np.linspace(start_index, end_index, target_num)
        # 创建插值函数（拟合）
        interp_func = interpolate.interp1d(original_indices, original_data, kind='linear')
        # 使用插值函数进行重采样
        resampled_data = interp_func(target_indices)
        # 确保开始值和结束值不变
        resampled_data[0] = original_data[0]
        resampled_data[-1] = original_data[-1]
        return resampled_data
    else:
        output_data=np.zeros((target_num,num_feature))
        for i in range(num_feature): #每个特征
            original_data_i=original_data[:,i]
            # 创建原始数据索引
            original_indices = np.arange(start_index, end_index + 1)  # 天气和航行数据索引列
            # 生成新的目标索引数组（生成类似机舱索引列）
            target_indices = np.linspace(start_index, end_index, target_num)
            # 创建插值函数（拟合）
            interp_func = interpolate.interp1d(original_indices, original_data_i, kind='linear')
            # 使用插值函数进行重采样
            resampled_data = interp_func(target_indices)
            # 确保开始值和结束值不变
            resampled_data[0] = original_data_i[0]
            resampled_data[-1] = original_data_i[-1]
            output_data[:,i]=resampled_data
        return output_data

def preproceing_other(index=3,base_path=None,save_path=None):
    """
    :param index: 原始数据的航次（weather、voyage、machine）
    :param base_path: 原始数据的根目录
    :param save_path:保存total_data的目录
    :return:
    """

    # 时间序列数据
    time_data = loadmat(os.path.join(base_path, 'date_voyage.mat'))['date_v'].reshape(-1) #(n)
    #天气数
    weather_data = loadmat(os.path.join(base_path, 'WeatherData.mat'))['data'].T  # 10个类型数据（天气数据） (n,10)
    # 航行数据
    voyage_data =  loadmat(os.path.join(base_path, 'VoyageData.mat'))  # 2个类型数据航行数据
    # 机舱数据
    machine_data = loadmat(os.path.join(base_path, f'MachineData_No.{index}.mat'))['Machine'][0][0][-1]  # (m,22)22个类型数据

    # 'data_voyage'(9个特征)，'lat_v', 'lon_v', 'utc_time']
    #计算得到航向
    course = np.concatenate(voyage_data['data_voyage'][:,4]).reshape(-1)
    #还要得到航速
    speed = np.concatenate(voyage_data['data_voyage'][:,5]).reshape(-1)
    #得到起止时间
    start_time = machine_data[:,0][0]
    end_time= machine_data[:,0][-1]
    #得到起始索引(最接近的机舱数据时间的索引)
    start_index = np.where(np.isclose(time_data, start_time))[0][0]
    end_index = np.where(np.isclose(time_data, end_time))[0][0]

    #由于采样频率不同，通过线性插值的方式进行统一数据量（将航行和天气数据与航行数据进行统一）
    #进行提取出所需的天气数据和航行数据
    weather_data=liner_interpolation(start_index=start_index,end_index=end_index,original_data=weather_data,target_num=len(machine_data),num_feature=10)
    course=liner_interpolation(start_index=start_index,end_index=end_index,original_data=course,target_num=len(machine_data),num_feature=1)
    speed=liner_interpolation(start_index=start_index,end_index=end_index,original_data=speed,target_num=len(machine_data),num_feature=1)

    #进行存储
    new_weather_data = pd.DataFrame(weather_data,
                                    columns=['海洋表面温度', '海平面大气压', '海平面10m处经度方向风场分量',
                                             '海平面10m处纬度方向风场分量', '海平面2m处空气温度',
                                             '海平面2处露点温度', '海平面粗糙度', '联合风浪和涌浪的特征波高',
                                             '平均波浪方向', '平均波浪周期'])
    # 航行数据
    new_voyage_data = pd.DataFrame({'航速': speed,
                                    '航向': course})
    # 机舱数据
    new_machine_data = pd.DataFrame(machine_data,
                                    columns=['采样时间', '冷却海水温度', '主机燃油进机温度', '主机1号涡轮增压器转速',
                                             '主机2号涡轮增压器转速', '轴转速', '轴功率'
                                        , '主、辅机燃油流量', '主机扫气压力', '主机扫气箱温度', '主机1号气缸排气温度',
                                             '主机2号气缸排气温度', '主机3号气缸排气温度', '主机4号气缸排气温度',
                                             '主机5号气缸排气温度', '主机6号气缸排气温度',
                                             '主机1号涡轮增压器废气进口温度', '主机2号涡轮增压器废气进口温度',
                                             '主机1号涡轮增压器废气出口温度', '主机2号涡轮增压器废气出口温度',
                                             '辅机燃油流量', '辅机燃油回油流量'])
    # 保存csv文件数据
    total_data = pd.concat([new_voyage_data, new_weather_data, new_machine_data], axis=1)  # 合并数据

    total_data.to_csv(os.path.join(save_path, f'total_data{index}.csv'), index=False, encoding='gbk')

#分别提取出所有的航行数据和天气数据到csv文件（不进行统一时间轴操作）
def extract_voyage_weather(index=3,base_path=None,save_path=None):
    """
    :param index: 原始数据的航次（weather、voyage、machine）
    :param base_path: 原始数据的根目录
    :param save_path:保存total_data的目录
    :return:
    """
    # 时间序列数据
    time_data = loadmat(os.path.join(base_path, 'date_voyage.mat'))['date_v'].reshape(-1)  # (n)
    # 天气数
    weather_data = loadmat(os.path.join(base_path, 'WeatherData.mat'))['data'].T  # 10个类型数据（天气数据） (n,10)
    # 航行数据
    voyage_data = loadmat(os.path.join(base_path, 'VoyageData.mat'))  # 2个类型数据航行数据
    # 机舱数据
    machine_data = loadmat(os.path.join(base_path, f'MachineData_No.{index}.mat'))['Machine'][0][0][-1]  # (m,22)22个类型数据

    # 'data_voyage'(9个特征)，'lat_v', 'lon_v', 'utc_time']
    # 计算得到航向
    course = np.concatenate(voyage_data['data_voyage'][:, 4]).reshape(-1)
    # 还要得到航速
    speed = np.concatenate(voyage_data['data_voyage'][:, 5]).reshape(-1)
    # 得到起止时间
    start_time = machine_data[:, 0][0]
    end_time = machine_data[:, 0][-1]
    # 得到起始索引(最接近的机舱数据时间的索引)
    start_index = np.where(np.isclose(time_data, start_time))[0][0]
    end_index = np.where(np.isclose(time_data, end_time))[0][0]

    #进行提取出所需的天气数据和航行数据
    time_data=time_data[start_index:end_index+1]
    weather_data=np.concatenate([time_data.reshape(-1,1),weather_data[start_index:end_index+1]],axis=1)
    course=course[start_index:end_index+1]
    speed=speed[start_index:end_index+1]

    new_weather_data = pd.DataFrame(weather_data,
                                    columns=['采样时间','海洋表面温度', '海平面大气压', '海平面10m处经度方向风场分量',
                                             '海平面10m处纬度方向风场分量', '海平面2m处空气温度',
                                             '海平面2处露点温度', '海平面粗糙度', '联合风浪和涌浪的特征波高',
                                             '平均波浪方向', '平均波浪周期'])
    # 航行数据
    new_voyage_data = pd.DataFrame({'采样时间':time_data,
                                    '航速': speed,
                                    '航向': course})

    new_weather_data.to_csv(os.path.join(save_path,f'WeatherData_No.{index}.csv'), index=False, encoding='gbk')
    new_voyage_data.to_csv(os.path.join(save_path,f'VoyageData_No.{index}.csv'), index=False, encoding='gbk')


