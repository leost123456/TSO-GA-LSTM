import os
import numpy as np
import pandas as pd
from Universal import plot_curve

#计算油耗率
def fueloil(ma_f,ma_rf,Q_f,t,P_e):
    """
    :param ma_f:辅机进口流量，单位m3/s
    :param ma_rf:辅机回油流量，单位m3/s
    :param Q_f:主辅机进口流量，单位m3/s
    :param t: 生成油耗率的时间间隔,单位min
    :param P_e:轴功率，单位kW
    :return:
    """
    m_f=(np.diff(Q_f)-np.diff(ma_f)+np.diff(ma_rf))/(t*60) #计算出体积流量 单位为m3/s
    m_f=np.concatenate(([m_f[0]],m_f),axis=0) #补充第一个数据
    pho_f20 = 0.84  #密度修正 单位g/ml
    fc = m_f * pho_f20 * 10 ** 3  # 质量流量,单位kg/s
    result = fc * 10 ** 3 * 3600 / P_e  # 燃油消耗率，单位g/kwh

    return result

def data_preproceing(index,base_path,save_path,T=5,plot=False):
    """
    :param index: 航次号
    :param base_path: total_data的根目录
    :param save_path: 保存new_total_data的目录
    :param T: 重采样时间间隔
    :param plot: 是否绘制特征曲线
    :return:
    """
    # 读取文件数据
    total_data = pd.read_csv(os.path.join(base_path, f'total_data{index}.csv'), encoding='gbk')  # 所有数据
    #total_data.fillna(0, inplace=True)  # 将所有缺失值填充为0
    # 将所有小于0的数据转换为0
    total_data[total_data < 0] = 0

    # 缺失值检测
    # print(total_data.isnull().sum())

    # 利用线性插值补足缺失值
    total_data.interpolate(method='linear', inplace=True)

    # 处理采样时间，将其转化为标准格式(注意要减去matlab中的偏移)
    total_data['采样时间'] = pd.to_datetime(total_data['采样时间'] - 719529, unit='D')
    total_data.index = total_data['采样时间']  # 将时间设置为索引

    # 按照时间间隔进行重采样
    T = T  # 重采样时间（min）
    total_data = total_data.resample(f'{T}T').mean()
    total_data = total_data.interpolate(method='linear')  # 插值补全缺失值

    # 计算油耗率指标
    total_data['油耗率'] = fueloil(total_data['辅机燃油流量'], total_data['辅机燃油回油流量'],
                                   total_data['主、辅机燃油流量'], T, total_data['轴功率'])
    total_data['油耗率'].replace([np.inf, -np.inf], 0, inplace=True)  # 将无穷大的数据替换为0
    total_data.loc[total_data['油耗率'] < 0, '油耗率'] = 0  # 将负值也变为0
    total_data['油耗率'].fillna(0, inplace=True)

    # 去除其余无用数据
    total_data.drop(columns=['采样时间', '海洋表面温度', '海平面2m处空气温度', '海平面2处露点温度', '辅机燃油流量',
                             '辅机燃油回油流量', '主、辅机燃油流量'], inplace=True)
    total_columns = total_data.columns  # 数据列名
    total_data.to_csv(os.path.join(save_path,f'new_total_data{index}.csv'), index=True, encoding='gbk')
    # 进行可视化各个特征数据先观察异常值的情况
    if plot:
        y_label_ls=['Speed (kn)','Course（°）','Sea level atmospheric pressure(Pa)','Zonal wind component at 10m above sea level (m/s)','Meridional wind component at 10m above sea level (m/s)','Sea surface roughness','Significant wave height (m)','Mean wave direction (°)',
                    'Mean wave period (s)','Cooling seawater temperature (°F)','Main engine fuel inlet temperature (°F)','Main engine turbocharger No. 1 speed (rpm)','Main engine turbocharger No. 2 speed (rpm)','Shaft speed (Rpm)','Shaft power','Main engine scavenging air pressure','Main engine scavenging air box temperature (°F)',
                    'Main engine cylinder No. 1 exhaust temperature (°F)','Main engine cylinder No. 2 exhaust temperature (°F)','Main engine cylinder No. 3 exhaust temperature (°F)','Main engine cylinder No. 4 exhaust temperature (°F)','Main engine cylinder No. 5 exhaust temperature (°F)','Main engine cylinder No. 6 exhaust temperature (°F)','Main engine turbocharger No. 1 exhaust gas inlet temperature (°F)',
                    'Main engine turbocharger No. 2 exhaust gas inlet temperature (°F)','Main engine turbocharger No. 1 exhaust gas outlet temperature (°F)','Main engine turbocharger No. 2 exhaust gas outlet temperature (°F)','Fuel consumption rate'] #所有特征的单位和名称
        for i,column in enumerate(total_columns):
            plot_curve(total_data.index, total_data[column], xlabel='Date',ylabel=y_label_ls[i],type='curve', periods=5,
                       save_path=rf'./image/特征可视化/{column}{index}.png')





