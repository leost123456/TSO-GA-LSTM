import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from GA import GA
from LSTM import LSTM_model
from Universal import plot_curve,save_data,plot_loss,plot_bar,plot_pred_curve,plot_pred_after,plot_degradation_curve\
    ,mean_absolute_error,mean_square_error,Mean_Absolute_Percentage_Error,computeCorrelation,plot_fit

def prepare_data_mult(input_X,input_y, n_steps): #用于多特征时间序列，且y可以是多维的
    """
    :param input_X: 多维特征矩阵
    :param input_y: 因变量（可多维）
    :param n_steps: 时间步长
    :return:
    """
    X, y = [], []
    for i in range(len(input_X)-int(n_steps)+1): #每个样本，总的构造数据量（n-n_step）
        end_ix = i + int(n_steps) #下一个需要预测的时间序列点
        mult_x, mult_y = input_X[i:end_ix], input_y[end_ix-1] #取出n_step个时间步长和预测的当前时间序列值
        X.append(mult_x)
        y.append(mult_y)
    return np.array(X), np.array(y)

#进行数据预处理操作，返回时间序列数据、标准化后的自变量集和因变量
def data_preproceing(data,X_scaler,param):
    """
    :param data: 输入dataframe数据
    :param scaler: 标准化模型
    :param param: 模型参数
    :return:
    """
    #n=len(data) #样本总数
    #n_features= int(sum(param[:27]))  # 特征数
    data.index = pd.to_datetime(data['采样时间'])
    data.drop(columns=['采样时间'], inplace=True)
    X = X_scaler.transform(data.drop(columns=['航速']).values).reshape(-1,1,total_num_features) #进行标准
    X=X[:,:,param[:total_num_features] == 1] #(n,1,n_features)
    y = data['航速'].values.reshape(-1,1) #(n,1)

    return data.index.tolist(),X,y

if __name__ == '__main__':
    # 数据导入
    data1 = pd.read_csv('./data/new_total_data2.csv', encoding='gbk')
    data2 = pd.read_csv(r'./data/new_total_data3.csv', encoding='gbk')
    draught_data = pd.read_csv(r'./data/ship_info.csv', encoding='gbk')
    model_weights1='./model/two_stage_LSTM2.hdf5'
    model_weights2='./model/two_stage_LSTM3.hdf5'

    #采样时间处理
    draught_data['出发时间'] = pd.to_datetime(draught_data['出发时间'])  # 出发时间
    draught_data['到达时间'] = pd.to_datetime(draught_data['到达时间'])  # 到达时间
    data1.index = pd.to_datetime(data1['采样时间'])
    data1.drop(columns=['采样时间'], inplace=True)
    data2.index = pd.to_datetime(data2['采样时间'])
    data2.drop(columns=['采样时间'], inplace=True)

    # 区分出自变量和因变量
    n1 = len(data1)  # 样本总数
    n2 = len(data2)  # 样本总数
    X1 = data1.drop(columns=['航速'])
    X2 = data2.drop(columns=['航速'])
    y1 = data1['航速']
    y2 = data2['航速']

    # 进行数据标准化（分别利用第二航次和第三航次的数据训练标准器模型)
    X_scaler1 = MinMaxScaler(feature_range=(0, 1))
    X_scaler2 = MinMaxScaler(feature_range=(0, 1))
    y_scaler1 = MinMaxScaler(feature_range=(0, 1))
    y_scaler2 = MinMaxScaler(feature_range=(0, 1))

    norm_X1 = X_scaler1.fit_transform(X1.values)
    norm_X2 = X_scaler2.fit_transform(X2.values)
    norm_y1 = y_scaler1.fit_transform(y1.values.reshape(-1, 1))
    norm_y2 = y_scaler2.fit_transform(y2.values.reshape(-1, 1))
    total_num_features=27 #总特征数

    param1 = np.array([0,1,1,0,1,1,1,1,0,0,0,0,1,1,0,0,0,1,1,0,0,0,1,0,0,0,1,50,7,1,252,72,0.36459363,1])  # 注意这里将drop_out_rate设置为0，这样更优
    # 转换类型
    param1 = param1.tolist()
    param1[:32] = [int(x) for x in param1[:32]]
    param1[32] = float(param1[32])
    param1[33] = int(param1[33])
    param1[28] = 2 ** param1[28]  # batchsize,注意是2的次方
    n_features1=int(sum(param1[:total_num_features])) #特征数

    model1=LSTM_model(None, None, None, None,*param1[27:],n_features1,save_best_path='./model/two_stage_LSTM2.hdf5')
    #导入最佳权重
    model1.model.load_weights(model_weights1)

    param2 = np.array([1,1,1,0,1,1,0,1,1,1,0,0,1,1,1,1,0,1,0,0,0,0,0,0,0,0,1,50,7,1,202,58,0.26774217,1])
    # 转换类型[1,1,1,0,1,1,0,1,1,1,0,0,1,1,1,1,0,1,0,0,0,0,0,0,0,0,1,50,7,1,202,58,0.26774217,1]
    param2 = param2.tolist()
    param2[:32] = [int(x) for x in param2[:32]]
    param2[32] = float(param2[32])
    param2[33] = int(param2[33])
    param2[28] = 2 ** param2[28]  # batchsize,注意是2的次方
    n_features2 = int(sum(param2[:total_num_features]))  # 特征数

    model2 = LSTM_model(None, None, None, None, *param2[27:], n_features2,save_best_path='./model/two_stage_LSTM3.hdf5')
    # 导入最佳权重
    model2.model.load_weights(model_weights2)

    start_voyage=2
    end_voyage=12
    total_data=[pd.read_csv(f'./data/new_total_data{index}.csv',encoding='gbk') for index in range(start_voyage,end_voyage+1)] #所有需要待测数据
    x_scaler = {0: X_scaler1, 1: X_scaler2}
    y_scaler = {0: y_scaler1, 1: y_scaler2}
    param = {0:np.array(param1),1:np.array(param2)}
    model = {0:model1,1:model2}

    #将所有需要测试的数据如果有缺失值即去除
    total_data_after=[data.fillna(method='bfill').fillna(method='ffill')  for data in total_data]

    time_data_ori_ls=[]
    X_data_ls=[]
    y_data_ori_ls=[]
    for i in range(len(total_data_after)):
        time_ls,norm_X_,y_=data_preproceing(total_data_after[i],X_scaler=x_scaler[i%2],param=param[i%2])
        time_data_ori_ls.append(time_ls)
        X_data_ls.append(norm_X_)
        y_data_ori_ls.append(y_.reshape(-1))

    #按照航次数据输入对应的模型中得到预测航次并进行反标准化
    y_pred_ori_ls=[]
    for i in range(len(total_data_after)):
        y_pred=model[i%2].model.predict(X_data_ls[i]) #预测
        y_pred=y_scaler[i%2].inverse_transform(y_pred) #反标准化
        y_pred_ori_ls.append(y_pred.reshape(-1)) #添加

    time_data_ls=time_data_ori_ls #[time_data_ori_ls[i][y_pred_ori_ls[i]>y_data_ori_ls[i]] for i in range(end_voyage-start_voyage+1)]
    y_pred_ls= y_pred_ori_ls #[y_pred_ori_ls[i][y_pred_ori_ls[i]>y_data_ori_ls[i]] for i in range(end_voyage-start_voyage+1)]
    y_data_ls = y_data_ori_ls #[y_data_ori_ls[i][y_pred_ori_ls[i] > y_data_ori_ls[i]] for i in range(end_voyage - start_voyage + 1)]

    #绘制与真实航速的对比线图(分航次进行)（航速残差作为面积图）
    plot_pred_after(4,end_voyage,time_data_ls[2:],y_pred_ls[2:],y_data_ls[2:],xlabel='Date',ylabel='Ship speed (kn)',periods=7,save_path='./image/航速预测对比图（未来）.png')

    ###########进行重采样再进行拟合得到船体及螺旋桨退化曲线##########
    #得到各个航次的实测值与预测值的残差序列
    residual_ls=[y_data_ls[i].reshape(-1)-y_pred_ls[i].reshape(-1) for i in range(len(time_data_ls))]
    #得到残差的dataframe序列（时间为index）
    residual_dataframe_ls=[pd.DataFrame({'residual':residual_ls[i]},index=time_data_ls[i]) for i in range(len(time_data_ls))]
    #进行重采样并线性插值去除缺失值
    T = 4320  # 重采样时间（min）
    resample_residual_dataframe_ls=[residual_dataframe_ls[i].resample(f'{T}T').mean() for i in range(len(time_data_ls))]
    resample_residual_dataframe_ls=[resample_residual_dataframe_ls[i].interpolate(method='linear') for i in range(len(time_data_ls))] #插值补全缺失值

    #绘制船体及螺旋桨退化点，并利用拟合曲线拟合(同时绘制吃水信息，双坐标轴绘制)

    draught_num=[21.3,10.9,21.3,10.85,21.26,10.8,21.48,10.83,21.51] #每个航次的固定吃水(第四航次往后)
    resample_time_ls=[resample_residual_dataframe_ls[i].index.tolist() for i in range(len(time_data_ls))] #时间
    resample_residual_ls=[resample_residual_dataframe_ls[i]['residual'].values.reshape(-1) for i in range(len(time_data_ls))] #残差
    draught_ls = [np.array([draught_num[i]] * len(resample_time_ls[i+2])) for i in range(len(draught_num))]  # 获得每个航次的吃水数据（与时间序列相对应）

    plot_degradation_curve(4,end_voyage,resample_time_ls[2:],resample_residual_ls[2:],draught_data=draught_ls,xlabel='Date',ylabel='Residuel (kn)',periods=7,fit=True,degree=9,save_path='./image/船体及螺旋桨退化曲线.png')
