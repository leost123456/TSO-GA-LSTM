import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import joblib

from LSTM import LSTM_model,GRU_model,BPNN
from Universal import plot_curve,save_data,plot_loss,plot_bar,plot_pred_curve,mean_absolute_error,mean_square_error,Mean_Absolute_Percentage_Error,computeCorrelation
from 数据预处理 import data_preproceing

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

if __name__ == '__main__':
    # 数据导入
    data = pd.read_csv(r'./data/new_total_data3.csv', encoding='gbk')
    data.index = pd.to_datetime(data['采样时间'])
    data.drop(columns=['采样时间'], inplace=True)

    # 区分出自变量和因变量
    n = len(data)  # 样本总数
    X = data.drop(columns=['航速'])
    y = data['航速']

    # 进行数据标准化
    X_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    norm_X = X_scaler.fit_transform(X.values)
    norm_y = y_scaler.fit_transform(y.values.reshape(-1, 1))
    # 用于GRU和TSO-GA-LSTM模型输入数据准备
    trans_norm_X, trans_norm_y = prepare_data_mult(norm_X, norm_y, n_steps=1) #(n,1,n_features),(n,1)

    # 进行区分训练集和测试集（其余模型）
    test_radio = 0.2  # 测试集比例
    X_train, X_test = norm_X[:int((1 - test_radio) * (n - 1 + 1))], norm_X[int((1 - test_radio) * (n - 1 + 1)):] #(n,27)
    y_train, y_test = norm_y[:int((1 - test_radio) * (n - 1 + 1))], norm_y[int((1 - test_radio) * (n - 1 + 1)):] #(n,1)

    #用于GRU和TSO-GA-LSTM模型输入数据（训练集和测试集）
    X_train_trans, X_test_trans = trans_norm_X[:int((1 - test_radio) * (n - 1 + 1))], trans_norm_X[int((1 - test_radio) * (n - 1 + 1)):]
    y_train_trans, y_test_trans = trans_norm_y[:int((1 - test_radio) * (n - 1 + 1))], trans_norm_y[int((1 - test_radio) * (n - 1 + 1)):]
    total_num_features = 27  # 总特征数

    ###########进行各模型的创建与训练#############
    #1进行TSO-GA-LSTM模型的预测
    param1 = np.array([1,1,1,0,1,1,0,1,1,1,0,0,1,1,1,1,0,1,0,0,0,0,0,0,0,0,1,50,7,1,202,58,0.26774217,1])
    X_train1, X_test1 = X_train_trans[:, :, param1[:total_num_features] == 1], X_test_trans[:, :, param1[:total_num_features] == 1]
    # 转换类型
    param1 = param1.tolist()
    param1[:32] = [int(x) for x in param1[:32]]
    param1[32] = float(param1[32])
    param1[33] = int(param1[33])
    param1[28] = 2 ** param1[28]  # batchsize,注意是2的次方
    n_features = X_train1.shape[2]  # 特征数
    # 创建模型并加载最优模型
    model1 = LSTM_model(X_train1, X_test1, y_train_trans, y_test_trans, *param1[27:], n_features,save_best_path='./model/two_stage_LSTM3.hdf5')

    #2GRU模型(需要用维度转化后的作为输入)
    model2 = GRU_model(X_train_trans, X_test_trans, y_train_trans, y_test_trans, epoch=50, batchsize=32, n_layer=1, hidden_unit1=224,
                        hidden_unit2=64, dropout_rate=0.2, n_steps=1, n_feature=27, save_best_path='./model/GRU3.hdf5')
    # 模型训练同时保存最优的模型
    model2.run()

    #3BPNN（BP神经网络模型）
    model3=BPNN(X_train,X_test,y_train,y_test,epoch=50,batchsize=32,dropout_rate=0.1,n_feature=27, save_best_path='./model/BPNN3.hdf5')
    model3.run()

    #4随机森林算法（RF）
    model4 = RandomForestRegressor()
    model4.fit(X_train, y_train)
    joblib.dump(model4, './model/RF3.pkl') #保存模型

    #5XGboost
    model5=XGBRegressor()
    model5.fit(X_train, y_train)
    joblib.dump(model5, './model/XGboost3.pkl')  # 保存模型

    #6Adaboost
    model6=AdaBoostRegressor()
    model6.fit(X_train, y_train)
    joblib.dump(model6, './model/Adaboost3.pkl')  # 保存模型

    #7SVM
    model7=SVR()
    model7.fit(X_train, y_train)
    joblib.dump(model7, './model/SVM3.pkl')  # 保存模型

    ################进行预测#################
    #导入最佳模型权重
    model1.model.load_weights('./model/two_stage_LSTM3.hdf5')
    model2.model.load_weights('./model/GRU3.hdf5')
    model3.model.load_weights('./model/BPNN3.hdf5')
    model4=joblib.load('./model/RF3.pkl')
    model5=joblib.load('./model/XGboost3.pkl')
    model6=joblib.load('./model/Adaboost3.pkl')
    model7=joblib.load('./model/SVM3.pkl')

    # 进行预测得到全部数据(fan)
    y_pred1 = y_scaler.inverse_transform(model1.model.predict(np.concatenate([X_train1,X_test1],axis=0))).reshape(-1)
    y_pred2 = y_scaler.inverse_transform(model2.model.predict(trans_norm_X)).reshape(-1)
    y_pred3 = y_scaler.inverse_transform(model3.model.predict(norm_X)).reshape(-1)
    y_pred4 = y_scaler.inverse_transform(model4.predict(norm_X).reshape(-1,1)).reshape(-1)
    y_pred5 = y_scaler.inverse_transform(model5.predict(norm_X).reshape(-1,1)).reshape(-1)
    y_pred6 = y_scaler.inverse_transform(model6.predict(norm_X).reshape(-1,1)).reshape(-1)
    y_pred7 = y_scaler.inverse_transform(model7.predict(norm_X).reshape(-1,1)).reshape(-1)
    y_true = y_scaler.inverse_transform(norm_y.reshape(-1,1)).reshape(-1)

    #绘制预测曲线图（子图放大）
    x_date = data.index
    plot_pred_curve(x_date,[y_pred1,y_pred2,y_pred3,y_pred4,y_pred5,y_pred6,y_pred7,y_true],label=['Proposed model','GRU','BPNN','RF','XGBoost','AdaBoost','SVM','Observation'],
                    xlabel='Date',ylabel='Ship speed (kn)',save_path='./image/pred_curve_other_compare.png')
    # 保存数据（所有）
    save_data(**{'date': x_date, 'Proposed model': y_pred1, 'GRU': y_pred2, 'BPNN': y_pred3, 'RF': y_pred4,'XGBoost': y_pred5,'AdaBoost': y_pred6,'SVM': y_pred7,
           'Observation': y_true, 'save_path': './data/pred_data_other.csv'})

    # 下面利用MAE、MSE、MAPE、R方四个指标进行评价四个模型，输出测试集最终结果
    # 测试集起始序号位置
    test_start_index = int((1 - test_radio) * n)
    # 得到各模型测试集预测数据
    y_pred1_test = y_pred1[test_start_index:]
    y_pred2_test = y_pred2[test_start_index:]
    y_pred3_test = y_pred3[test_start_index:]
    y_pred4_test = y_pred4[test_start_index:]
    y_pred5_test = y_pred5[test_start_index:]
    y_pred6_test = y_pred6[test_start_index:]
    y_pred7_test = y_pred7[test_start_index:]
    y_true_test = y_true[test_start_index:]  # 观测真实值

    # 计算指标
    mae_ls = [mean_absolute_error(y, y_true_test) for y in [y_pred1_test, y_pred2_test, y_pred3_test, y_pred4_test,y_pred5_test,y_pred6_test,y_pred7_test]]
    mse_ls = [mean_square_error(y, y_true_test) for y in [y_pred1_test, y_pred2_test, y_pred3_test, y_pred4_test,y_pred5_test,y_pred6_test,y_pred7_test]]
    mape_ls = [Mean_Absolute_Percentage_Error(y, y_true_test) for y in [y_pred1_test, y_pred2_test, y_pred3_test, y_pred4_test,y_pred5_test,y_pred6_test,y_pred7_test]]
    r_square_ls = [computeCorrelation(y, y_true_test) ** 2 for y in [y_pred1_test, y_pred2_test, y_pred3_test, y_pred4_test,y_pred5_test,y_pred6_test,y_pred7_test]]
    print('mae为:', mae_ls)
    print('mse为:', mse_ls)
    print('mape为:', mape_ls)
    print('r方为:', r_square_ls)
    # 绘制相关图像
    plot_bar(x=['Proposed model','GRU','BPNN','RF','XGBoost','AdaBoost','SVM'], y=mae_ls, xlabel='Model', ylabel='MAE',color='#62B197',
             save_path='./image/MAE_compare_other.svg')
    plot_bar(x=['Proposed model','GRU','BPNN','RF','XGBoost','AdaBoost','SVM'], y=mse_ls, xlabel='Model', ylabel='MSE',color='#62B197',
             save_path='./image/MSE_compare_other.svg')
    plot_bar(x=['Proposed model','GRU','BPNN','RF','XGBoost','AdaBoost','SVM'], y=mape_ls, xlabel='Model', ylabel='MAPE',color='#62B197',
             save_path='./image/MAPE_compare_other.svg')
    plot_bar(x=['Proposed model','GRU','BPNN','RF','XGBoost','AdaBoost','SVM'], y=r_square_ls, xlabel='Model', ylabel='R square',color='#62B197',
             save_path='./image/R_square_compare_other.svg')






