import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from GA import GA
from LSTM import LSTM_model
from Universal import plot_curve,save_data,plot_loss,plot_bar,plot_pred_curve,mean_absolute_error,mean_square_error,Mean_Absolute_Percentage_Error,computeCorrelation

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
    # 进行LSTM输入数据准备
    norm_X, norm_y = prepare_data_mult(norm_X, norm_y, n_steps=1)

    # 进行区分训练集和测试集
    test_radio = 0.2  # 测试集比例
    X_train, X_test = norm_X[:int((1 - test_radio) * (n - 1 + 1))], norm_X[int((1 - test_radio) * (n - 1 + 1)):]
    y_train, y_test = norm_y[:int((1 - test_radio) * (n - 1 + 1))], norm_y[int((1 - test_radio) * (n  -1 + 1)):]
    total_num_features = 27  # 总特征数

    #############开始构建模型#########
    #1双阶段GA-LSTM
    param1=np.array([1,1,1,0,1,1,0,1,1,1,0,0,1,1,1,1,0,1,0,0,0,0,0,0,0,0,1,50,7,1,202,58,0.26774217,1])
    X_train1,X_test1=X_train[:,:,param1[:total_num_features]==1],X_test[:,:,param1[:total_num_features]==1]
    #转换类型
    param1=param1.tolist()
    param1[:32] = [int(x) for x in param1[:32]]
    param1[32] = float(param1[32])
    param1[33] = int(param1[33])
    param1[28] = 2 ** param1[28]  # batchsize,注意是2的次方
    n_features = X_train1.shape[2]  # 特征数
    #创建模型
    model1 = LSTM_model(X_train1, X_test1, y_train, y_test,*param1[27:],n_features,save_best_path='./model/two_stage_LSTM3.hdf5')
    # 模型训练同时保存最优的模型
    model1.run()

    param2 = np.array([0,1,0,0,1,1,1,0,1,0,1,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,50,5,1,224,64,0.2,1])
    X_train2, X_test2 = X_train[:, :, param2[:total_num_features] == 1], X_test[:, :, param2[:total_num_features] == 1]
    # 转换类型
    param2 = param2.tolist()
    param2[:32] = [int(x) for x in param2[:32]]
    param2[32] = float(param2[32])
    param2[33] = int(param2[33])
    param2[28] = 2 ** param2[28]  # batchsize,注意是2的次方
    n_features = X_train2.shape[2]  # 特征数
    # 创建模型
    model2 = LSTM_model(X_train2, X_test2, y_train, y_test, *param2[27:], n_features,
                        save_best_path='./model/first_stage_LSTM3.hdf5')
    # 模型训练同时保存最优的模型
    #model2.run()

    param3 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,50,7,1,226,83,0.108120672,1])  
    X_train3, X_test3 = X_train[:, :, param3[:total_num_features] == 1], X_test[:, :, param3[:total_num_features] == 1]
    # 转换类型
    param3 = param3.tolist()
    param3[:32] = [int(x) for x in param3[:32]]
    param3[32] = float(param3[32])
    param3[33] = int(param3[33])
    param3[28] = 2 ** param3[28]  # batchsize,注意是2的次方
    n_features = X_train3.shape[2]  # 特征数
    #创建模型
    model3 = LSTM_model(X_train3, X_test3, y_train, y_test, *param3[27:], n_features,
                        save_best_path='./model/second_stage_LSTM3.hdf5')
    # 模型训练同时保存最优的模型
    model3.run()

    # 4LSTM模型
    model4 = LSTM_model(X_train, X_test, y_train, y_test, epoch=50, batchsize=32, n_layer=1, hidden_unit1=224,
                        hidden_unit2=64, dropout_rate=0.2, n_steps=1, n_feature=27, save_best_path='./model/LSTM3.hdf5')
    # 模型训练同时保存最优的模型
    model4.run()

    loss_data={'TSO-GA-LSTM':model1.history.history['loss'],'FSO-GA-LSTM':model2.history.history['loss'],
               'SSO-GA-LSTM':model3.history.history['loss'],'LSTM':model4.history.history['loss']}
    plot_loss(X=[np.arange(1,51),np.arange(1,51),np.arange(1,51),np.arange(1,51)],
              Y=[loss_data['TSO-GA-LSTM'],loss_data['FSO-GA-LSTM'],loss_data['SSO-GA-LSTM'],loss_data['LSTM']],
              label=['TSO-GA-LSTM','FSO-GA-LSTM','SSO-GA-LSTM','LSTM'],xlabel='Epoch',ylabel='Loss function',save_path='./image/loss_curve.svg'
              )
    save_data(epoch=np.arange(1,51),**loss_data,save_path='./data/train_loss_compare.csv')

    loss_data=pd.read_csv('./data/train_loss_compare.csv')
    plot_loss(X=[np.arange(1, 51), np.arange(1, 51), np.arange(1, 51), np.arange(1, 51)],
              Y=[loss_data['TSO-GA-LSTM'], loss_data['FSO-GA-LSTM'], loss_data['SSO-GA-LSTM'], loss_data['LSTM']],
              label=['TSO-GA-LSTM', 'FSO-GA-LSTM', 'SSO-GA-LSTM', 'LSTM'], xlabel='Epoch', ylabel='Loss function',
              save_path='./image/loss_curve.svg'
              )

    #下面得到各个模型的测试集预测值，并进行反标准化，绘制预测值和观测值对比图（子图放大）同时进行保存数据
    #导入各模型最佳权重
    model1.model.load_weights('./model/two_stage_LSTM3.hdf5')
    model2.model.load_weights('./model/first_stage_LSTM3.hdf5')
    model3.model.load_weights('./model/second_stage_LSTM3.hdf5')
    model4.model.load_weights('./model/LSTM3.hdf5')
    #得到测试集预测值(同时逆标准化)
    y_pred1= y_scaler.inverse_transform(model1.model.predict(np.concatenate([X_train1,X_test1],axis=0))).reshape(-1)
    y_pred2 = y_scaler.inverse_transform(model2.model.predict(np.concatenate([X_train2,X_test2],axis=0))).reshape(-1)
    y_pred3 = y_scaler.inverse_transform(model3.model.predict(np.concatenate([X_train3,X_test3],axis=0))).reshape(-1)
    y_pred4 = y_scaler.inverse_transform(model4.model.predict(np.concatenate([X_train,X_test],axis=0))).reshape(-1)
    y_true=y_scaler.inverse_transform(norm_y).reshape(-1)

    #进行绘制曲线放大子图（注意横轴是日期）
    x_date=data.index
    # 只绘制TSO-GA-LSTM与observation进行对比的曲线(第二航次同理)
    plot_pred_curve(x_date, [y_pred1, y_true],
                    label=['TSO-GA-LSTM', 'Observation'], xlabel='Date',
                    ylabel='Ship speed (kn)',
                    color=['#F27970', '#FF1F56'],
                    linestyle=['solid','solid'],
                    save_path='./image/pred_curve(voyage3).png')
    #绘制四个模型对比图像
    """plot_pred_curve(x_date,[y_pred1,y_pred2,y_pred3,y_pred4,y_true],label=['TSO-GA-LSTM','FSO-GA-LSTM','SSO-GA-LSTM','LSTM','Observation'],xlabel='Date',ylabel='Ship speed (kn)',
                    color=['#F27970','#FFBE7A','#BECFC9','#82B0D2','#FF1F56'],linestyle=['solid', 'dotted', 'dashed', 'dashdot','solid'],save_path='./image/pred_curve_self_compare.png')
"""
    #保存数据（所有）
    save_data(**{'date':x_date,'TSO-GA-LSTM':y_pred1,'FSO-GA-LSTM':y_pred2,'SSO-GA-LSTM':y_pred3,'LSTM':y_pred4,'Observation':y_true,'save_path':'./data/pred_data_self.csv'})

    #下面利用MAE、MSE、MAPE、R方四个指标进行评价四个模型，输出测试集最终结果
    #测试集起始序号位置
    test_start_index = int((1 - test_radio) * n)
    #得到各模型测试集预测数据
    y_pred1_test=y_pred1[test_start_index:]
    y_pred2_test = y_pred2[test_start_index:]
    y_pred3_test = y_pred3[test_start_index:]
    y_pred4_test = y_pred4[test_start_index:]
    y_true_test=y_true[test_start_index:] #观测真实值

    #计算指标
    mae_ls=[mean_absolute_error(y,y_true_test) for y in [y_pred1_test,y_pred2_test,y_pred3_test,y_pred4_test]]
    mse_ls=[mean_square_error(y,y_true_test) for y in [y_pred1_test,y_pred2_test,y_pred3_test,y_pred4_test]]
    mape_ls=[Mean_Absolute_Percentage_Error(y,y_true_test) for y in [y_pred1_test,y_pred2_test,y_pred3_test,y_pred4_test]]
    r_square_ls=[computeCorrelation(y,y_true_test)**2 for y in [y_pred1_test,y_pred2_test,y_pred3_test,y_pred4_test]]
    print('mae为:',mae_ls)
    print('mse为:',mse_ls)
    print('mape为:',mape_ls)
    print('r方为:',r_square_ls)
    #绘制相关图像
    plot_bar(x=['TSO-GA-LSTM','FSO-GA-LSTM','SSO-GA-LSTM','LSTM'],y=mae_ls,xlabel='Model',ylabel='MAE',color='#E18E6D',save_path='./image/MAE_compare_self.svg')
    plot_bar(x=['TSO-GA-LSTM','FSO-GA-LSTM','SSO-GA-LSTM','LSTM'],y=mse_ls,xlabel='Model',ylabel='MSE',color='#E18E6D',save_path='./image/MSE_compare_self.svg')
    plot_bar(x=['TSO-GA-LSTM','FSO-GA-LSTM','SSO-GA-LSTM','LSTM'],y=mape_ls,xlabel='Model',ylabel='MAPE',color='#E18E6D',save_path='./image/MAPE_compare_self.svg')
    plot_bar(x=['TSO-GA-LSTM','FSO-GA-LSTM','SSO-GA-LSTM','LSTM'],y=r_square_ls,xlabel='Model',ylabel='R square',color='#E18E6D',save_path='./image/R_square_compare_self.svg')




