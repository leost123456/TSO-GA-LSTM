import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from GA import GA
from LSTM import LSTM_model
from Universal import plot_curve,save_data

#读取数据
data=pd.read_csv(r'./data/new_total_data2.csv',encoding='gbk')
data.index=data['采样时间']
data.drop(columns=['采样时间'],inplace=True)

#区分出自变量和因变量
n=len(data) #样本综述
X = data.drop(columns=['航速'])
y = data['航速']

#进行数据标准化
X_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))
norm_X = X_scaler.fit_transform(X.values)
norm_y = y_scaler.fit_transform(y.values.reshape(-1,1))

#数据准备
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

#设置目标函数
def f(param,X=norm_X,y=norm_y):
    """
    :param param: 参数序列，其中从0-26标号为特征选择（0，1），27为epoch（50-200）,28为batchsize(2^0-2^7),
    29为n_layer(1-4),30为hidden_unit1(150-300),31为hidden_unit2(50-100),32为dropout_rate(0.1-0.5),33为n_steps(1-20)
    :param X: 特征集（n,n_feature）
    :param y: 航速(n）
    :return: 返回最优目标函数值（MSE）
    """
    #进行特征选择
    X=X[:,param[:27]==1] #注意重新赋值，不会影响到外部的X
    n_features=X.shape[1] #特征数
    #转换类型
    param = param.tolist()
    param[:32] = [int(x) for x in param[:32]]
    param[32] = float(param[32])
    param[33] = int(param[33])

    #进行LSTM输入数据准备
    X,y=prepare_data_mult(X,y,n_steps=param[33])#注意重新赋值，不会影响到外部的X

    # 进行区分训练集和测试集
    test_radio = 0.2  # 测试集比例
    X_train, X_test = X[:int((1 - test_radio) * (n - param[-1]+1))], X[int((1 - test_radio) * (n - param[-1]+1)):]
    y_train, y_test = y[:int((1 - test_radio) * (n - param[-1]+1))], y[int((1 - test_radio) * (n - param[-1]+1)):]

    #创建模型并训练
    param[28]=2**param[28] #batchsize,注意是2的次方
    model=LSTM_model(X_train,X_test,y_train,y_test,*param[27:],n_features,save_best_path=None)
    model.run()

    return model.best_obj()

if __name__ == '__main__':
    total_num_features=27 #总特征数
    param_bound=[[0,1]]*total_num_features #特征选择值域范围（27个）
    epoch_bound=[[20,20]] #epoch范围
    batchsize_bound=[[0,7]] #batchsize范围（注意是2的次方）
    n_layer_bound=[[1,1]] #n_layer范围(1-4)
    hidden_unit1_bound=[[150,300]] #hidden_unit1范围
    hidden_unit2_bound=[[50,100]] #hidden_unit2范围
    dropout_rate_bound=[[0.1,0.5]] #dropout_rate范围
    n_steps_bound=[[1,1]] #n_steps范围(1-20)
    varbound = np.array(param_bound+epoch_bound+batchsize_bound+n_layer_bound+hidden_unit1_bound+hidden_unit2_bound+dropout_rate_bound+n_steps_bound)  # 变量值域范围
    vartype = np.array([['int']] * 32 +[['real']]+[['int']] )  # 变量类型(34个决策变量)

    # 默认是最小化目标函数
    model = GA(function=f,  # 目标函数
               dimension=34,  # 决策变量数量(总共34个)
               variable_type=vartype,  # 变量类型（序列形式，real,int）
               variable_boundaries=varbound,  # 各变量值s域
               eq_constraints=  None,  # 等式约束函数
               ineq_constraints=None,  # 不等式约束函数
               function_timeout=100000, # 延时（函数没响应）
               eq_cons_coefficient=0.1,  # 等式约束系数（值越小，等式约束越严格）
               eq_optimezer='soft',
               max_num_iteration=20,  # 最大迭代次数
               population_size=30,  # 个体数量（初始解的数量）
               penalty_factor=10,  # 惩罚因子（用于约束条件）,越大约束作用越大（要选择合适的值1比较合适）
               mutation_probability=0.1,  # 变异概率
               elit_ratio=0.1,  # 精英选择的比例（每轮都保留一定数量的较优目标函数的个体）
               crossover_probability=0.5,  # 交叉概率
               parents_portion=0.3,  # 父代比例(用于每代的交叉和变异)
               crossover_type='uniform',  # 交叉类型（”one_point“ ”two_point“ "uniform"）
               max_iteration_without_improv=None,  # 多少轮没更新后退出
               convergence_curve=True,  # 是否绘制算法迭代收敛曲线
               progress_bar=True,  # 进度条显示
               plot_path=r'./image/Genetic Algorithm(two stage GA-LSTM3).svg'  # 保存收敛曲线svg图像的完整路径
               )
    model.run()
    # 输出信息
    best_variable = model.best_variable  # 最优解
    best_function = model.best_function  # 最优目标函数
    report = model.report  # 每轮的最优目标函数
    #保存最优函数迭代值
    save_data(**{'iteration':np.arange(1,len(report)+1),'value':report,'save_path':'./data/Genetic Algorithm(two stage GA-LSTM3).csv'})





