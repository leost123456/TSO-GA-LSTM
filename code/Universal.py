import os
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import ConnectionPatch
from matplotlib import rcParams #导入包
import tensorflow as tf

config = {"font.family":'Times New Roman'}  # 设置字体类型
rcParams.update(config)    #进行更新配置

#绘制输入数据的曲线,用于初始特征可视化操作（jpg图像）
def plot_curve(x,y,xlabel='',ylabel='',type='scatter',color='#D95319',periods=None,save_path=None):
    plt.figure(figsize=(10, 6))
    plt.tick_params(size=5, labelsize=13)  # 坐标轴
    plt.grid(alpha=0.3)  # 是否加网格线
    if type=='scatter':
        plt.scatter(x,y,marker='o',color=color,s=1,alpha=1)
    elif type=='curve':
        plt.plot(x,y,color='#D95319',lw=1.5,alpha=1)
    plt.xlabel(xlabel, fontsize=13, family='Times New Roman')  # 注意加粗目前英文才可以使用（中文需要变换字体）
    plt.ylabel(ylabel, fontsize=13, family='Times New Roman')
    #plt.title(title, fontsize=15, family='SimSun')
    #调整x轴标签范围(对于时间序列数据来说)
    if periods:
        plt.xticks(pd.date_range(x[0],x[-1],periods=periods))
    if save_path:
        plt.savefig(save_path,format='png', bbox_inches='tight',dpi=1000) #注意需要完成的路径
    plt.show()

#可以用于绘制单排柱形图(svg图片)
def plot_bar(x,y,xlabel,ylabel,width=0.35,color='#e74c3c',save_path=None):
    """
   :param x: 标签序列
   :param y: 数字序列
   :param xlabel: x轴标签
   :param ylabel: y轴标签
   :param title: 标题
   :param width: 宽度
   :param color: 颜色
   :param save_path: 保存路径(完整)
   :return:
   """
    index=np.arange(len(x)) #序号列表
    plt.figure(figsize=(8,6))
    plt.tick_params(size=5,labelsize = 13) #坐标轴
    plt.grid(alpha=0.3)                    #是否加网格线

    #注意下面可以进行绘制误差线，如果是计算均值那种的簇状柱形图的话(注意类别多的话可以用循环的方式搞)
    plt.bar(index,y,width=width,color=color)

    #下面进行打上标签(也可以用循环的方式进行绘制)(颜色就存储在一个列表中)
    for i,count in enumerate(y):
        plt.text(index[i],count+0.00000001,round(count,5),horizontalalignment='center',fontsize=13)

    plt.xlabel(xlabel, fontsize=13,family='Times New Roman')
    plt.ylabel(ylabel, fontsize=13,family='Times New Roman')
    plt.legend()
    plt.xticks(index,x)
    plt.savefig(os.path.join(save_path),format='svg',bbox_inches='tight')
    plt.show()
def save_data(**kwargs):
    """
    :param kwargs:{name1:[1,2,...],name2:[1,2,...]} 字典形式的数据，其中可以包含save_path参数，表示存储csv文件的地址
    :return: csv文件
    """
    save_path=kwargs['save_path']
    del kwargs['save_path'] #删除保存路径
    data=pd.DataFrame(kwargs)
    data.to_csv(save_path,index=False,encoding='gbk')

#绘制遗传算法曲线(svg图像)，
def plot_GA_curve(X,Y,label,xlabel,ylabel,save_path=None):
    """
    :param X: x轴数据，形式为[[],[]]
    :param Y: y轴数据，形式为[[],[]]
    :param label: 标签，形式为[]
    :param xlabel: x轴标签
    :param ylabel: y轴标签
    :param save_path: 保存的完整路径地址
    :return:
    """
    label=[None]*len(X) if label==None else label
    color = ['#F27970','#BB9727','#54B345','#32B897']  # 颜色序列
    linestyle = ['solid', 'dotted', 'dashed', 'dashdot', 'dotted', 'dashed', 'dashdot']  # 线形序列
    plt.figure(figsize=(8, 6))
    plt.tick_params(size=5, labelsize=13)  # 坐标轴
    plt.grid(alpha=0.3)  # 是否加网格线
    for i in range(len(X)):
        plt.plot(X[i], Y[i], color=color[i],ls=linestyle[i],lw=1.5,label=label[i])
    plt.legend()
    plt.xlabel(xlabel, fontsize=13,family='Times New Roman')
    plt.ylabel(ylabel, fontsize=13,family='Times New Roman')
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')  # 注意需要完成的路径
    plt.show()

#绘制四个模型的loss曲线他
def plot_loss(X,Y,label,xlabel,ylabel,save_path=None):
    """
    :param X: x轴数据，形式为[[],[]]
    :param Y: y轴数据，形式为[[],[]]
    :param label: 标签，形式为[]
    :param xlabel: x轴标签
    :param ylabel: y轴标签
    :param save_path: 保存的完整路径地址
    :return:
    """
    color =['#F27970','#BB9727','#54B345','#32B897','#05B9E2','#8983BF','#C76DA2', '#fe0000'] # 颜色序列
    linestyle = ['solid', 'dotted', 'dashed', 'dashdot', 'dotted', 'dashed', 'dashdot','solid']  # 线形序列
    plt.figure(figsize=(8, 6))
    plt.tick_params(size=5, labelsize=13)  # 坐标轴
    plt.grid(alpha=0.3)  # 是否加网格线
    for i in range(len(X)):
        plt.plot(X[i], Y[i], color=color[i], ls=linestyle[i], lw=1.5,label=label[i])
    plt.legend()
    plt.xlabel(xlabel, fontsize=13, family='Times New Roman')
    plt.ylabel(ylabel, fontsize=13, family='Times New Roman')
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')  # 注意需要完成的路径
    plt.show()

#绘制模型预测值和观测值曲线（子图放大,jpg图像）
def plot_pred_curve(x,Y,label,xlabel,ylabel,color=['#F27970','#BB9727','#54B345','#32B897','#05B9E2','#8983BF','#C76DA2', '#FF1F56'] ,
                    linestyle=['solid', 'dotted', 'dashed', 'dashdot', 'dotted', 'dashed', 'dashdot','solid'],save_path=None):
    """
    :param x: x轴数据(日期数据)
    :param Y: y轴数据，形式为[[],[]]
    :param label: 标签，形式为[]
    :param xlabel: x轴标签
    :param ylabel: y轴标签
    :param color: 颜色序列
    :param save_path: 保存的完整路径地址
    :return:
    """
    color = color  # 颜色序列
    linestyle = linestyle # 线形序列
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    axins = ax.inset_axes((0.4, 0.14, 0.3, 0.2))
    ax.tick_params(size=5, labelsize=13)  # 坐标轴
    ax.grid(alpha=0.3)  # 是否加网格线·
    axins.tick_params(size=5, labelsize=13)  # 坐标轴
    axins.grid(alpha=0.3)  # 是否加网格线·

    for i in range(len(Y)):
        ax.plot(x, Y[i], color=color[i], ls=linestyle[i], lw=1, label=label[i])
        axins.plot(x, Y[i], color=color[i], ls=linestyle[i], lw=1)

    # 下面进行设置x轴的显示范围（如果知道范围的话可以直接设置数字）
    xlim0 = x[7500]
    xlim1 = x[-1400]

    # 下面计算子图y轴的显示范围（如果知道范围的话可以直接设置数字）s
    ylim0 =11.5 #8.1
    ylim1 =14.8 #=13

    # 下面进行设置x轴和y轴的显示范围（可以直接设置）
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)
    axins.set_xticks([xlim0,xlim1])  # 设置子图的x轴标签,要让其隐藏就直接写[]

    # 原图中画方框
    tx0 = xlim0
    tx1 = xlim1
    ty0 = ylim0
    ty1 = ylim1
    # 绘制方框
    sx = [tx0, tx1, tx1, tx0, tx0]  # 方框x坐标
    sy = [ty0, ty0, ty1, ty1, ty0]  # 方框y坐标
    ax.plot(sx, sy, "black")
    # 画两条线(要注意方向)
    xy = (xlim0, ylim1)  # 子图方框左上点（也就是较大的在外面的框）
    xy2 = (xlim0, ylim0)  # 原图方框左下点（也就是较小的标记数据的那个小框）
    con = ConnectionPatch(xyA=xy, xyB=xy2, coordsA="data", coordsB="data",  # 注意是xyA连接xyB（方向也就是子图连接原图）
                          axesA=axins, axesB=ax)
    axins.add_artist(con)
    xy = (xlim1, ylim1)  # 子图方框右上点
    xy2 = (xlim1, ylim0)  # 原图方框右下点
    con = ConnectionPatch(xyA=xy, xyB=xy2, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax)
    axins.add_artist(con)

    #添加分割线
    ax.axvline(x=x[int(0.8 * (len(x) - 1 + 1))-1], color='black',lw=2,linestyle=':')
    axins.axvline(x=x[int(0.8 * (len(x) - 1 + 1)) - 1], color='black', lw=2, linestyle=':')
    #添加标签
    plt.text(x=x[6800], y=15, s='Train', color='black',fontsize=15)
    plt.text(x=x[-1500], y=15, s='Test', color='black',fontsize=15)

    ax.legend(loc='best')
    ax.set_xticks(pd.date_range(x[0],x[-1],periods=5)) #x轴日期bi标签
    axins.set_ylim(ylim0, ylim1)
    ax.set_xlabel(xlabel, fontsize=13, family='Times new roman')
    ax.set_ylabel(ylabel, fontsize=13, family='Times new roman')

    if save_path:
        plt.savefig(save_path,format='png', bbox_inches='tight',dpi=1000)  # 注意需要完成的路径
    plt.show()

#进行绘制第三航次以后的预测曲线和真实值曲线的对比（同时利用速度残差作为面积）
def plot_pred_after(start_index,end_index,time_ls,y_pred_ls,y_true_ls,xlabel='',ylabel='',periods=5,save_path=None):
    """
    :param start_index: 开始航次
    :param end_index: 结束航次
    :param time_ls: 时间列表，形式为[[],[]]
    :param y_pred_ls: 预测数据列表，形式为p[array([]),array([])]
    :param y_true_ls: 真实值数据列表，形式为[array([]),array([])]
    :param xlabel: x轴标签
    :param ylabel: y轴标签
    :param periods: 显示的时间段
    :param save_path: 保存具体路径
    :return:
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(30, 6))
    ax1.tick_params(size=5, labelsize=13)  # 坐标轴
    ax1.grid(alpha=0.3)  # 是否加网格线·

    #绘制预测曲线
    for i in range(len(time_ls)):
        if i ==0:
            ax1.plot(time_ls[i], y_pred_ls[i], color='#62B197', lw=1.3, alpha=1,label='Prediction')
        else:
            ax1.plot(time_ls[i], y_pred_ls[i], color='#62B197', lw=1.3, alpha=1)
    #绘制真实值曲线
    for i in range(len(time_ls)):
        if i==0:
            ax1.plot(time_ls[i], y_true_ls[i], color='#E18E6D', lw=1.3, alpha=1,label='Observation')
        else:
            ax1.plot(time_ls[i], y_true_ls[i], color='#E18E6D', lw=1.3, alpha=1)

    #绘制残差
    for i in range(len(time_ls)):
        error = y_true_ls[i].reshape(-1) - y_pred_ls[i].reshape(-1)
        if i==0:
            ax1.fill_between(time_ls[i], error.reshape(-1), color='#E18E6D', alpha=0.5, label='Residual')
        else:
            ax1.fill_between(time_ls[i], error.reshape(-1), color='#E18E6D', alpha=0.5)

    #绘制分界线
    for i in range(len(time_ls)):
        plt.axvline(x=time_ls[i][0], color='black', lw=1.8, linestyle=':') #'#bf0000'
        plt.axvline(x=time_ls[i][-1], color='black', lw=1.8, linestyle=':')

    #写上文字标签
    #text=['Fourth voyage','Fifth voyage','Sixth voyage','Seventh voyage','Eighth voyage','Ninth voyage','tenth voyage','Eleventh voyage','Twelfth voyage']
    text=np.arange(start_index,end_index+1)-1
    for i in range(len(time_ls)):
        plt.text(x=time_ls[i][int(len(time_ls[i])//2.65)], y=15.4, s=f'{text[i]}', color='black',fontsize=13)

    ax1.legend(loc='lower left')
    ax1.set_ylim(-10,17)
    ax1.set_xlabel(xlabel, fontsize=13, family='Times New Roman')  # 注意加粗目前英文才可以使用（中文需要变换字体）
    ax1.set_ylabel(ylabel, fontsize=13, family='Times New Roman')
    # 调整x轴标签范围(对于时间序列数据来说)
    if periods:
        plt.xticks(pd.date_range(time_ls[0][0], time_ls[-1][-1], periods=periods))
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=1000)  #注意需要完成的路径
    plt.show()

#绘制船舶及螺旋桨船速退化点，并拟合退化曲线
def plot_degradation_curve(start_index,end_index,time_ls,residuel_ls,draught_data,xlabel='',ylabel='',periods=7,fit=False,degree=4,save_path=None):
    """
    :param start_index: 开始航次
    :param end_index: 结束航次
    :param time_ls: 时间序列列表，形式为[[],[]]
    :param residuel_ls: 残差序列列表，形式为[array([]),array([])]
    :param draught_data: 吃水数据，形式dataframe的数据形式
    :param xlabel: x轴标签
    :param ylabel: y轴标签
    :param periods: 显示的时间段
    :param fit: 是否进行拟合
    :param save_path: 保存的完整路径地址
    :return:
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(20, 6))
    ax1.tick_params(axis='y', labelcolor='#bf0000',labelsize=12)  #坐标轴
    ax1.grid(alpha=0.3)  #是否加网格线

    # 绘制船速残差点
    for i in range(len(time_ls)):
        if i == 0:
            ax1.scatter(time_ls[i], residuel_ls[i], color='#E18E6D',s=10,alpha=0.5, label='Residual')
        else:
            ax1.scatter(time_ls[i], residuel_ls[i], color='#E18E6D',s=10 , alpha=0.5)

    #绘制拟合曲线
    total_time=np.concatenate(time_ls).reshape(-1) #完整时间序列
    total_residuel=np.concatenate(residuel_ls).reshape(-1) #完整残差序列
    total_draught=np.concatenate(draught_data).reshape(-1) #完整吃水序列
    # 创建一个基准时间（例如，第一个时间点）作为参考点
    base_time = total_time[0]

    # 计算每个时间点与基准时间之间的差异（以秒为单位）
    trans_total_time = pd.DataFrame({'time':total_time})
    base_time = trans_total_time['time'].min()
    trans_total_time['second']=(trans_total_time['time']-base_time).dt.total_seconds() #注意dt.total_seconds()是将时间转换为秒(只能针对应用于时间间隔对象（timedelta），而不是日期时间对象（datetime），因此需要减个数转化)

    data=pd.DataFrame({'time':trans_total_time['second'],'draught':total_draught})
    #进行自定义方程回归
    def func(data, a, b, c):  # 注意data是一个dataframe格式的数据，取第一列为时间，第二列为吃水
        return a*data.iloc[:,0]+b*data.iloc[:,1]+c

    # 下面进行函数拟合操作(回归出参数)
    params, pcov = curve_fit(func, data,total_residuel)
    print('参数为：',params)
    # 下面得到一维形式的拟合残差（可以将数据和参数直接传入func函数中直接得到）
    y_fit = params[0] * data.iloc[:, 0] + params[1]* data.iloc[:, 1] +params[2]
    if fit == True:  # 进行拟合
        # 绘制曲线
        ax1.plot(total_time, y_fit, lw=1.5, color='#bf0000', label='Curve fitting')
    print('退化量为：',y_fit.values[-1]-y_fit.values[0])
    # 绘制分界线
    for i in range(len(time_ls)):
        plt.axvline(x=time_ls[i][0], color='black', lw=1.8, linestyle=':')
        plt.axvline(x=time_ls[i][-1], color='black', lw=1.8, linestyle=':')

    # 写上文字标签
    #text = ['Fourth voyage', 'Fifth voyage', 'Sixth voyage', 'Seventh voyage', 'Eighth voyage', 'Ninth voyage']
    text=np.arange(start_index,end_index+1)-1
    for i in range(len(time_ls)):
        plt.text(x=time_ls[i][int(len(time_ls[i])//2)], y=np.max(total_residuel)+0.1, s=f'{text[i]}', color='black',fontsize=13)

    #注意这里提前为ax2的图像进行创建图例
    ax1.plot([],[],lw=1.5,marker='o',markersize=2.2,color='#52D896',label='Draught')  # 注意这里是为了提前创建图例

    #ax1.set_ylim(-10, 17)
    # 进行打上图例
    ax1.legend(loc='lower left')  # 打上图例
    ax1.set_xlabel(xlabel, fontsize=13, family='Times New Roman')  # 注意加粗目前英文才可以使用（中文需要变换字体）
    ax1.set_ylabel(ylabel, fontsize=13, family='Times New Roman',color='#bf0000')

    #绘制吃水数据
    ax2 = ax1.twinx()  # 和第一类数据共享x轴（新设置图
    #ax2.plot(total_time,total_draught, lw=1.5, markersize=2.2, color='#52D896',label='Draught')
    for i in range(len(draught_data)): #绘制线(间断形式)
        ax2.plot([time_ls[i][0],time_ls[i][-1]],[draught_data[i]]*2,lw=1.5,color='#52D896')


    ax2.set_ylabel('Draught (m)', fontsize=13, family='Times New Roman',color='#52D896')  # 设置ax2的y轴参数
    ax2.spines['left'].set_color('#bf0000')  # 设置右边 y 轴的颜色
    ax2.spines['right'].set_color('#52D896')  # 设置右边 y 轴的颜色
    ax2.tick_params(axis='y', labelcolor='#52D896', labelsize=12)  # 设置ax2的y轴颜色,同时设置y轴的字体大小,注意还可以
    ax2.set_ylim(0, 50)

    # 调整x轴标签范围(对于时间序列数据来说)
    if periods:
        plt.xticks(pd.date_range(time_ls[0][0], time_ls[-1][-1], periods=periods))
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=1000)  # 注意需要完成的路径
    plt.show()

#绘制散点图或者折线图并并进行拟合
def plot_fit(x,y,xlabel='',ylabel='',type='scatter',fit=True,degree=4,save_path=None):
    """
    :param x: 输入x数据
    :param y: 输入y数据
    :param xlabel: x轴标签
    :param ylabel: y轴标签
    :param type: 绘制类型
    :param fit: 是否进行拟合
    :param degree: 拟合的多项式次数
    :param save_path: 保存地址
    :return:
    """
    plt.figure(figsize=(10, 6))
    plt.tick_params(size=5, labelsize=13)  # 坐标轴
    plt.grid(alpha=0.3)  # 是否加网格线

    if type == 'scatter':
        plt.scatter(x, y, marker='o', color='#348FCA', s=30, alpha=1,label='Residual')
    elif type == 'curve':
        plt.plot(x, y, color='#FFDD24', lw=1.5,marker='o',alpha=1,markersize=3,label='Performance degradation curves')

    if fit == True:  # 进行拟合
        # 进行一元多项式回归
        coefficients = np.polyfit(x, y, degree)  # 得到多项式系数
        # 使用拟合的系数创建多项式函数
        polynomial = np.poly1d(coefficients)
        # 生成拟合曲线
        new_x = np.linspace(x.min(), x.max(), 100)
        y_fit = polynomial(new_x)
        # 绘制曲线
        plt.plot(new_x, y_fit, lw=1.5, color='#bf0000', label='Curve fitting')
    plt.legend()
    plt.xlabel(xlabel, fontsize=13, family='Times New Roman')  # 注意加粗目前英文才可以使用（中文需要变换字体）
    plt.ylabel(ylabel, fontsize=13, family='Times New Roman')

    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')  # 注意需要完成的路径
    plt.show()

# 平均绝对误差（MAE）
def mean_absolute_error(y1, y2):  # 其中y1是预测序列，y2是真实序列
    return np.sum(np.abs(np.array(y1) - np.array(y2))) / len(y1)

def mean_square_error(y1, y2):  # y1是预测值序列，y2是真实值序列
    return np.sum((np.array(y1) - np.array(y2)) ** 2) / len(y1)

# 平均绝对百分比误差(MAPE)
def Mean_Absolute_Percentage_Error(y1, y2):  # 其中y1是预测序列，y2是真实序列
    return np.mean(np.abs(np.array(y1) - np.array(y2)) / np.abs(np.array(y2)))

def computeCorrelation(x, y):  # 其中x,y均为序列，x是预测值，y是真实值,这里是计算Pearson相关系数，最后需要平方注意
    xBar = np.mean(x)  # 求预测值均值
    yBar = np.mean(y)  # 求真实值均值
    covXY = 0
    varX = 0  # 计算x的方差和
    varY = 0  # 计算y的方差和
    for i in range(0, len(x)):
        diffxx = x[i] - xBar  # 预测值减预测值均值
        diffyy = y[i] - yBar  # 真实值减真实值均值
        covXY += (diffxx * diffyy)
        varX += diffxx ** 2
        varY += diffyy ** 2
    return covXY / np.sqrt(varX * varY)
