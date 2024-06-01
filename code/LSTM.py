import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout,Bidirectional,Activation
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from Universal import plot_curve
#创建LSTM模型
def creat_model(n_layer=1,hidden_unit1=200,hidden_unit2=50,dropout_rate=0.5,n_feature=27,n_steps=1,kerner='lstm'): #n_step表示时间步长
    model = Sequential()
    for i in range(int(n_layer)): #确定LSTM的层数
        if kerner=='lstm':
            if i==0:
                model.add(LSTM(hidden_unit1, activation='relu', input_shape=(n_steps, n_feature),return_sequences=n_layer>1)) #双向LSTM：model.add(Bidirectional(LSTM(64),activation='relu', input_shape=(n_steps, 1)))
            else:
                model.add(LSTM(hidden_unit1, activation='relu',return_sequences=(i < (n_layer - 1))))
        elif kerner=='gru':
            if i==0:
                model.add(GRU(hidden_unit1, activation='relu', input_shape=(n_steps, n_feature))) #双向LSTM：model.add(Bidirectional(LSTM(64),activation='relu', input_shape=(n_steps, 1)))
            else:
                model.add(GRU(hidden_unit1, activation='relu',return_sequences=(i < n_layer - 1)))

    model.add(Dropout(dropout_rate))  # 加入0.1的Dropout
    model.add(Dense(hidden_unit2, input_dim=hidden_unit1, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(1,input_dim=hidden_unit2,kernel_initializer='glorot_uniform'))
    optimizer = Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='mse') #默认初始学习率为0.001
    return model

class LSTM_model():
    def __init__(self,X_train,X_test,y_train,y_test,epoch=100,batchsize=64,n_layer=1,hidden_unit1=200,hidden_unit2=50,dropout_rate=0.5,n_steps=1,n_feature=27,save_best_path=None):
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
        self.epoch=epoch
        self.batchsize=batchsize
        self.n_layer=n_layer
        self.hidden_unit1=hidden_unit1
        self.hidden_unit2=hidden_unit2
        self.dropout_rate=dropout_rate
        self.n_feature=n_feature
        self.n_steps=n_steps
        self.save_best_path=save_best_path #是否保存最优模型的具体路径（模型最后保存的文件是.hdf5文件后缀）
        #创建LSTM模型
        self.model=creat_model(n_layer=n_layer,hidden_unit1=hidden_unit1,hidden_unit2=hidden_unit2,dropout_rate=dropout_rate,n_feature=n_feature,n_steps=n_steps)
        self.model_checkpoint=ModelCheckpoint(self.save_best_path,monitor='val_loss',save_best_only=False,save_weights_only=True)
        #self.early_stopping=EarlyStopping(monitor='val_loss',patience=10,verbose=1)
        #下面是微调学习率，如果在5轮内验证集精度未上升，则学习率减半
        self.reduce_lr=ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=10,verbose=1)

    def run(self):
        """
        训练模型
        """
        if self.save_best_path is None:
            self.history = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batchsize, validation_data=(self.X_test, self.y_test), shuffle=False,
                                verbose=2, callbacks=[self.reduce_lr])
        else:
            self.history = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batchsize,
                                          validation_data=(self.X_test, self.y_test), shuffle=False,
                                          verbose=2, callbacks=[self.reduce_lr,self.model_checkpoint])
    def best_obj(self):
        """
        :return: 最好的测试精度
        """
        return np.min(self.history.history['val_loss'])

class GRU_model():
    def __init__(self,X_train,X_test,y_train,y_test,epoch=100,batchsize=64,n_layer=1,hidden_unit1=200,hidden_unit2=50,dropout_rate=0.5,n_steps=1,n_feature=27,save_best_path=None):
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
        self.epoch=epoch
        self.batchsize=batchsize
        self.n_layer=n_layer
        self.hidden_unit1=hidden_unit1
        self.hidden_unit2=hidden_unit2
        self.dropout_rate=dropout_rate
        self.n_feature=n_feature
        self.n_steps=n_steps
        self.save_best_path=save_best_path #是否保存最优模型的具体路径（模型最后保存的文件是.hdf5文件后缀）
        #创建LSTM模型
        self.model=creat_model(n_layer=n_layer,hidden_unit1=hidden_unit1,hidden_unit2=hidden_unit2,dropout_rate=dropout_rate,n_feature=n_feature,n_steps=n_steps,kerner='gru')
        self.model_checkpoint=ModelCheckpoint(self.save_best_path,monitor='val_loss',save_best_only=True,save_weights_only=True)
        #self.early_stopping=EarlyStopping(monitor='val_loss',patience=10,verbose=1)
        #下面是微调学习率，如果在5轮内验证集精度未上升，则学习率减半
        self.reduce_lr=ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5,verbose=1)

    def run(self):
        """
        训练模型
        """
        if self.save_best_path is None:
            self.history = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batchsize, validation_data=(self.X_test, self.y_test), shuffle=False,
                                verbose=2, callbacks=[self.reduce_lr])
        else:
            self.history = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batchsize,
                                          validation_data=(self.X_test, self.y_test), shuffle=False,
                                          verbose=2, callbacks=[self.reduce_lr,self.model_checkpoint])
    def best_obj(self):
        """
        :return: 最好的测试精度
        """
        return np.min(self.history.history['val_loss'])

#创建BP神经网络模型
class BPNN():
    def __init__(self,X_train,X_test,y_train,y_test,epoch=100,batchsize=64,dropout_rate=0.1,n_feature=27,save_best_path=None):
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
        self.epoch=epoch
        self.batchsize=batchsize
        self.dropout_rate=dropout_rate
        self.n_feature=n_feature
        self.save_best_path=save_best_path #是否保存最优模型的具体路径（模型最后保存的文件是.hdf5文件后缀）
        #创建LSTM模型
        self.model=self.creat_BP_model()
        self.model_checkpoint=ModelCheckpoint(self.save_best_path,monitor='val_loss',save_best_only=True,save_weights_only=True)
        #self.early_stopping=EarlyStopping(monitor='val_loss',patience=10,verbose=1)
        #下面是微调学习率，如果在5轮内验证集精度未上升，则学习率减半
        self.reduce_lr=ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5,verbose=1)

    def creat_BP_model(self):
        # 先初始化一个BP神经网络
        model = Sequential()
        # 加入输入层
        model.add(Dense(units=224, input_dim=self.n_feature, kernel_initializer='glorot_uniform',
                        activation='relu'))  # unit是神经元的数量,input_dim是输入的纬度，activation是激活函数
        model.add(Dropout(rate=self.dropout_rate))  # 加入dropout防止过拟合
        # 加入隐含层
        model.add(Dense(units=112, input_dim=224, kernel_initializer='glorot_uniform',
                        activation='relu'))  # 这里的input_dim要与上一层的神经员个数对上
        model.add(Dropout(rate=self.dropout_rate))  # 加入dropout防止过拟合
        model.add(Dense(units=56, input_dim=112, kernel_initializer='glorot_uniform', activation='relu'))
        model.add(Dropout(rate=self.dropout_rate))  # 加入dropout防止过拟合
        model.add(Dense(units=16, input_dim=56, kernel_initializer='glorot_uniform', activation='relu'))
        model.add(Dropout(rate=self.dropout_rate))  # 加入dropout防止过拟合
        # 添加输出层
        model.add(Dense(units=1, input_dim=16, kernel_initializer='glorot_uniform'))

        #进行模型配置
        optimizer = Adam(lr=0.0001)
        model.compile(optimizer=optimizer, loss='mse')  # 默认初始学习率为0.001

        return model

    def run(self):
        """
        训练模型
        """
        if self.save_best_path is None:
            self.history = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batchsize, validation_data=(self.X_test, self.y_test), shuffle=False,
                                verbose=2, callbacks=[self.reduce_lr])
        else:
            self.history = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batchsize,
                                          validation_data=(self.X_test, self.y_test), shuffle=False,
                                          verbose=2, callbacks=[self.reduce_lr,self.model_checkpoint])
    def best_obj(self):
        """
        :return: 最好的测试精度
        """
        return np.min(self.history.history['val_loss'])







