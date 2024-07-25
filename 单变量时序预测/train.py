import os
import json
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from core.data_processor import DataLoader
from core.model import Model
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler

# 绘图展示结果
def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    print(predicted_data[0])
    plt.plot(predicted_data[0], label='Prediction')
    plt.legend()
    plt.show()



def main():
    #读取所需参数
    configs = json.load(open('fa-year.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
    #读取数据
    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )
    #数据标准化
    scaler_train = MinMaxScaler(feature_range=(-1, 1))
    scaler_test = MinMaxScaler(feature_range=(-1, 1))
    data.data_train = scaler_train.fit_transform(data.data_train.reshape(-1, 1))
    data.data_test = scaler_test.fit_transform(data.data_test.reshape(-1, 1))

    
    #创建模型
    model = Model()
    mymodel = model.build_model(configs)
    
    
    #加载训练数据
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
    print (x.shape)
    print (y.shape)
    
	# 训练模型
    model.train(
		x,
		y,
		epochs = configs['training']['epochs'],
		batch_size = configs['training']['batch_size'],
		save_dir = configs['model']['save_dir']
	)
	

   #测试结果
    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    y_test = scaler_test.inverse_transform(y_test)
    
    #展示测试效果
    predictions_multiseq = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['pred_length'])
    predictions_multiseq = scaler_test.inverse_transform(predictions_multiseq)
    plot_results_multiple(predictions_multiseq, y_test, configs['data']['sequence_length'])

    
if __name__ == '__main__':
    main()