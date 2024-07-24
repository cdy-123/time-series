import os
import json
import time
import csv
import matplotlib.pyplot as plt
import numpy as np
from core.data_processor import DataLoader
from core.model import Model
from sklearn.preprocessing import MinMaxScaler

# 绘图展示结果
def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='GPT')
    print(predicted_data[0])
    with open('fa-pre.csv','w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['fa'])
        for i in predicted_data[0]:
            writer.writerow([i])
    plt.plot(predicted_data[0], label='Prediction')
    plt.legend()
    plt.show()

#时间序列
def main():
    #读取所需参数
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
    #读取数据
    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )
    scaler_train = MinMaxScaler(feature_range=(-1, 1))##新改的
    scaler_test = MinMaxScaler(feature_range=(-1, 1))##新改的
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
    
	
    model.load_model("saved_models/06062024-094254-e100.keras")

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