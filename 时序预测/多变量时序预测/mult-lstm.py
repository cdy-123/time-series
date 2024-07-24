import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib

## 读取数据集
df=pd.read_csv("train.csv",parse_dates=["Date"],index_col=[0])
print(df.shape)
print(df.head())
print(df.tail())
test_split=round(len(df)*0.20) # 测试集和训练集划分
df_for_training=df[:-test_split]
df_for_testing=df[-test_split:]
print(df_for_training.shape)
print(df_for_testing.shape)

## 标准化数据
scaler_train = MinMaxScaler(feature_range=(-1, 1))
df_for_training_scaled = scaler_train.fit_transform(df_for_training)
df_for_testing_scaled=scaler_train.transform(df_for_testing)
print(df_for_training_scaled)
print(df_for_testing_scaled)

def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
        dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
        dataY.append(dataset[i,0])
    return np.array(dataX),np.array(dataY)   

## 准备模型输入数据
trainX,trainY=createXY(df_for_training_scaled,30) ## 单位时间序列长度30，可根据需求更改
testX,testY=createXY(df_for_testing_scaled,30) ## 单位时间序列长度30，可根据需求更改
print("trainX Shape: ",trainX.shape)
print("trainY Shape: ",trainY.shape)
print("testX Shape: ",testX.shape)
print("testY Shape: ",testY.shape)

## 构建模型
def build_model():
    grid_model = Sequential()
    grid_model.add(LSTM(50,return_sequences=True,input_shape=(30,5))) ## input_shape根据单位时间序列长度更改
    grid_model.add(LSTM(50))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(1))

    grid_model.compile(loss = 'mse')
    return grid_model

grid_model = KerasRegressor(build_fn=build_model,verbose=1)
parameters = {'batch_size' : [8,16],
              'epochs' : [8,10],
              'optimizer' : ['adam','Adadelta'] }

grid_search  = GridSearchCV(estimator = grid_model,
                            param_grid = parameters,
                            cv = 2)

## 训练
grid_search = grid_search.fit(trainX,trainY,validation_data=(testX,testY)) 
print(grid_search.best_params_)
my_model=grid_search.best_estimator_
joblib.dump(my_model, 'Model.h5')
print(my_model)
print('Model Saved!')

## 预测
my_model = joblib.load('Model.h5') # 加载模型
prediction=my_model.predict(testX)
print("prediction\n", prediction)
print("\nPrediction Shape-",prediction.shape)
prediction_copies_array = np.repeat(prediction,5, axis=-1)
pred=scaler_train.inverse_transform(np.reshape(prediction_copies_array,(len(prediction),5)))[:,0]
original_copies_array = np.repeat(testY,5, axis=-1)
original=scaler_train.inverse_transform(np.reshape(original_copies_array,(len(testY),5)))[:,0]
print("Pred Values-- " ,pred)
print("\nOriginal Values-- ",original)

## 绘制预测结果
plt.plot(original, color = 'red', label = 'Real  Stock Price')
plt.plot(pred, color = 'blue', label = 'Predicted  Stock Price')
plt.title(' Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(' Stock Price')
plt.legend()
plt.show()

## 预测未来
df_30_days_past=df.iloc[-30:,:] # 过去30天的数据 
df_30_days_future=pd.read_csv("test.csv",parse_dates=["Date"],index_col=[0]) # 未来30天的其他数据
print(df_30_days_future.shape)

df_30_days_future["Open"]=0
df_30_days_future=df_30_days_future[["Open","High","Low","Close","Adj Close"]]
old_scaled_array=scaler_train.transform(df_30_days_past)
new_scaled_array=scaler_train.transform(df_30_days_future)
new_scaled_df=pd.DataFrame(new_scaled_array)
new_scaled_df.iloc[:,0]=np.nan
full_df=pd.concat([pd.DataFrame(old_scaled_array),new_scaled_df]).reset_index().drop(["index"],axis=1)
print(full_df.shape)
print(full_df)
full_df_scaled_array=full_df.values
all_data=[]
time_step=30
for i in range(time_step,len(full_df_scaled_array)):
    data_x=[]
    data_x.append(full_df_scaled_array[i-time_step:i,0:full_df_scaled_array.shape[1]])
    data_x=np.array(data_x)
    prediction=my_model.predict(data_x)
    all_data.append(prediction)
    full_df.iloc[i,0]=prediction

new_array=np.array(all_data)
new_array=new_array.reshape(-1,1)
prediction_copies_array = np.repeat(new_array,5, axis=-1)
y_pred_future_30_days = scaler_train.inverse_transform(np.reshape(prediction_copies_array,(len(new_array),5)))[:,0]
print(y_pred_future_30_days)
