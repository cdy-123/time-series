import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# 读取数据
data = pd.read_csv("data-train.csv")
print(data.head())
print("shape:",data.shape)

# 建立线性回归模型

# 使用 pandas 构建 X（特征向量）和 y（标签列）
feature_cols = ["people","urban","gdp","industry"] #影响因素列名
X = data[feature_cols]
y = data["power"] #预测列名

# 构建线性回归模型并训练
model = LinearRegression().fit(X,y)

# 输出模型结果
print("截距：",model.intercept_)
coef = zip(feature_cols, model.coef_)
print("回归系数：",list(coef))
y_pred=model.predict(X)
print("平均绝对误差（MAE）：",mean_absolute_error(y,y_pred))
print("均方误差（MSE）：",mean_squared_error(y,y_pred))
print("R方值：",r2_score(y,y_pred))

# 展示拟合结果（根据自己数据集修改）
x=np.arange(72)
plt.plot(range(len(y_pred)),y_pred,"b",label="predict")
plt.plot(range(len(y)),y,"r",label="true")
plt.xticks(x[::10],range(1952,2023,10))
plt.legend(loc='upper left')
plt.xlabel("year")
plt.ylabel("power")
plt.show()


# 预测
data = pd.read_csv("data-test.csv")
feature_cols = ["people","urban","gdp","industry"] #影响因素列名
X_test = data[feature_cols]
y_pred = model.predict(X_test)
print(y_pred)

# 保存预测的结果（根据自己数据集修改）
with open('fa-pre.csv','w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['year','power'])
    for i in range(0,27):
        writer.writerow([i+2024,y_pred[i]])

# 绘制预测曲线
x=np.arange(27)
plt.plot(range(len(y_pred)),y_pred,"b",label="predict")
plt.xticks(x[::5],range(2024,2050,5)) #根据自己的数据集修改
plt.legend(loc='upper left')
plt.xlabel("year")
plt.ylabel("power")
plt.show()
