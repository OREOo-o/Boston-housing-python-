from sklearn.linear_model import LinearRegression,Lasso,Ridge

import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

train_data = pd.read_csv(r'C:\Users\15389\Documents\Tencent Files\1538954098\FileRecv\BostonQAQ.csv')

target = train_data['MEDV']  # 样本的目标值
data = train_data.drop('MEDV', axis=1).values  # 样本的特征值

X_train = data[56:]
y_train = target[56:]
# 测试数据
X_test = data[:56]  # 测试数据的特征值
y_test = target[:56]  # 测试数据的结果

lr = LinearRegression()
rr = Ridge()
lasso = Lasso()

lr.fit(X_train,y_train)
rr.fit(X_train,y_train)
lasso.fit(X_train,y_train)

y_lr_ = lr.predict(X_test)
y_rr_ = rr.predict(X_test)
y_lasso_ = lasso.predict(X_test)

plt.plot(y_test,label='real')
plt.plot(y_lr_,label='lr')
plt.legend() 
plt.show()
plt.plot(y_test,label='real')
plt.plot(y_rr_,label='rr')
plt.legend() 
plt.show()
plt.plot(y_test,label='real')
plt.plot(y_lasso_,label='lasso')
plt.legend() 
plt.show()
print(lr.score(X_test,y_test))
print(rr.score(X_test,y_test))
print(lasso.score(X_test,y_test))


