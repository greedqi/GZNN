import pandas as pd
import numpy as np
import torch
import model
import math
from sklearn.metrics import mean_squared_error, r2_score
from parameters import *



data = pd.read_csv('Debutanizer_Data.txt', sep='\s+').values

x_all=[]
y_all=[]


for i in range(data.shape[0] - SEQ_LEN-SEQ2_LEN + 1):
    x_all.append(data[i: i + SEQ_LEN+SEQ2_LEN, 0:7])
    # y_all.append(data[i+SEQ_LEN : i + SEQ_LEN+SEQ2_LEN, 7])
    y_all.append(data[i: i + SEQ_LEN+SEQ2_LEN, 7])

#将数组沿一个新轴合并
x_all = np.stack(x_all, 0)
y_all = np.stack(y_all, 0)


x_train=x_all[:TRAIN_SIZE,:]
x_test=x_all[TRAIN_SIZE:,:]

y_train=y_all[:TRAIN_SIZE]
y_test=y_all[TRAIN_SIZE:]

x_train=torch.tensor(x_train, dtype=torch.float32, device=DEVICE)
x_test=torch.tensor(x_test, dtype=torch.float32, device=DEVICE)
y_train=torch.tensor(y_train, dtype=torch.float32, device=DEVICE)

if x_train.ndim != y_train.ndim :
    y_train=y_train.unsqueeze(-1)


my_model=model.Model(x_train,y_train,n_epoch=EPOCH,batch_size=BATCH_SIZE,  device=DEVICE).fit(x_train,y_train,x_test,y_test)

y_pred = my_model.predict(X=x_test)


print('\n测试集的MSE：', mean_squared_error(y_test, y_pred))
print('\n测试集的决定系数：', r2_score(y_test, y_pred))




print()
