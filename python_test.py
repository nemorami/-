import pandas as pd
from sklearn import preprocessing

import torch
import torch.optim as optim

# x_train = torch.FloatTensor([[1], [2], [3]])
# y_train = torch.FloatTensor([[2], [4], [6]])
#
# print(x_train)
# print(x_train.shape)
#
# w = torch.zeros(1, requires_grad=True)
# b = torch.zeros(1, requires_grad=True)
#
# h = x_train * w + b
# print(h)
# c = torch.mean((h - y_train)**2)
# print(c)
#
# o = optim.SGD([w, b], lr=0.01)
# o.zero_grad()
# c.backward()
# o.step()

data = {'score': [234, 24, 14, 27, 74, 46, 73, 18, 59, 160]}
cols = data.columns
df = pd.DataFrame(data)
df

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df)
df_normalized = pd.DataFrame(np_scaled, columns = cols)
df_normalized