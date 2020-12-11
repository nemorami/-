import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as pp


# ANN 모델 생성
class Model(nn.Module):
    def __init__(self, in_features=7, h1 =8,h2=9, out_features=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2, out_features)

    #순전파
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


#torch.manual_seed(32)
model = Model()



#df = pd.read_csv('dec.csv')
df = pd.read_csv('가상데이터_훈련_테스트.csv')
#
#x = df.drop(['sgd.id', 'sgd.date', 'declarant.id', 'office.id', 'quantity','gross.weight','fob.value','cif.value', 'total.taxes', 'revenue', 'illicit'], axis=1)
x = df.drop(['year', 'month', 'day', 'CIF_USD_EQUIVALENT', 'QUANTITY', 'GROSS.WEIGHT', 'RAISED_TAX_AMOUNT_USD', 'illicit' ], axis=1)
y = df['illicit']
x['TOTAL.TAXES.USD'] = df['CIF_USD_EQUIVALENT'] / df['GROSS.WEIGHT']
x['QTAXES'] = df['CIF_USD_EQUIVALENT']  / df['QUANTITY']

def string_to_digit(x, y):
    label_encoder = pp.LabelEncoder()
    x[y] = label_encoder.fit_transform(x[y])

#string_to_digit(x, 'importer.id')
#string_to_digit(x, 'tariff.code')

string_to_digit(x, 'OFFICE')
string_to_digit(x, 'IMPORTER.TIN')
string_to_digit(x, 'TARIFF.CODE')
string_to_digit(x, 'ORIGIN.CODE')
string_to_digit(x, 'DECLARANT.CODE')

print(x.head)
x = x.values
y = y.values
#
x_train,x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=33)

#위에서 설명한 데이터 텐서화
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

#손실함수 정의
criterion = torch.nn.CrossEntropyLoss()

#최적화 함수 정의
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 500 #훈련 횟수 1000번
losses = [] # loss를 담을 리스트, 시각화 하기 위함

for i in range(epochs):
    model.train()
    y_pred = model(x_train)

    loss = criterion(y_pred, y_train)
    losses.append(loss)

    if i % 10 ==0:
        print(f'epoch {i}, loss is {loss}')

    # 역전파 수행
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

correct = 0
no = 0

with torch.no_grad():
    for i, data in enumerate(x_test):
        y_val = model.forward(data)
        print(f'{i+1}.) {str(y_val.argmax().item())} {y_test[i]}')
        if y_val.argmax().item() == y_test[i]:
            correct += 1
        else:
            no += 1

print(f'We got {correct} correct no {no} : {correct/(correct+no)}.')