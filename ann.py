import ann_model
import torch
import pandas as pd
import numpy as np
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split

import ann_keras

#df = pd.read_csv('dec.csv')
df = pd.read_csv('가상데이터_훈련_테스트.csv')

# year, month, day, office, IMPORTER.TIN, TARIFF.CODE, DECLARANT.CODE, ORIGIN.CODE, cif_usd_equivalent, quantity, GROSS.WEIGHT, TOTAL.TAXES.USD
x = df.drop(['year', 'month', 'day', 'CIF_USD_EQUIVALENT', 'QUANTITY', 'GROSS.WEIGHT', 'TOTAL.TAXES.USD', 'RAISED_TAX_AMOUNT_USD', 'illicit' ], axis=1)
y = df['illicit']

x['GTAXES'] = df['CIF_USD_EQUIVALENT'] / df['GROSS.WEIGHT']
x['QTAXES'] = df['CIF_USD_EQUIVALENT'] / df['QUANTITY']

# add avg by HS

def string_to_digit(x, *strs):
    """문자열 피처를 숫자형으로 바꾼다."""
    label_encoder = pp.LabelEncoder()
    for str in strs:
        x[str] = label_encoder.fit_transform(x[str])

string_to_digit(x, 'OFFICE', 'IMPORTER.TIN','TARIFF.CODE','ORIGIN.CODE' , 'DECLARANT.CODE')

#x = x.apply(lambda y: (y - np.mean(y)) / (np.max(y) - np.min(y)))

#x = x.values
#y = y.values
#

x_train,x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=33)
print(x_test)
#np.set_printoptions(threshold=np.inf)
#print(x_train.shape)
#print(x_train.iloc[:,0].nunique())
model = ann_keras.AnnKeras(x_train, y_train, x_test, y_test)
#model.set_embedding([0,1,2,3,4])
model.set_embedding([])
model.set_model()
model.fit()
# model.predict()
#위에서 설명한 데이터 텐서화
# x_train = torch.FloatTensor(x_train)
# x_test = torch.FloatTensor(x_test)
# y_train = torch.LongTensor(y_train)
# y_test = torch.LongTensor(y_test)


#
# model = ann_model.AnnModel(x_train, y_train)
#
# model.train()
#
# model.test(x_test, y_test)