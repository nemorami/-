import ann_model
import torch
import pandas as pd
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
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


x = x.values
print(x)
y = y.values
#
x_train,x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=33)


#위에서 설명한 데이터 텐서화
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
#
# model = ann_model.AnnModel(x_train, y_train)
#
# model.train()
#
# model.test(x_test, y_test)