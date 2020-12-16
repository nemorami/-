import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Input, Reshape, Concatenate, Flatten
from tensorflow.keras.models import Model
import numpy as np

class AnnKeras:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def set_embedding(self, categories):
        self.ins = []
        self.concat = []
        for i in range(self.X_train.shape[1]):
            x = Input(shape=(1,), name = self.X_train.columns[i])
            self.ins.append(x)
            if i in categories:
                ncount = self.X_train.iloc[:,i].nunique()
                x = Embedding(input_dim=ncount, output_dim=min(1000, ncount//2 + 1))(x)
                x = Flatten()(x)

            self.concat.append(x)

    def set_model(self):
        output = Concatenate(name='combined')(self.concat)
        output = Dense(1000, activation='relu')(output)
        output = Dense(50, activation='relu')(output)
        output = Dense(1, activation='sigmoid')(output)
        self.model = Model(self.ins, output)
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

    def fit(self):
        X_train = {}
        for vars in self.X_train.columns.tolist():
            X_train[vars] = self.X_train[vars].values
        print(X_train)
        #self.model.fit(X_train, self.y_train, epochs=2)
        #self.model.summary()
        #print(self.X_train)

    def predict(self):
        X_test = {}
        for vars in self.X_test.columns.tolist():
            X_test[vars] = self.X_test[vars].values

        y_pred = self.model.predict(X_test)
        print(y_pred)




#모델구축


#model.fit(X_train, y_train)

#model.evaluate

