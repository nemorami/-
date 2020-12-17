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
        self.ins = []
        self.concat = []
        #self.model = 0

    def set_embedding(self, categories):

        for i in range(self.X_train.shape[1]):
            c_names = self.X_train.columns[i]
            print(c_names)
            x = Input(shape=(1,), name = c_names)
            self.ins.append(x)
            if i in categories:
                ncount = self.X_train.iloc[:,i].max()
                x = Embedding(input_dim=ncount+1, output_dim=min(1000, ncount//2 + 1), name='Embedding_'+c_names)(x)
                x = Flatten(name='Flatten_'+c_names)(x)

            self.concat.append(x)

    def set_model(self):
        output = Concatenate(name='combined')(self.concat)
        output = Dense(1000, activation='relu')(output)
        output = Dense(50, activation='relu')(output)
        output = Dense(1, activation='sigmoid')(output)
        self.model = Model(self.ins, output)
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

    def fit(self):
        x_train = {}
        for vars in self.X_train.columns.tolist():
            x_train[vars] = self.X_train[vars].values

        self.model.fit(x_train, self.y_train, batch_size=256,  epochs=2)
        #self.model.summary()
        #print(self.X_train)

    def predict(self):
        x_test = {}
        for vars in self.X_test.columns.tolist():
            x_test[vars] = self.X_test[vars].values

        y_pred = self.model.predict(x_test)
        print(y_pred.reshape(-1))




#모델구축


#model.fit(X_train, y_train)

#model.evaluate

