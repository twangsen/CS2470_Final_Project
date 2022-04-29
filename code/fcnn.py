from webbrowser import get
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense

def get_data(input_file):
    data = pd.read_csv(input_file)
    data = data.values
    # Standardization
    scaler = preprocessing.StandardScaler().fit(data)
    data = scaler.transform(data)
    # Scaling features to a range (0,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(data)
    x = data[:,1:]
    y = data[:,0]
    train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size=0.3, random_state=50)
    return train_X, train_Y, test_X, test_Y

file_name = 'combined_data/51features_RF.csv'
train_X, train_Y, test_X, test_Y = get_data(file_name)
print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape )

model = Sequential()
model.add(Dense(50, input_dim=49, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

hist = model.fit(train_X, train_Y,
          batch_size=32, epochs=100,
          validation_data=(test_X, test_Y))

print(model.evaluate(test_X, test_Y)[1])