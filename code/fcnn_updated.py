import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold


file_name = '../combined_data/51features_RF.csv'
# batch_size = 100
# no_epochs = 100


def get_data(input_file):
    data = pd.read_csv(input_file)
    data = data.values
    
    # Standardization
    scaler = preprocessing.StandardScaler().fit(data)
    data = scaler.transform(data)
    # Scaling features to a range (0,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)

    x = data[:, :-1]
    y = data[:, -1]

    # the data set are extremely unbalanced
    # the ratio of the orignal values are roughly 0.0004 : 0.05 : 0.36 : 0. 58
    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))
    
    # combined all data labled with 0, 0.3, 0.6.  (which are 0, 1, 2 before normalized)
    y[y < 1] = 0

    # the ratio of 0s 1s is roughly 0.42 : 0.58
    unique, counts = np.unique(y, return_counts=True)
    counts = counts / len(y)
    print(dict(zip(unique, counts)))

    train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size=0.3, shuffle=True)

    return train_X, train_Y, test_X, test_Y


model = Sequential([
    Dense(50, activation='relu'),
    Dense(32,  activation='relu'),
    Dense(1, activation='sigmoid')
])

train_X, train_Y, test_X, test_Y = get_data(file_name)
model.compile(optimizer='Adam',loss='binary_crossentropy', metrics=['accuracy'])

hist = model.fit(train_X, train_Y,
           batch_size=100, epochs=20,
           validation_data=(test_X, test_Y))