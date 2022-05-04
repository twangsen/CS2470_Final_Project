import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold


file_name = 'combined_data/51features_RF.csv'

def resampling_data(input_file):
    data = pd.read_csv(input_file)

    #print(data['OHAREC'].value_counts())
    class_count_4, class_count_3,class_count_2,class_count_1 = data['OHAREC'].value_counts()
    print(class_count_1, class_count_2,class_count_3,class_count_4)
    # Separate class
    class_1 = data[data['OHAREC'] == 1]# print the shape of the class
    class_2 = data[data['OHAREC'] == 2]
    class_3 = data[data['OHAREC'] == 3]
    class_4 = data[data['OHAREC'] == 4]

    # print('class 1:', class_1.shape)
    # print('class 2:', class_2.shape)
    # print('class 3:', class_3.shape)
    # print('class 4:', class_4.shape)

    class_3_under = class_3.sample(class_count_2)
    class_4_under = class_4.sample(class_count_2)
    class_1_over = class_1.sample(100, replace=True)

    test_under = pd.concat([class_1_over,class_2, class_3_under, class_4_under], axis=0)

    print("total class: ",test_under['OHAREC'].value_counts())# plot the count after under-sampeling
    #test_under['OHAREC'].value_counts().plot(kind='bar', title='count (target)')

    return test_under

def get_data(data):
    #data = pd.read_csv(input_file)
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
    

    train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size=0.1, shuffle=True)

    return train_X, train_Y, test_X, test_Y


data = resampling_data(file_name)
train_X, train_Y, test_X, test_Y = get_data(data)
model = Sequential([
    Dense(60, activation='relu'),
    Dense(40, activation='relu'),
    Dense(20,  activation='relu'),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer='Adam',loss='binary_crossentropy', metrics=['accuracy'])

hist = model.fit(train_X, train_Y,
           batch_size=100, epochs=20,
           validation_data=(test_X, test_Y))

print(model.evaluate(test_X, test_Y)[1])