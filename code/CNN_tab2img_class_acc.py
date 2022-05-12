import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import fbeta_score
from tab2img.converter import Tab2Img

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import regularizers

import matplotlib.pyplot as plt
from PIL import Image

df = pd.read_csv("../combined_data/51features_RF.csv")
y = df["OHAREC"].values
y = y - 1
X = df.drop(['OHAREC'], axis=1)

class1 = df[y == 0].drop(['OHAREC'], axis=1)
class2 = df[y == 1].drop(['OHAREC'], axis=1)
class3 = df[y == 2].drop(['OHAREC'], axis=1)
class4 = df[y == 3].drop(['OHAREC'], axis=1)

# try to resample the data
# class1 = class1.sample(1000, replace=True)
# class2 = class2.sample(3000, replace=True)
# class3 = class3.sample(3000)
# class4 = class4.sample(3000)
# print(class1.shape, class2.shape)

# X = np.concatenate((class1, class2, class3, class4))
# y = np.concatenate((np.zeros(1000), np.ones(3000), [2]*3000, [3]*3000))
print(X.shape, y.shape)

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.10, random_state=42)

# print(X_train.shape)
# print(X_test.shape)

# transform the data
scaler = preprocessing.StandardScaler().fit(X_train)
X_scaled_train = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)

min_max_scaler = preprocessing.MinMaxScaler().fit(X_scaled_train)
X_scaled_train = min_max_scaler.transform(X_scaled_train)
X_scaled_test = min_max_scaler.transform(X_scaled_test)



X_class1 = scaler.transform(class1)
X_class2 = scaler.transform(class2)
X_class3 = scaler.transform(class3)
X_class4 = scaler.transform(class4)
# y_scaler =  preprocessing.StandardScaler().fit(y_train.values.reshape(-1, 1))
# y_scaled_train = y_scaler.transform(y_train.values.reshape(-1, 1))
# y_scaled_test = y_scaler.transform(y_test.values.reshape(-1, 1))
# print(X_scaled_test.shape, y_scaled_test.shape)

# convert the data into images
model = Tab2Img()
train_images = model.fit_transform(X_scaled_train, y_train)
test_images = model.transform(X_scaled_test)
train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)

print(train_images.shape)

class1_images = model.transform(X_class1)
class1_images = np.expand_dims(class1_images, -1)
class2_images = model.transform(X_class2)
class2_images = np.expand_dims(class2_images, -1)
class3_images = model.transform(X_class3)
class3_images = np.expand_dims(class3_images, -1)
class4_images = model.transform(X_class4)
class4_images = np.expand_dims(class4_images, -1)
# print(train_images.shape, test_images.shape)
# print(y_train)

# visualize the transformed data images
fig,ax = plt.subplots(2,5)
for i in range(10):
    nparray = test_images[i].reshape(7,7)
    image = Image.fromarray(nparray * 255)
    ax[i%2][i//2].imshow(image)
fig.savefig("data.png")
fig.show()

# # train the model
model = keras.Sequential([
    # layers.Conv2D(32, (2, 2), activation='relu', padding='same'),
    layers.Conv1D(32, 2, activation='relu', padding='same'),
    layers.BatchNormalization(),
    # layers.MaxPooling2D((2, 2), padding='same'),
    #layers.Conv2D(64, (2, 2), activation='relu', padding='same'),
    layers.Conv1D(64, 2, activation='relu', padding='same'),
    layers.BatchNormalization(),
    # layers.MaxPooling2D((2, 2), padding='same'),
    # layers.Conv2D(128, (2, 2), activation='relu', padding='same'),
    layers.Conv1D(128, 2, activation='relu', padding='same'),
    layers.BatchNormalization(),
    # layers.MaxPooling2D((2, 2), padding='same'),
    # layers.Conv2D(64, (2, 2), activation='relu', padding='same'),
    layers.Conv1D(64, 2, activation='relu', padding='same'),
    layers.BatchNormalization(),
    # layers.MaxPooling2D((2, 2), padding='same'),
    # layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.Conv1D(128, 2, activation='relu', padding='same'),
    layers.BatchNormalization(),
    # layers.MaxPooling2D((2, 2), padding='same'),
    # layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.Conv1D(128, 2, activation='relu', padding='same'),
    layers.BatchNormalization(),
    # layers.MaxPooling2D((2, 2), padding='same'),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(4, activation='softmax')
])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(train_images, y_train, batch_size = 200, epochs=20, validation_data=(test_images, y_test))
# model.summary()



class1_pred = model.predict(class1_images)
print("class1:", np.sum(np.argmax(class1_pred, axis = 1) == 0) / len(class1_pred))

class2_pred = model.predict(class2_images)
print("class2:", np.sum(np.argmax(class2_pred, axis = 1) == 1) / len(class2_pred))

class3_pred = model.predict(class3_images)
print("class3:", np.sum(np.argmax(class3_pred, axis = 1) == 2) / len(class3_pred))

class4_pred = model.predict(class4_images)
print("class4:", np.sum(np.argmax(class4_pred, axis = 1) == 3) / len(class4_pred))

# y_test_pred = model.predict(test_images)
# f1_score = fbeta_score(y_test, np.argmax(y_test_pred, axis=1), beta=1, average='micro')
# print("f1 score: ", f1_score)

# METRICS = [
#       keras.metrics.TruePositives(name='tp'),
#       keras.metrics.FalsePositives(name='fp'),
#       keras.metrics.TrueNegatives(name='tn'),
#       keras.metrics.FalseNegatives(name='fn'), 
#       keras.metrics.BinaryAccuracy(name='accuracy'),
#       keras.metrics.Precision(name='precision'),
#       keras.metrics.Recall(name='recall'),
#       keras.metrics.AUC(name='auc'),
#       keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
# ]
