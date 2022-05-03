import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from tab2img.converter import Tab2Img

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

df = pd.read_csv("../combined_data/51features_RF.csv")
# df = df.iloc[: , 1:]
y = df["OHAREC"].values
y = y - 1
X = df.drop(['OHAREC'], axis=1)

# # split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20, random_state=42)

# print(X_train.shape)
# print(X_test.shape)

# transform the data
scaler = preprocessing.StandardScaler().fit(X_train)
X_scaled_train = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)
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
# print(train_images.shape, test_images.shape)
# print(y_train)

# # train the model
model = keras.Sequential([
    layers.Conv2D(32, (2, 2), activation='relu', padding='same'),
    layers.BatchNormalization(),
    # layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(32, (2, 2), activation='relu', padding='same'),
    layers.BatchNormalization(),
    # layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(32, (2, 2), activation='relu', padding='same'),
    layers.BatchNormalization(),
    # layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(64, (2, 2), activation='relu', padding='same'),
    layers.BatchNormalization(),
    # layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    # layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    # layers.MaxPooling2D((2, 2), padding='same'),
    layers.Flatten(),
    # layers.Dropout(0.2),
    layers.Dense(512, activation='relu'),
    # layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    # layers.Dropout(0.2),
    layers.Dense(4, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(train_images, y_train, batch_size = 100, epochs=20, validation_data=(test_images, y_test))


