import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

import pytorch_tabnet
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score

file_name = './combined_data/51features_RF.csv'
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
    #unique, counts = np.unique(y, return_counts=True)
    counts = counts / len(y)
    print(dict(zip(unique, counts)))

    train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size=0.2, shuffle=True)

    return train_X, train_Y, test_X, test_Y

train_X, train_Y, test_X, test_Y = get_data(file_name)

clf = TabNetClassifier()  #TabNetRegressor()

clf.fit(
  train_X, train_Y,
  eval_set=[(test_X, test_Y)], max_epochs=50, eval_metric=['accuracy']
)


preds = clf.predict(train_X)
preds2 = clf.predict(test_X)

# overall accuray
print(np.count_nonzero(preds == train_Y) / len(preds)) # training 
print(np.count_nonzero(preds2 == test_Y) / len(preds2)) # testing