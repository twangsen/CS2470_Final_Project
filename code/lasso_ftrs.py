import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

def get_data(input_file):
    data = pd.read_csv(input_file)
    #print(data.head())
    features = data.columns[:-1]
    data = data.values
    #print(data)
    # Standardization
    scaler = preprocessing.StandardScaler().fit(data)
    data = scaler.transform(data)
    # Scaling features to a range (0,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(data)
    x = data[:,:-1]
    y = data[:,-1]
    train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size=0.3, random_state=50)
    return features,train_X, train_Y, test_X, test_Y

file_name = 'combined_data/51features_RF.csv'
features, train_X, train_Y, test_X, test_Y = get_data(file_name)
print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape )
print(features)

pipeline = Pipeline([                     
                     ('model',Lasso())
])
search = GridSearchCV(pipeline,
                      {'model__alpha':np.arange(0.1,10,0.1)},
                      cv = 5, scoring="neg_mean_squared_error",verbose=3
                      )
search.fit(train_X,train_Y)

coefficients = search.best_estimator_.named_steps['model'].coef_
importance = np.abs(coefficients)
imp_ftrs = np.array(features)[importance > 0]
