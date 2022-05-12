import pandas as pd
import numpy as np

df = pd.read_csv("combined_data/51features_RF.csv")
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier

import tensorflow as tf
from keras import layers
from keras import backend as K
from keras import regularizers
from keras.constraints import max_norm
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.layers import Input,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D,AveragePooling2D,MaxPooling2D,Dropout

import matplotlib.pyplot as plt
import seaborn as sns

train_df = df.copy()
y = train_df['OHAREC'].values 
train_df.drop(['OHAREC'],axis=1,inplace=True)

y = y-1
corr_matrix = train_df.corr().abs()

class1 = df[y == 0].drop(['OHAREC'], axis=1)
class2 = df[y == 1].drop(['OHAREC'], axis=1)
class3 = df[y == 2].drop(['OHAREC'], axis=1)
class4 = df[y == 3].drop(['OHAREC'], axis=1)

#%% [markdown]
# A sharp spike and nothing else, this proves that the columns in the dataset are uncorrelated with each other.
# Lets extract the significant features from the dataset, this can be achieved using random forest classifiers feature extrator routine

#%%
model = ExtraTreesClassifier()
model.fit(train_df,y)
#print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization

#%% [markdown]
# Plotting the significant features

#%%
feat_importances = pd.Series(model.feature_importances_, index=train_df.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.title("feature importance")
plt.savefig('ftr_importance.png',dpi=300)
plt.show()

#%% [markdown]
# Using argsort in descending order and picking out the first 5 elements in the important features list

#%%
feat_imp = model.feature_importances_
feat_imp_desc = np.argsort(feat_importances)[::-1][0:5]
feat_imp_desc

#%%
feature_names = train_df.columns.values


#%%
top = train_df.loc[:, feature_names[feat_imp_desc]]

top['target'] = y
#%%
sns.pairplot(top,diag_kind='hist' )

#%%
#--Feature selection
features = [X for X in train_df.columns.values.tolist()]

#--Scaling data and store scaling values
scaler = preprocessing.StandardScaler().fit(train_df[features].values)

X = scaler.transform(train_df[features].values)

X_class1 = scaler.transform(class1)
X_class2 = scaler.transform(class2)
X_class3 = scaler.transform(class3)
X_class4 = scaler.transform(class4)

#--training & test stratified split
X_train, X_valid, y_train, y_valid = train_test_split(X, y,stratify=y, test_size=0.10)
X_train.shape

# reshape dataset
X_train=np.reshape(X_train,(23064,7,7,1))
X_valid=np.reshape(X_valid,(2563,7,7,1))
print(X_train.shape)
print(X_valid.shape)

# Build the model using the functional API
i = Input(shape=X_train[0].shape)
# x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
# x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
# x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
#x = MaxPooling2D((2, 2))(x)
# x = Dropout(0.2)(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
#x = MaxPooling2D((2, 2))(x)
# x = Dropout(0.2)(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
#x = MaxPooling2D((2, 2))(x)
# x = Dropout(0.2)(x)

# x = GlobalMaxPooling2D()(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(4, activation='softmax')(x)

model = Model(i, x)

# Compile
# Note: make sure you are using the GPU for this!
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit
r1 = model.fit(X_train,y_train, validation_data=(X_valid, y_valid), epochs=16)

X_class1 = np.reshape(X_class1,(X_class1.shape[0],7,7,1))
X_class2 = np.reshape(X_class2,(X_class2.shape[0],7,7,1))
X_class3 = np.reshape(X_class3,(X_class3.shape[0],7,7,1))
X_class4 = np.reshape(X_class4,(X_class4.shape[0],7,7,1))

class1_pred = model.predict(X_class1)
print("class1:", np.sum(np.argmax(class1_pred, axis = 1) == 0) / len(class1_pred))

class2_pred = model.predict(X_class2)
print("class2:", np.sum(np.argmax(class2_pred, axis = 1) == 1) / len(class2_pred))

class3_pred = model.predict(X_class3)
print("class3:", np.sum(np.argmax(class3_pred, axis = 1) == 2) / len(class3_pred))

class4_pred = model.predict(X_class4)
print("class4:", np.sum(np.argmax(class4_pred, axis = 1) == 3) / len(class4_pred))