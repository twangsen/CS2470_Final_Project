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

df = pd.read_csv("combined_data/51features_RF.csv")
train_df = df.copy()
y = train_df['OHAREC'].values 
y = y-1
train_df.drop(['OHAREC'],axis=1,inplace=True)

# heatmap for correlation matrix
# A rather neat looking zero correlation heatmap.
# However since the number of features are not clearly visible (due to display size) a distribution plot of the correlation matrix will show how values are intertwined with each other.
corr_matrix = train_df.corr().abs()
sns.heatmap(corr_matrix)


# A sharp spike and nothing else, this proves that the columns in the dataset are uncorrelated with each other.
dist_features = corr_matrix.values.flatten()
sns.distplot(dist_features, color="Red", label="train")

model = ExtraTreesClassifier()
model.fit(train_df,y)
#print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization

#%% [markdown]
# Plotting the significant features

#%%
feat_importances = pd.Series(model.feature_importances_, index=train_df.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

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


########
# Data preprocessing
#%%
#--Feature selection
features = [X for X in train_df.columns.values.tolist()]
#--Scaling data and store scaling values
scaler = preprocessing.StandardScaler().fit(train_df[features].values)
X = scaler.transform(train_df[features].values)

#--training & test stratified split
X_train, X_valid, y_train, y_valid = train_test_split(X, y,stratify=y, test_size=0.10)

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

model.summary()

# Compile
# Note: make sure you are using the GPU for this!
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit
r1 = model.fit(X_train,y_train, validation_data=(X_valid, y_valid), epochs=20)