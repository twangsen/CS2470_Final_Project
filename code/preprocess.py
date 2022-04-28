import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing

df = pd.read_csv('combined_data/30_data_mean.csv')
column_name = df.columns
# Standardization
scaler = preprocessing.StandardScaler().fit(df)
df = scaler.transform(df)
# Scaling features to a range (0,1)
normalizing = preprocessing.MinMaxScaler()
result = normalizing.fit_transform(df)

normalized_df = pd.DataFrame(result,columns=column_name)
normalized_df.to_csv("normalize_data/normalized_30_data_mean.csv",sep='\t',encoding='utf-8') 
