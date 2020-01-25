import scipy
import pandas
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math

data = pandas.read_csv('numerical_chevron_data.csv')
# print(data.head())
# print(data.columns.values.tolist())

X = data.drop(['rate_of_penetration', 'Unnamed: 0', 'segment_id', 'wellbore_chev_no_id', 'area_id', 'formation_id', 'bit_model_id', 'min_depth',  'max_depth'], axis = 1)
y = data['rate_of_penetration']

# print(X.head())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from keras import models
from keras import layers


# linear stack of layers
model = models.Sequential()
model.add(layers.Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(1, kernel_initializer='normal'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_test, epochs=200, verbose=0)