import scipy
import pandas
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math

data = pandas.read_csv('numerical_chevron_data.csv')
# print(data.head())
print(data.columns.values.tolist())

X = data.drop(['rate_of_penetration', 'Unnamed: 0', 'segment_id', 'wellbore_chev_no_id', 'area_id', 'formation_id', 'bit_model_id', 'min_depth',  'max_depth'], axis = 1)
y = data['rate_of_penetration']

print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = Ridge(alpha = 20)

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

print(math.sqrt(mean_squared_error(y_test, y_predict)))





# model =