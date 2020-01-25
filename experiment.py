from sklearn.naive_bayes import GaussianNB
import scipy
import pandas
from sklearn.linear_model import BayesianRidge
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy import stats

data = pandas.read_csv('numerical_chevron_data.csv')
data = data.drop(['segment_id'], axis = 1)
data = data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]
# print(data.head())
cols = data.columns.values.tolist()

for col1 in cols:
    if col1 != 'segment_id' and col1 != 'rate_of_penetration':
        data[col1 + 'sqrt'] = np.sqrt(data[col1])
        data[col1 +'log'] = np.log(data[col1]+1)
        data[col1 + 'inverse'] = 1/(data[col1]+1)
        data[col1 + ' squared'] = data[col1] ** 2
        data[col1 + ' cubed'] = data[col1] ** 3
    for col2 in cols:
        if col1 != 'segment_id' and col2 != 'segment_id' and col1 != 'rate_of_penetration' and col2 != 'rate_of_penetration':
            data[col1 + ' times '+ col2] = data[col1] * data[col2]
            data[col1 + ' divided ' + col2] = data[col1] / data[col2]
    # print(col)
    # if col != 'segment_id':
    #     data[col + ' squared'] = data[col] ** 2
    #     data[col + ' cubed'] = data[col] ** 3


correlations = data[data.columns[1:]].corr()['rate_of_penetration'][:]

c_dict = correlations.to_dict()
del c_dict['rate_of_penetration']
print(max(c_dict.values()))
print(min(c_dict.values()))
print(c_dict)

x_feat = 'surface_weight_on_bit'
y_feat = 'surface_rpm'


x = (data[x_feat] * data[y_feat]**3)
y = data['rate_of_penetration']
z = data['drillbit_size']


# gauss = plt.scatter(x, y)
cm = plt.cm.get_cmap('inferno')
gauss = plt.scatter(x, y, c=z, cmap=cm)
plt.colorbar(gauss)
plt.xlabel(x_feat + ' * ' + y_feat + 'cubed')
plt.ylabel('rate_of_penetration')
plt.show()

