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

data = pandas.read_csv('numerical_chevron_data.csv')
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

data['patrick'] = data['surface_rpm']*data['surface_weight_on_bit'] /data['drillbit_size']
data['pat'] = data['surface_rpm']*data['surface_weight_on_bit']**3


correlations = data[data.columns[1:]].corr()['rate_of_penetration'][:]

c_dict = correlations.to_dict()
del c_dict['rate_of_penetration']
print(max(c_dict.values()))
print(min(c_dict.values()))
print(c_dict)




data['sqrt'] = data['surface_weight_on_bit']**0.5

X = data[['sqrt', 'surface_rpm']]
y = data['rate_of_penetration']

# print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = BayesianRidge()

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

print(math.sqrt(mean_squared_error(y_test, y_predict)))

x_feat = 'surface_weight_on_bit'
y_feat = 'surface_rpm'


x = data[x_feat]*data[y_feat]**2
y = data['rate_of_penetration']
z = data['drillbit_size'] ** 2

plt.xlabel(x_feat)
plt.ylabel(y_feat)
plt.title(x_feat + ' vs. ' + y_feat + ' impact on Rate of Penetration')

# x = data['surface_weight_on_bit']**0.5
# y = data['surface_rpm']
# z = data['rate_of_penetration']


cm = plt.cm.get_cmap('inferno')
gauss = plt.scatter(x, y, c=z, cmap=cm)
plt.colorbar(gauss)
plt.show()


# depth_mean = x.describe()['mean']
# depth_std = x.describe()['std']
# depth_min = x.describe()['min']
# depth_max = x.describe()['max']
# print(depth_mean)
# print(depth_std)
# print(depth_min)
# print(depth_max)
#
#
# gauss_x = np.linspace(depth_min, depth_max, 100)
# gauss_y = norm.pdf(gauss_x, depth_mean, depth_std)
# plt.plot(gauss_x, 4000000*gauss_y)
# plt.show()

