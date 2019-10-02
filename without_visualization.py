import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


dataset = load_boston()

features = pd.DataFrame(dataset.data, columns=dataset.feature_names)
features['MEDV'] = dataset.target

min_max_scaler = preprocessing.MinMaxScaler()

column_sels = ['LSTAT', 'PTRATIO', 'RM', 'DIS']

# Keeping only new features that we want.
new_features = features.loc[:, column_sels]

x = pd.DataFrame(data=min_max_scaler.fit_transform(new_features), columns=column_sels)
y = features['MEDV']

# removing skewness from data through log transform..
y = np.log1p(y)
for i in x.columns:
    if np.abs(x[i].skew()) > 0.3:
        x[i] = np.log1p(x[i])

# spliting data into training set and test set.
train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.3)

model = LinearRegression()
model.fit(train_X, train_y)
prediction = model.predict(test_X)

# In regression we calculate r2 score.
score = r2_score(prediction, test_y)


print(score)
