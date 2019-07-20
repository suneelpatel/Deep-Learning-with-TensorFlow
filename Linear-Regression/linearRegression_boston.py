import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score
#from sklearn.cross_validation import

boston = load_boston()


print(boston.keys())
print(boston.data.shape)


print(boston.feature_names)


print(boston.DESCR)


bos = pd.DataFrame(boston.data)

print(bos.head())

bos.columns = boston.feature_names

bos['PRICE'] = boston.target

print(bos.head())

#print(bos.describe())

X = bos.drop('PRICE', axis = 1)

Y = bos['PRICE']


X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.33, random_state = 5)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

lm = LinearRegression()

lm.fit(X_train, Y_train)


Y_train_pred = lm.predict(X_train)

Y_test_pred = lm.predict(X_test)



df=pd.DataFrame(Y_test_pred,Y_test)
print(df)


mse = mean_squared_error(Y_test, Y_test_pred)
print(mse)

plt.scatter(Y_train_pred, Y_train_pred - Y_train,c='blue',marker='o',label='Training data')
plt.scatter(Y_test_pred, Y_test_pred - Y_test,c='lightgreen',marker='s',label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc= 'upper left')
plt.hlines(y=0,xmin=0,xmax=50)
plt.plot()
plt.show()







