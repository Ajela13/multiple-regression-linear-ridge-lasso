import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.preprocessing import StandardScaler


data=pd.read_csv('CarPrice_Assignment_Simple.csv')

#Building data understanding 
print('shape of dataframe is:', data.shape)
data.info()
data['car_ID']=data['car_ID'].astype('object')

data.describe()





#build data understanding 
data['drivewheel'].value_counts()
#avg price for drivewheel
data.groupby('drivewheel')['price'].agg('mean').round(decimals=2)


#all numeric (float and int) variables in the dataset
data_numeric=data.select_dtypes(include=['Float64','int64'])
data_numeric.head()

#Correlation matrix
cor=data_numeric.corr()
cor.round(2)

plt.figure(figsize=(16,8))
sns.heatmap(cor.round(1),cmap='YlGnBu',annot=True)
plt.show()

#Data preparation
data.columns
X=data.loc[:,['drivewheel','carlength','carwidth','carheight','curbweight','horsepower']]
Y=data['price']

#Dummy variable creation

data_categorical=X.select_dtypes(include=['object'])
data_categorical.head()

#checking unique values in drivewheel column
data_categorical['drivewheel'].unique()
data_dummies=pd.get_dummies(data_categorical,drop_first=True)
data_dummies.head()

#Drop categorical variables
X=X.drop(list(data_categorical.columns),axis=1)
#concar dummy variables with X
X=pd.concat([X,data_dummies],axis=1)
X.head()

#split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y,train_size = 0.8, test_size = 0.2,random_state=100)

X_train.shape
X_train.head()

X_test.shape
X_test.head()

#Model
lr = LinearRegression()

#Fit model
lr.fit(X_train, y_train)

#predict
#prediction = lr.predict(X_test)

#actual
actual = y_test

train_score_lr = lr.score(X_train, y_train)
test_score_lr = lr.score(X_test, y_test)

print("The train score for lr model is {}".format(train_score_lr))
print("The test score for lr model is {}".format(test_score_lr))


#Ridge Regression Model
ridgeReg = Ridge(alpha=10)

ridgeReg.fit(X_train,y_train)

#train and test scorefor ridge regression
train_score_ridge = ridgeReg.score(X_train, y_train)
test_score_ridge = ridgeReg.score(X_test, y_test)

print("\nRidge Model............................................\n")
print("The train score for ridge model is {}".format(train_score_ridge))
print("The test score for ridge model is {}".format(test_score_ridge))

#Lasso regression model
print("\nLasso Model............................................\n")
lasso = Lasso(alpha = 10)
lasso.fit(X_train,y_train)
train_score_ls =lasso.score(X_train,y_train)
test_score_ls =lasso.score(X_test,y_test)

print("The train score for ls model is {}".format(train_score_ls))
print("The test score for ls model is {}".format(test_score_ls))


#Using the linear CV model
from sklearn.linear_model import RidgeCV

#Lasso Cross validation
ridge_cv = RidgeCV(alphas = [0.0001, 0.001,0.01, 0.1, 1, 10,20,30]).fit(X_train, y_train)

#score
print("The train score for ridge model is {}".format(ridge_cv.score(X_train, y_train)))
print("The train score for ridge model is {}".format(ridge_cv.score(X_test, y_test)))

#Using the linear CV model
from sklearn.linear_model import LassoCV

#Lasso Cross validation
lasso_cv = LassoCV(alphas = [0.0001, 0.001,0.01, 0.1, 1, 10,20,30,40,50], random_state=0).fit(X_train, y_train)


#score
print(lasso_cv.score(X_train, y_train))
print(lasso_cv.score(X_test, y_test))



#plot size
plt.figure(figsize = (10, 10))
#add plot for ridge regression
plt.plot(X.columns,ridge_cv.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Ridge; $\alpha = 10$',zorder=7)

#add plot for lasso regression
plt.plot(lasso_cv.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'lasso; $\alpha = grid$')

#add plot for linear model
plt.plot(X.columns,lr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Linear Regression')

#rotate axis
plt.xticks(rotation = 90)
plt.legend()
plt.title("Comparison plot of Ridge, Lasso and Linear regression model")
plt.show()


# Plot Initial Data
plt.figure(figsize=(10, 6))
plt.scatter(np.arange(len(Y)),Y, color='blue', label='Original Data')
plt.xlabel('Price')
plt.ylabel('predicted Price')

# Plot Regression Results
plt.plot( lr.predict(X), color='green', label='Linear Regression')
plt.plot( ridge_cv.predict(X), color='red', label='Ridge Regression')
plt.plot( lasso_cv.predict(X), color='orange', label='Lasso Regression')
plt.legend()
plt.show()



from sklearn.metrics import r2_score,mean_squared_error


print(r2_score(Y, lr.predict(X)))
print(r2_score(Y, ridge_cv.predict(X)))
print(r2_score(Y, lasso_cv.predict(X)))


print(mean_squared_error(Y, lr.predict(X)))
print(mean_squared_error(Y, ridge_cv.predict(X)))
print(mean_squared_error(Y, lasso_cv.predict(X)))
