import mpmath
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from Pre_processing import *
from sklearn.metrics import r2_score
import time




pd.options.mode.chained_assignment = None  # default='warn'
#Load players data

desired_width=320
pd.set_option('display.width',desired_width)
pd.set_option('display.max_columns',12)

data1=pd.read_csv('movies-revenue.csv')
data2=pd.read_csv('movie-director.csv')
data3=pd.read_csv('movie-voice-actors.csv')



print(data3.isnull().sum())


data1.dropna(inplace=True)



data2.rename(columns={"name":"movie_title"},inplace=True)
data3.rename(columns={"movie":"movie_title"},inplace=True)
data3.pop('character')

data3['voice-actor']=data3['voice-actor'].replace('None',np.nan)
data3['voice-actor']=data3['voice-actor'].replace('Unknown',np.nan)
print(data3)
data3.dropna(inplace=True)




# aggregation_func={'voice-actor': 'sum'}
# data3=data3.groupby(data3['movie_title']).aggregate(aggregation_func)
# print(data3)




output1=pd.merge(data2,data3,on='movie_title',how='outer')
data=pd.merge(data1,output1,on='movie_title',how='outer')


print(data)

column_to_move=data.pop('revenue')
data.insert(6,'revenue',column_to_move)
data['revenue']=data['revenue'].str.replace(',','')
data['revenue']=data['revenue'].str.replace('\$','')
data['revenue']=data['revenue'].astype(float)


data=data.dropna(subset=['revenue'],how='all')

data.fillna('unknown',inplace=True)

X=data.iloc[:,:-1] #Features

Y=data['revenue'] #Label


movie_data = data.iloc[:,:]
cols=('movie_title','genre','MPAA_rating','director','release_date','voice-actor')
X=Feature_Encoder(X,cols)
print(X)







# X=featureScaling(X,0,1)

movie_data=Feature_Encoder(movie_data,cols)


corr = movie_data.corr()
#Top 50% Correlation training features with the Value
print(movie_data)

top_feature = corr.index[abs(corr['revenue'])>0.2]
#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = movie_data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
top_feature = top_feature.delete(-1)
print(top_feature)
X=X[top_feature]


X=featureScaling(X,0,1)




start=time.time()
#Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30,shuffle=True,random_state=10)

#X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.50,shuffle=True)

poly_features = PolynomialFeatures(degree=4)

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)
ypred=poly_model.predict(poly_features.transform(X_test))

# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(X_test))
prediction1 = poly_model.predict(poly_features.fit_transform(X_train))


stop=time.time()
print('Co-efficient of linear regression',poly_model.coef_)
print('Intercept of linear regression model',poly_model.intercept_)
print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))
print('Mean Square Error of training', metrics.mean_squared_error(y_train, prediction1))
print('Root Square Error of test',mpmath.sqrt(metrics.mean_squared_error(y_test, prediction)))
print('Accuracy of test', r2_score(y_test, prediction))
print('Accuracy of training', r2_score(y_train, prediction1))
print('time taken: ',stop-start)
print('-------------------------------------'
      'Multilinear Regression')

cls = linear_model.LinearRegression()
cls.fit(X,Y)
predictionlinear= cls.predict(X_test)
print('Co-efficient of linear regression',cls.coef_)
print('Intercept of linear regression model',cls.intercept_)
print('Mean Square Error of linear', metrics.mean_squared_error(np.asarray(y_test), predictionlinear))
print('Accuracy', r2_score(y_test, predictionlinear))


true_revenue_value=np.asarray(y_test)[50]
predicted_revenue_value=prediction[50]
print('True revenue for the movie is : ' + str(true_revenue_value))
print('Predicted revenue for the movie is : ' + str(predicted_revenue_value))

