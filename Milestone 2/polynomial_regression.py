import mpmath
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.discrete.discrete_model
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from Pre_processing import *
from sklearn.metrics import r2_score, confusion_matrix, classification_report
import time
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm




pd.options.mode.chained_assignment = None  # default='warn'
#Load players data

desired_width=320
pd.set_option('display.width',desired_width)
pd.set_option('display.max_columns',12)

data1=pd.read_csv('movies-revenue-classification.csv')
data2=pd.read_csv('movie-director.csv')
data3=pd.read_csv('movie-voice-actors.csv')



#print(data3.isnull().sum())


data1.dropna(inplace=True)



data2.rename(columns={"name":"movie_title"},inplace=True)
data3.rename(columns={"movie":"movie_title"},inplace=True)
data3.pop('character')

data3['voice-actor']=data3['voice-actor'].replace('None',np.nan)
data3['voice-actor']=data3['voice-actor'].replace('Unknown',np.nan)
#print(data3)
data3.dropna(inplace=True)




# aggregation_func={'voice-actor': 'sum'}
# data3=data3.groupby(data3['movie_title']).aggregate(aggregation_func)
# print(data3)




output1=pd.merge(data2,data3,on='movie_title',how='outer')
data=pd.merge(data1,output1,on='movie_title',how='outer')


#print(data)

column_to_move=data.pop('MovieSuccessLevel')
data.insert(6,'MovieSuccessLevel',column_to_move)
# data['revenue']=data['MovieSuccessLevel'].str.replace(',','')
# data['MovieSuccessLevel']=data['MovieSuccessLevel'].str.replace('\$','')
# data['MovieSuccessLevel']=data['MovieSuccessLevel'].astype(float)


data=data.dropna(subset=['MovieSuccessLevel'],how='all')

data.fillna('unknown',inplace=True)

X=data.iloc[:,:-1] #Features

Y=data['MovieSuccessLevel'] #Label


movie_data = data.iloc[:,:]
cols=('movie_title','genre','MPAA_rating','director','release_date','voice-actor')
X=Feature_Encoder(X,cols)
#print(X)







# X=featureScaling(X,0,1)

movie_data=Feature_Encoder(movie_data,cols)


corr = movie_data.corr()
#Top 50% Correlation training features with the Value
#print(movie_data)

# top_feature = corr.index[abs(corr['MovieSuccessLevel'])>0.2]
# #Correlation plot
# plt.subplots(figsize=(12, 8))
# top_corr = movie_data[top_feature].corr()
# sns.heatmap(top_corr, annot=True)
# plt.show()
# top_feature = top_feature.delete(-1)
# print(top_feature)
# X=X[top_feature]


#X=featureScaling(X,0,1)




start=time.time()
#Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,shuffle=True,random_state=30)
#logit_model=sm.Logit(y_train,X_train)
#result=logit_model.fit()
#print(result.summary2())

logreg=LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
print("Accuracy",logreg.score(X_test,y_test))














# # Standardize features by removing mean and scaling to unit variance:
# scaler = StandardScaler()
# scaler.fit(X_train)
#
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
#
# # Use the KNN classifier to fit data:
# classifier = KNeighborsClassifier(n_neighbors=5)
# classifier.fit(X_train, y_train)
#
# # Predict y data with classifier:
# y_predict = classifier.predict(X_test)
#
# # Print results:
# print("here")
# print(confusion_matrix(y_test, y_predict))
# print(classification_report(y_test, y_predict))