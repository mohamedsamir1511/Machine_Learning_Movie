import mpmath
import numpy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import time
import imdb;
from Pre_processing import *
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def preprocessfiles(f1,f2,f3):
    data1 = pd.read_csv(f1)
    data2 = pd.read_csv(f2)
    data3 = pd.read_csv(f3)


    data1.dropna(inplace=True)

    data2.rename(columns={"name": "movie_title"}, inplace=True)
    data3.rename(columns={"movie": "movie_title"}, inplace=True)
    data3.pop('character')

    data3['voice-actor'] = data3['voice-actor'].replace('None', np.nan)
    data3['voice-actor'] = data3['voice-actor'].replace('Unknown', np.nan)

    # print(data3)
    data3.dropna(inplace=True)
    f = 0
    aclist = []

    j = 0
    counter = 0
    data3 = data3.sort_values('movie_title')
    unique = data3['movie_title'].unique()
    index = 0;
    foundFirst = False
    for i in range(len(data3)):
        if (data3['movie_title'].iloc[i] == unique[f]):
            st = data3['voice-actor'].iloc[i]
            aclist.append(st)
        else:
            data3['voice-actor'].iloc[index] = aclist
            index = i
            f += 1
            aclist = []

    data3 = data3.drop_duplicates(subset=['movie_title'], keep='first')

    # print(data3)

    # aggregation_func={'voice-actor': 'sum'}
    # data3=data3.groupby(data3['movie_title']).aggregate(aggregation_func)

    output1 = pd.merge(data2, data3, on='movie_title', how='outer')
    data = pd.merge(data1, output1, on='movie_title', how='outer')

    data = data.dropna(subset=['revenue'], how='all')
    ia = imdb.IMDb()

    # for i in range(len(data)):
    #     try:
    #         name = data['movie_title'].iloc[i]
    #         search = ia.search_movie(name)
    #         id = search[0].movieID
    #         movie = ia.get_movie(id)
    #         data.loc[i, ['director']] = str(movie['director'][0])
    #     except:
    #         continue
    data = data.iloc[:, 1:]
    data['release_date'] = pd.to_datetime(data["release_date"], format='%d-%b-%y')
    data['year'] = pd.DatetimeIndex(data["release_date"]).year
    data['year'] = data['year'].astype(str)
    data['year'] = data['year'].str[2:]
    data.pop('release_date')
    data.fillna('unknown', inplace=True)
    data['revenue'] = data['revenue'].str.replace(',', '')
    data['revenue'] = data['revenue'].str.replace('\$', '')
    data['revenue'] = data['revenue'].astype(float)
    column_to_move = data.pop('revenue')
    data.pop('voice-actor')
    data['revenue'] = column_to_move
    return data




pd.options.mode.chained_assignment = None  # default='warn'
# #Load players data
#
desired_width=320
pd.set_option('display.width',desired_width)
pd.set_option('display.max_columns',12)
data=preprocessfiles('movies-revenue-train.csv','movie-director-train.csv','movie-voice-actors-train.csv')
print(data)
#data = data.iloc[: , 1:]

#data.to_csv('Full Data2.csv')


print(data)

X=data.iloc[:,:-1] #Features
#X.to_csv('X.csv')
Y=data['revenue'] #Label
print(X)
le=LabelEncoder()
label=le.fit_transform(X['director'])
X.drop("director", axis=1, inplace=True)
X["director"]=label

label=le.fit_transform(X['MPAA_rating'])
X.drop("MPAA_rating", axis=1, inplace=True)
X["MPAA_rating"]=label


label=le.fit_transform(X['genre'])
X.drop("genre", axis=1, inplace=True)


X["genre"]=label
label=le.fit_transform(X['year'])
X.drop("year", axis=1, inplace=True)
X["year"]=label
print(X)


movie_data = data.iloc[:,:]
cols=('genre','MPAA_rating','director','year')
# X=Feature_Encoder(X,cols)
# print(X)
#
#
#
#
#
#
movie_data=Feature_Encoder(movie_data,cols)

print('stop')
corr = movie_data.corr()
#Top 50% Correlation training features with the Value
print(movie_data)

top_feature = corr.index[abs(corr['revenue'])>0.1]
#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = movie_data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
print(top_feature)
top_feature = top_feature.delete(-1)

X=X[top_feature]

print(X)
#X=featureScaling(X,0,1000)

print("he")
start=time.time()
#Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,shuffle=True,random_state=10)

#X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.50,shuffle=True)

poly_features = PolynomialFeatures(degree=6)

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression(fit_intercept=True)
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
print('r2of test: ', r2_score(y_test, prediction))
print('r2 of training: ', r2_score(y_train, prediction1))
#print('accuracy: ', plt.clf.score(X_test,y_test))
print('time taken: ',stop-start)
print('-------------------------------------')



