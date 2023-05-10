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

from sklearn.preprocessing import LabelEncoder




pd.options.mode.chained_assignment = None  # default='warn'
# #Load players data
#
desired_width=320
pd.set_option('display.width',desired_width)
pd.set_option('display.max_columns',12)

data1=pd.read_csv('movies-revenue-classification.csv')
data2=pd.read_csv('movie-director.csv')
data3=pd.read_csv('movie-voice-actors.csv')



print(data3.isnull().sum())


data1.dropna(inplace=True)



data2.rename(columns={"name":"movie_title"},inplace=True)
data3.rename(columns={"movie":"movie_title"},inplace=True)
data3.pop('character')

data3['voice-actor']=data3['voice-actor'].replace('None',np.nan)
data3['voice-actor']=data3['voice-actor'].replace('Unknown',np.nan)

#print(data3)
data3.dropna(inplace=True)
f=0
aclist=[]



j=0
counter=0
data3=data3.sort_values('movie_title')
unique=data3['movie_title'].unique()
index=0;
foundFirst=False
for i in range(len(data3)):
    if(data3['movie_title'].iloc[i]==unique[f]):
        st=data3['voice-actor'].iloc[i]
        aclist.append(st)
    else:
        data3['voice-actor'].iloc[index]=aclist
        index=i
        f+=1
        aclist=[]
print(data3)

data3=data3.drop_duplicates(subset=['movie_title'],keep='first')



# aggregation_func={'voice-actor': 'sum'}
# data3=data3.groupby(data3['movie_title']).aggregate(aggregation_func)





output1=pd.merge(data2,data3,on='movie_title',how='outer')
data=pd.merge(data1,output1,on='movie_title',how='outer')

data = data.dropna(subset=['MovieSuccessLevel'], how='all')
print(data)
ia=imdb.IMDb()

for i in range(len(data)):
    try:
        name=data['movie_title'].iloc[i]
        search = ia.search_movie(name)
        id = search[0].movieID
        movie = ia.get_movie(id)
        data.loc[i,['director']]=str(movie['director'][0])
        print(str(movie['director'][0]))
    except:
        continue
column_to_move=data.pop('MovieSuccessLevel')
data['MovieSuccessLevel']=column_to_move
#data.to_csv('Full Data2.csv')
data.to_csv("FullDataClassification.csv")