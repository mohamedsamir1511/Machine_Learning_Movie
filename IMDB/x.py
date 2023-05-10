import datetime

import pandas as pd

date="16-Feb-01"

print(datetime.datetime.strptime(date, '%d-%b-%y').strftime('%Y'))
print(dir(pd.DatetimeIndex))
data=pd.read_csv('Full Data2.csv')
data['release_date'] = pd.to_datetime(data["release_date"],format='%d-%b-%y')
data['year'] = pd.DatetimeIndex(data["release_date"]).year
data['year'] = data['year'].astype(str)
data['year']=data['year'].str[2:]

print(data)