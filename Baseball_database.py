import pandas as pd
import csv
import pickle
import numpy as np

path = 'baseballdatabank-master/core/'

####MAKE INTO DATAFRAME CALLED df_batting
df_batting=pd.read_csv(path + 'Batting.csv')

df_batting.fillna(0, inplace=True)

#Limit years to All-Star era, make sure a batter had a decent amount of at bats and games
train_df = df_batting[df_batting.AB > 0]
train2 = train_df[train_df['yearID'] >=1933]
df_batting_train = train2[train2.G >=35]


#df_all is short for for df all-star
df_all = pd.read_csv(path + 'AllstarFull.csv')
df_all.info()

#There was one player at the end of the data with all Nan values that this eliminated
df_all = df_all.iloc[:-1, : ]
df_all.tail(3)

df_all[(df_all.gameID.isnull())].head()

#Got rid of 1945 as there was no game and elimated the 2nd games in 1959-1961
df_all = df_all[df_all['yearID'] != 1945.0]
df_all = df_all[df_all['gameNum']<=1.0]
df_all.isnull().sum()

#Here I switched years to integers and turned NANs in starting position to zero
df_all['yearID']=df_all['yearID'].astype(int)
df_all['startingPos'].fillna(0, inplace=True)

#Used this player to test df out
df_all.loc[df_all['playerID']=='foxne01']

#Drop gameID and add column of 1s so when merged, non-All-Stars will get 0s
df_all = df_all.drop(['gameID'], axis=1)
df_all['All-Star'] = np.array([1]*5144)

#The big merge!!!
df = pd.merge(df_batting_train, df_all, on=['playerID', 'yearID', 'teamID', 'lgID'], how='left')
df = df.drop(['gameNum'], axis=1)
df.info()


#Again testing df with this a different player
df.loc[df['playerID']=='applilu01']

#Change NANs
df['All-Star'].fillna(0, inplace=True)
df.sample(2)

#Change types of all the unneccesary floats to integers
df[['RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF', 'GIDP', 'All-Star']] = df[
    ['RBI', 'SB','CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF', 'GIDP', 'All-Star']].astype(int)


#Determine Batting Average
df['BA']= (df.H /df.AB)

def good_BA(x):
    if x >= 0.275:
        return 1
    else:
        return 0
df['Good_BA'] = df.BA.apply(good_BA)

#Move All-Star to end
cols = list(df.columns)
df = df[cols[0:-3] + cols[-2:] + [cols[-3]]]
df.sample()

df.to_csv('My_All_Star_DF.csv')

df.to_pickle("my_df.pkl")

with open('my_df.pkl', 'wb') as fh:
    pickle.dump(df, fh)

with open('my_df.pkl', 'rb') as f:
    df = pickle.load(f)

df.sample()

#Move All-Star to end on pickled data
cols = list(df.columns)
df = df[cols[0:-2] + [cols[-1]] + [cols[-2]]]

df['All-Star'].value_counts()

#Rate of being an All-Star is 10.96%