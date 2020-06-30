import pandas as pd
import csv
import pickle


import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, f1_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn import preprocessing
import imblearn.over_sampling
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import fbeta_score, classification_report
from sklearn.model_selection import cross_val_score

with open('my_df.pkl', 'rb') as f:
    df = pickle.load(f)


#Move All-Star to end on pickled data
cols = list(df.columns)
df = df[cols[0:-2] + [cols[-1]] + [cols[-2]]]

X=df.iloc[:, 5:-5]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# setup for the ratio argument of RandomOverSampler initialization

n_pos = np.sum(y_train == 1)
n_neg = np.sum(y_train == 0)
ratio = {1: n_pos * 5, 0: n_neg}

# randomly oversample positive samples: create 5x as many
ROS = imblearn.over_sampling.RandomOverSampler(sampling_strategy=ratio, random_state=42)
X_rs, y_rs = ROS.fit_sample(X_train, y_train)

def Accuracy_RS(a):
    a.fit(X_rs, y_rs)
    pred=a.predict(X_test)
    score = accuracy_score(y_test, pred)
    return score
print(Accuracy_RS(rf))

rf = RandomForestClassifier()
rf.fit(X_rs, y_rs)
pred_rf = rf.predict(X_test)

y_pred_rf = (rf.predict_proba(X_test)[:,1]>0.25)

#below you can see the various metrix tested on the Random Forest Model
print(Accuracy_RS(rf))
print("Threshold of 0.25:")
print("Precision: {:6.4f},   Recall: {:6.4f}".format(precision_score(y_test, y_pred_rf),
                                                     recall_score(y_test, y_pred_rf)))


print('Random Forest Regression on Oversampled Train Data; Test F1: %.3f, Test AUC: %.3f' % \
      (f1_score(y_test, rf.predict(X_test)), roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])))

print(cross_val_score(rf, X_test, y_test, cv=2))

#My confusion matrix to find the correct threshold
def make_confusion_matrix(model, threshold=0.25):
    y_predict = (model.predict_proba(X_test)[:, 1] >= threshold)
    as_confusion = confusion_matrix(y_test, y_predict)
    plt.figure(dpi=80)
    sns.heatmap(as_confusion, cmap=plt.cm.Blues, annot=True, square=True, fmt='d',
           xticklabels=['False', 'True'],
           yticklabels=['False', 'True']);
    plt.xlabel('prediction')
    plt.ylabel('actual')
make_confusion_matrix(rf, threshold=0.25)

#Feature Importance
features = list(X.columns)
importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print(importance)

''' Below is the Predictions of the model at work'''

def final_predictor(name, year):
    line = df.loc[(df['playerID']==name) & (df['yearID']==year),:].index.values
    final = rf.predict(X)
    return final[line]

#Below is a sample test case
print(final_predictor('aaronha01', 1976))

#My main test case was the 1999 New York Yankees
df_yanks = df.loc[(df['teamID']=='NYA') & (df['yearID']==1999), :]
df_yanks['Predicted'] = np.vectorize(final_predictor)(df_yanks.playerID, df_yanks.yearID)
df_yanks

#Two players, one that should have been on the All-Star team and one likely one that should not have been
pd.options.display.max_columns = None
df_yanks.loc[(df['playerID']=='oneilpa01') | (df['playerID']=='knoblch01')]



