import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np


# load the dataset
data = pd.read_csv('network-traffic.csv')
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data=data.dropna()

# preprocess the data
X = data.drop([' Label'], axis=1)
X=X.astype('float')
Y = data[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
clf = IsolationForest()
clf.fit(X)

# make predictions
y_pred = clf.predict(X)

# print the results
print('Number of anomalies:', len(y_pred[y_pred == -1]))
print('Number of normal samples:', len(y_pred[y_pred == 1]))
