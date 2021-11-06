import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",header = None)
data.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                'Alcalinity of ash', 'Magnesium', 'Total phenols',
                'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                'Proline']

X = data.iloc[:,1:]
y = data.iloc[:,0]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)

feat_labels = data.columns[1:]
forest = RandomForestClassifier(n_estimators=500,random_state=1)

forest.fit(X_train,y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))

plt.title("Feature importance")
plt.bar(range(X_train.shape[1]),importances[indices],align="center")
plt.xticks(range(X_train.shape[1]),feat_labels[indices],rotation=90)
plt.xlim([-1,X_train.shape[1]])
plt.tight_layout()
plt.show()