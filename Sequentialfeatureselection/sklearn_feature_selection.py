import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",header = None)

data.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                'Alcalinity of ash', 'Magnesium', 'Total phenols',
                'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                'Proline']


X = data.iloc[:,1:]
y = data.iloc[:,0]

X_train, X_test, y_train , y_test = train_test_split(X,y,test_size = 0.3,random_state = 1)

forest = RandomForestClassifier(n_estimators=500,random_state=1)
forest.fit(X_train,y_train)

sfm = SelectFromModel(forest,threshold=0.1,prefit=True)
X_selected = sfm.transform(X_train)
important = forest.feature_importances_
indices = np.argsort(important)[::-1]


print("number of features that meet this threshold criterion : ",X_selected.shape[1])
print(data.columns[1:][indices][:5])