import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",header=None)
data.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                'Alcalinity of ash', 'Magnesium', 'Total phenols',
                'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                'Proline']
print("class labels: {}".format(np.unique(data['Class label'])))

X,y = data.iloc[:,1:].values , data.iloc[:,0].values 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)

std = StandardScaler()
X_train_std = std.fit_transform(X_train)
X_test_std = std.fit_transform(X_test)

lr = LogisticRegression(penalty="l2",C=1.0,solver="liblinear",multi_class="ovr")
lr.fit(X_train_std,y_train)
print("training accuracy: ",lr.score(X_train_std,y_train))
print("test accuracy: ",lr.score(X_test_std,y_test))

fig = plt.figure()
ax=plt.subplot(111)
colors=['blue','green','red','cyan','magenta','yellow','black','pink','lightgreen','lightblue','gray','indigo','orange']
weights,params = [],[]
for c in np.arange(-4.,6.):
    lr = LogisticRegression(penalty="l1",C=10.**c,solver="liblinear",multi_class="ovr")
    lr.fit(X_train_std,y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
weights = np.array(weights)

for column , color in zip(range(weights.shape[1]),colors):
    plt.plot(params,weights[:,column],label=data.columns[column+1],color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)
plt.show()
