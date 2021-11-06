import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from SequentialBackwardSelection import SBS 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",header=None)
data.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                'Alcalinity of ash', 'Magnesium', 'Total phenols',
                'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                'Proline']
X = data.iloc[:,1:]
y = data.iloc[:,0]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)

std = StandardScaler()
X_train_std = std.fit_transform(X_train)
X_test_std = std.fit_transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
sbs = SBS(knn,k_features=1)
sbs.fit(X_train_std,y_train)
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat,sbs.scores_,marker="o")
plt.ylim([0.7,1.02])
plt.ylabel("accuracy")
plt.xlabel("number of features")
plt.grid()
plt.tight_layout()
plt.show()

k3 = list(sbs.subsets_[9])
print(data.columns[1:][k3])

knn.fit(X_train_std,y_train)
print("Training accuracy: ",knn.score(X_train_std,y_train))
print("test accuracy: ",knn.score(X_test_std,y_test))
print("\n")
knn.fit(X_train_std[: ,k3],y_train)
print("Training accuracy: ",knn.score(X_train_std[:,k3],y_train))
print("test accuracy: ",knn.score(X_test_std[:,k3],y_test)) 