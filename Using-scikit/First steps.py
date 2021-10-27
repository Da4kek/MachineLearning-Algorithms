import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target
print("Class labels: ",np.unique(y))
print("\n")

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)
print("samples in y: ",len(y))
print("samples in y_train: ",len(y_train))
print("samples in y_test: ",len(y_test))
print("\n")

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
print("X_train: ",X_train[0:6])
print("X_train_std: ",X_train_std[0:6])
print("\n")

ppn = Perceptron(eta0=0.01,random_state=1)
ppn.fit(X_train_std,y_train)

y_pred = ppn.predict(X_test_std)
print("Missclassified examples: %d" % (y_test != y_pred).sum())
print("Accuracy: ",np.mean(y_pred ==y_test))
print("\n")

ppn1 = Perceptron(eta0=0.0001,max_iter=1000)
ppn1.fit(X_train,y_train)

y_pred_1 = ppn1.predict(X_test)
print("Missclassified samples: %d" % (y_test != y_pred_1).sum())
print("Accuracy: ",np.mean(y_test == y_pred_1))
print("\n")