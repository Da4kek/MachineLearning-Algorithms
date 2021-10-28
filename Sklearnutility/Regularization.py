import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data[:,[2,3]]
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
std = StandardScaler()
std.fit(X_train)
std.fit(X_test)
X_train_std = std.transform(X_train)
X_test_std = std.transform(X_test)

weights , params = [] , []
for c in np.arange(-5,5):
    lr = LogisticRegression(C = 10.**c,random_state=0,solver="lbfgs",multi_class="ovr")
    lr.fit(X_train_std,y_train)
    weights.append(lr.coef_[1])
    params.append(10.0**c)
weights = np.array(weights)
plt.plot(params,weights[:,0],label="petal length")
plt.plot(params,weights[:,1],linestyle="--",label="petal width")
plt.ylabel("weight coefficient")
plt.xlabel("C")
plt.legend(loc="best")
plt.xscale("log")
plt.show()