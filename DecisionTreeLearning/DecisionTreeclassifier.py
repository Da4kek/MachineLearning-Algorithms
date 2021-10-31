import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import classification_report


iris = load_iris()

X = iris.data[:,[2,3]]
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
std = StandardScaler()
std.fit(X_train)
std.fit(X_test)
X_train_std = std.transform(X_train)
X_test_std = std.transform(X_test)

treemodel = DecisionTreeClassifier(criterion="entropy",max_depth=4,random_state=1)
treemodel.fit(X_train,y_train)

X_combined = np.vstack((X_train,X_test))
y_combined = np.hstack((y_train,y_test))


def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ("s", "x", "o", "*", "^")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8,
                    c=colors[idx], marker=markers[idx], label=cl, edgecolor="black")
    plt.xlabel("sepal length")
    plt.ylabel("petal length")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

plot_decision_regions(X_combined,y_combined,classifier=treemodel)

prediction = treemodel.predict(X_test)
print(classification_report(prediction,y_test))