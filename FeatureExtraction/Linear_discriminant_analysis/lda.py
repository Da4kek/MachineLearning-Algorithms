from pandas.io.stata import precision_loss_doc
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('https://archive.ics.uci.edu/ml/'
                   'machine-learning-databases/wine/wine.data',
                   header=None)

data.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',

                'Alcalinity of ash', 'Magnesium', 'Total phenols',
                'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                'Color intensity', 'Hue',
                'OD280/OD315 of diluted wines', 'Proline']

X = data.iloc[:,1:]
y = data.iloc[:,0]

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
std = StandardScaler()
X_train_std = std.fit_transform(X_train)
X_test_std = std.fit_transform(X_test)

np.set_printoptions(precision = 4)
mean_vecs = []
for label in range(1,4):
    mean_vecs.append(np.mean(X_train_std[y_train==label],axis=0))
    print(f"MV {label} : {mean_vecs[label-1]}")

d=13 
S_W = np.zeros((d,d))
for label,mv in zip(range(1,4),mean_vecs):
    class_scatter = np.zeros((d,d))
    for row in X_train_std[y_train == label]:
        row,mv = row.reshape(d,1), mv.reshape(d,1)
        class_scatter += (row-mv).dot((row-mv).T)
    S_W += class_scatter
print(f"\nWithin class scatter matrix: {S_W.shape[0]}x{S_W.shape[1]}")

print(f"\nClass label distribution: {np.bincount(y_train)[1:]}")

d = 13 
S_W = np.zeros((d,d))
for label, mv in zip(range(1,4),mean_vecs):
    class_scatter = np.cov(X_train_std[y_train==label].T)
    S_W += class_scatter
print(f"\nScaled within-class scatter matrix: {S_W.shape[0]}x{S_W.shape[1]}")

mean_overall = np.mean(X_train_std,axis=0)
d=13
S_B = np.zeros((d,d))
for i,mean_vec in enumerate(mean_vecs):
    n = X_train_std[y_train==i+1,:].shape[0]
    mean_vec = mean_vec.reshape(d,1)
    mean_overall = mean_overall.reshape(d,1)
    S_B += n * (mean_vec-mean_overall).dot((mean_vec-mean_overall).T)
print(f"\nBetween-class scatter matrix: {S_B.shape[0]}x{S_B.shape[1]}")

eigen_vals,eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs,key=lambda k: k[0],reverse=True)
print("Eigen values in descending order: \n")
for eigen_val in eigen_pairs:
    print(eigen_val[0])

w = np.hstack((eigen_pairs[0][1][:,np.newaxis].real, 
               eigen_pairs[1][1][:,np.newaxis].real))

X_train_lda = X_train_std.dot(w)
colors = ['r','b','g']
markers = ['x','o','*']
for l,c,m in zip(np.unique(y_train),colors,markers):
    plt.scatter(X_train_lda[y_train==l,0],
                X_train_lda[y_train==l,1]*(-1),c=c,label=l,marker=m)
plt.xlabel("LD 1")
plt.ylabel("LD 2")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
