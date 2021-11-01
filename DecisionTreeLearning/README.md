# Decision Tree Classifier
---

> Decision trees can build complex decision boundaries by dividing the feature space into rectangles.

> Decision tree classifiers are attractive models if we care about interpretability.

*Using the decision algorithm, we start at the tree root and split the data on the feature that results in the largest `information gain (IG)`*

*In an iterative process, we can then repeat this splitting procedure at each child node until the leaves are pure.This means that the training examples at each node all belong to the same class.*

---
### Maximizing IG (information gain):

> In order to split the nodes at the most informative features, we need to define an objective function that we want to optimize via the tree learning algorithm.

* The information gain is simply the difference between the impurity of the parent node and the sum of the child node impurities.
* The lower the impurities of the child nodes, the larger the information gain.

**the three impurity measures or splitting criteria that are commonly used in binary decision trees are `Gini impurity`,`entropy`,`classification error`.**

> * The entropy criterion attempts to maximize the mutual information in the tree.
> * The gini impurity can be understood as the criterion to minimize the probability of misclassification.
> * *similar to entropy, the gini impurity is maximal if the classes are perfectly mixed.*
> * the classification error criterion is useful for pruning but not recommended for growing a decision tree, since it is less sensitive to changes in the class probabilities of the nodes.

