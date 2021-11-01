# Random Forests Classifier
---

> Ensemble methods have gained huge popularity in applications of machine learning during the last decade due to their good classification performance and robustness towards overfitting.

* *The random forest is a decision tree based algorithm, which is known for its good scalability and ease of use.*
* *The idea behind a random forest is to average multiple (deep) decision trees that individually suffer from high variance to build a more robust model that has better generalization performance and is less susceptible to overfitting.*

**The random forest algorithm can be summarized in four simple steps:**
> * Draw a random `bootstrap` sample of size n(randomly choose n examples from the training dataset with replacement).
> * Grow a decision tree from the bootstrap sample. At each node:
> 1. Randomly select d features without replacement.
> 2. Split the node using the feature that provides the best split according to the objective function, for instance, maximizing the information gain.
> * Repeat the steps 1-2 k times.
> * Aggregate the prediction by each tree to assign the class label by majority vote.