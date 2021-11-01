# K-Nearest Neighbors
---
> knns are particularly interesting because it is fundamentally different from the learning algorithms that we have discussed so far.

> KNN is a typical example of a laze learner. It is called "lazy" not because of its apparent simpliciy, but because it doesn't learn a discriminative function from the training data but memorizes the training dataset instead.

**The KNN algorithm itself is fairly straightforward and can be summarized by the following steps:**
1. Choose the number of k and a distance metric
2. Find the k-nearest neighbors of the data record that we want to classify
3. Assign the class label by majority vote.

### Dimensionality:

> It is important to mention that KNN is very susceptible to overfitting due to the **curse of dimensionality**.

> The curse of dimensionality describes the phenomenon where the feature space becomes increasingle sparse for an increasing number of dimensions of a fixed size training dataset.