# MachineLearning-Algorithms
---
## The perceptron:

The whole idea behind the Rosenblatt's threshold perceptron model is to use a reductionist approach to mimic how a single neuron in the brain works.
A preceptron model is a slightly tweaked version of the MCP model.
Here, the neurons are also called Linear threshold unit(LTU).


**The perceptron algorithm :**
1. Initialize the weights to 0 or small random numbers
2. For each training example 

    -Compute the output value 
    
    -update the weights
---
## Mccullloch-Pitt's Neuron model:

This model imitates the functionality of a biological neuron.Thus also called Artificial Neuron.
An artificial neuron accepts binary inputs and produces a binary output based on a certain thershold value which can be adjusted, this can be mainly used in classification problems.

---
## Adaptive Linear Neuron model:

This algorithm is interesting because it illustrates the key concepts of defining and minimizing continous cost functions.
the difference between perceptron and adaline is that the weights are updated based on a linear activation rather than a unit step function like in perceptron.

---
## Gradient Descent:

> used to optimize and minimizing the cost function by using ***The sum of squared errors(SSE)***

### Stochastic Gradient Descent:
> The way of minimizing a cost function by taking a step in the opposite direction of a cost gradient that is calculated from the whole training dataset this is why this approach often referred as batch gradient descent
> A popular alternative to the batch gradient descent algorithm is stochastic gradient descent(SGD).
> Here we update the weights incrementally for each training example.

### Minibatch Gradient Descent:
> A compromise between gradient descent and SGD is so called minibatch learning.It can be understood as applying batch gradient descent to smaller subsets of the training data.
> Convergence is reached faster via mini-batches because of the more frequent weight updates.

---

