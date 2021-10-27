import matplotlib.pyplot as plt
import numpy as np
from Sigmoid import sigmoid

def cost_1(z):
    return - np.log(sigmoid(z))

def cost_2(z):
    return - np.log(1-sigmoid(z))

z =np.arange(-10,10,0.1)
phi_z = sigmoid(z)

c1 = [cost_1(x) for x in z]
plt.plot(phi_z,c1,label="J(w) if y=1",color="red")

c2 = [cost_2(x) for x in z]
plt.plot(phi_z,c2,label="J(w) if y=0",linestyle = "--",color="green")

plt.ylim(0.0,5.1)
plt.xlim([0,1])
plt.xlabel("sigmoid of z")
plt.ylabel("J(w)")
plt.legend(loc="best")
plt.tight_layout()
plt.show()