import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from preconditioners.datasets import generate_data


sigma2 = 0.1
n=1000
# dataset = generate_data('quadratic', n=n, d=2, regime='autoregressive', ro=.5, r1=1, sigma2=sigma2)
dataset = generate_data('MLP', n=n, d=2, regime='autoregressive', ro=.5, sigma2=sigma2, num_layers=3, hidden_channels=32)

x, y = dataset
print(x)
print(y)
plt.scatter(x[:,0], x[:,1], c=y)
plt.show()