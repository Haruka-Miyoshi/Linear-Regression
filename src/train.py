import os
import torch
import numpy as np
from lr import LR
import matplotlib.pyplot as plt

if not os.path.exists('./figs'):
    os.mkdir('./figs')

x=torch.load('./data/x.txt')
y=torch.load('./data/y.txt')

model=LR(1,1)
model.fit(x, y, mode=True)
w1, b = model.get_params()

x_h=np.array([-30, 30])
y_h=w1*x_h+b
plt.plot(x_h, y_h, "r")
plt.title('fitting')
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(x.numpy(), y.numpy())
plt.savefig('./figs/fit.png')
plt.show()