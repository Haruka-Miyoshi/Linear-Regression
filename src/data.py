import os
import torch
import matplotlib.pyplot as plt

if not os.path.exists('./data'):
    os.mkdir('./data')

if not os.path.exists('./figs'):
    os.mkdir('./figs')

x = torch.randn(100, 1) * 10
y = x + torch.randn(100, 1) * 3

torch.save(x, './data/x.txt')
torch.save(y, './data/y.txt')

plt.plot(x.numpy(), y.numpy(), "o")
plt.ylabel("Y")
plt.xlabel("X")
plt.savefig('./figs/data.png')