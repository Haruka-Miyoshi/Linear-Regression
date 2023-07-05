# %%
import os
import torch
from lr import LR

if not os.path.exists('./data'):
    os.mkdir('./data')

x = torch.randn(100, 1) * 10
y = x + torch.randn(100, 1) * 3

torch.save(x, './data/x.txt')
torch.save(y, './data/y.txt')

torch.manual_seed(1)
model=LR(1,1)

model.fit(x, y, mode=True)

#[w,b] = model.parameters()
#def get_params():
#    return (w[0][0].item(), b[0].item())
# %%
