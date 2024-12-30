import torch
import numpy as np
import torch.backends

print(torch.backends.mps.is_available())

# Test two processes
from time import time
# device
device = torch.device('mps')
t1 = time()
it_max = 1000
for i in range(it_max):
    x = torch.rand((2000,2000),device=device)
    y = torch.rand((2000,2000),device=device)
    z = x @ y

t2 = time()

print('With MPS:',(t2-t1)/it_max)

t1 = time()
device = torch.device('cpu')
for i in range(it_max):
    x = torch.rand((2000,2000),device=device)
    y = torch.rand((2000,2000),device=device)
    z = x @ y
t2 = time()

print('With CPU:',(t2-t1)/it_max)