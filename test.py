import random

import torch.nn as nn
import torch
random.seed(10)
m = nn.ConvTranspose2d(1, 1, (5,45), stride=(7, 1), padding=(5, 1))
n = nn.Conv2d(1,1,(5,45), stride=(7, 1), padding=(5, 1))
a= torch.randint(1,100,(1,1,20,86)).float()
b=m(a)
c=n(b)
print(a,b,b.shape,c,c.shape)