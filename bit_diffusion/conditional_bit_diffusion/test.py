import torch
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
BITS=4
def decimal_to_bits(x, bits = BITS):
    """ expects image tensor ranging from 0 to 1, outputs bit tensor ranging from -1 to 1 """
    device = x.device

    # x = (x * 255).int().clamp(0, 255)

    mask = 2 ** torch.arange(bits - 1, -1, -1, device = device)
    mask = rearrange(mask, 'd -> d 1 1')

    x = rearrange(x, 'b c h w -> b c 1 h w')
    print(mask, mask.shape,x)
    bits = ((x & mask) ).float()
    bits = rearrange(bits, 'b c d h w -> b (c d) h w')
    bits = bits * 2 - 1
    return bits

x=torch.randint(0,15,(1,1,2,2))
print(x)
y=decimal_to_bits(x)
print(y)
# print(y,y.shape,y==raw_decimal_to_bits(x))