import torch
from network.deeplab import ASPP
m = ASPP(in_channels=2048, atrous_rates=[6,12,18], out_channels=256)
x = torch.randn(2, 2048, 32, 32)
y = m(x)
print(y.shape)  #[2, 256, 32, 32]