import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# 1) Instantiate dataset
root = "/datasets/data"   
tfm  = JointBasicToTensor(size=(320,320))
ds   = VOCSegmentation(root=root, image_set="train", transform=tmf)

print("Samples:", len(ds))
img0, mask0 = ds[0]
print("Single sample -> image:", img0.shape, img0.dtype, "mask:", mask0.shape, mask0.dtype)

# 2) Quick value checks
u = torch.unique(mask0)
print("Unique labels in sample[0]:", u.tolist()[:30], "…", "(count:", len(u), ")")

# 3) Build a DataLoader and fetch one batch
loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2, drop_last=True)
imgs, masks = next(iter(loader))
print("Batch imgs:", imgs.shape, imgs.dtype)   # (B,3,H,W), float32
print("Batch masks:", masks.shape, masks.dtype)  # (B,H,W), int64 (long)

# 4) Visualize one pair to ensure alignment
color = VOCSegmentation.decode_target(masks[0])  # ndarray (H,W,3) uint8
plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.title("Image"); plt.imshow(imgs[0].permute(1,2,0)); plt.axis("off")
plt.subplot(1,2,2); plt.title("Mask (colored)"); plt.imshow(color); plt.axis("off")
plt.tight_layout(); plt.show()
