import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np

from PIL import Image


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    # color map for each category
    cmap = voc_cmap()

    def __init__(self,
                 root,
                 image_set='train',
                 transform=None):

        self.root = os.path.expanduser(root)
        self.transform = transform
        
        self.image_set = image_set
        base_dir = "VOCtrainval_11-May-2012/VOCdevkit/VOC2012"
        #"/home/cropthecoder/Downloads/18794_HW3_F25/hw3_semantic_segmentation/datasets/data/VOCtrainval_11-May-2012"
        
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' Please check your data download.')
        
        mask_dir = os.path.join(voc_root, 'SegmentationClass')
        splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.

            What you should do
            1. read the image (jpg) as an PIL image in RGB format.
            2. read the mask (png) as a single-channel PIL image.
            3. perform the necessary transforms on image & mask.
        """
        # TODO Problem 1.1
        # =================================================
        #read image as PIL image in RGB format
        img_path = self.images[index]
        image = Image.open(img_path).convert('RGB')

        #read mask as single-channel PIL image
        mask_path = self.masks[index]
        #mask_pil = Image.open(mask_path) 
        #mask = np.array(mask_pil, dtype=np.uint8)  # (H,W), values {0..20,255}
        mask = Image.open(mask_path).convert('L')

        #perform necessary transforms on image & mask
        if self.transform is not None:
            image, mask = self.transform(image, mask)
            

        return image, mask


        #raise NotImplementedError

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image for visualization, using the color map"""

    
        mask = mask.astype(np.int64)

        # Clamp unknowns to 255
        mask[(mask < 0) | (mask > 255)] = 255

        color_mask = cls.cmap[mask]  #(H,W,3), uint8 because cmap is uint8

        ignore = (mask == 255)
        if ignore.any():
            color_mask[ignore] = np.array([0, 0, 0], dtype=np.uint8)
        return color_mask


        # TODO Problem 1.1
        # =================================================
        #raise NotImplementedError
        # =================================================


#     @classmethod
#     def main_test(cls, root, image_set="train", show=True):
#         """
#         Quick smoke test for the dataset & DataLoader using only the code in this file.
#         - Instantiates the dataset (no transform)
#         - Loads one sample, prints shapes/dtypes
#         - Decodes and (optionally) visualizes the mask
#         - Runs a single DataLoader iteration with batch_size=2
#         """
#         import os
#         import matplotlib.pyplot as plt
#         from torch.utils.data import DataLoader

    
#         base_dir = "VOCdevkit/VOC2012"
#         voc_root = os.path.join(os.path.expanduser(root), base_dir)
#         assert os.path.isdir(voc_root), "VOC root not found at: {}".format(voc_root)

     
#         ds = cls(root=root, image_set=image_set, transform=None)
#         print("Dataset loaded. image_set,", image_set,  "samples=", len(ds))

   
#         img, mask = ds[0]   
#         print("[Sample 0] image type={}, size={} (W,H), mode={}".format(type(img), img.size, img.mode))
#         print("[Sample 0] mask  type={}, shape={}, dtype={}".format(type(mask), getattr(mask, 'shape', None), getattr(mask, 'dtype', None)))


#         if hasattr(mask, "dtype"):
#             import numpy as np
#             uniq = np.unique(mask)
#             print("[Sample 0] unique labels (first 30):", uniq[:30], "", "count=", len(uniq))


#         color_mask = cls.decode_target(mask)
#         print("[Decode] color_mask shape={}, dtype={}".format(color_mask.shape, color_mask.dtype))

#         if show:
#             plt.figure(figsize=(10,4))
#             plt.subplot(1,2,1); plt.title("Image"); plt.imshow(img); plt.axis("off")
#             plt.subplot(1,2,2); plt.title("Mask (colored)"); plt.imshow(color_mask); plt.axis("off")
#             plt.tight_layout(); plt.show()

        
#         loader = DataLoader(ds, batch_size=2, shuffle=True,
#                             num_workers=0,  
#                             collate_fn=lambda batch: list(zip(*batch)))
#         images, masks = next(iter(loader))  
#         print("got batch: images={}, masks={}".format(len(images), len(masks)))
#         print("image[0] type={}, size={}".format(type(images[0]), images[0].size))
#         if hasattr(masks[0], 'shape'):
#             print("[Loader] mask[0]  shape={}, dtype={}".format(masks[0].shape, masks[0].dtype))

#         print("works")


# if __name__ == "__main__":
#     ROOT = "/home/cropthecoder/Downloads/18794_HW3_F25/hw3_semantic_segmentation/datasets/data/VOCtrainval_11-May-2012"
#     #"/datasets/data/VOCtrainval_11-May-2012"
#     VOCSegmentation.main_test(root=ROOT, image_set="train", show=True)

# #hw3_semantic_segmentation/datasets/data/VOCtrainval_11-May-2012
#/home/cropthecoder/Downloads/18794_HW3_F25/hw3_semantic_segmentation/datasets/data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012

import math
import numpy as np
import matplotlib.pyplot as plt


VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

def make_one_per_class_montage(
    root,
    image_set="train",
    save_path="voc_one_per_class_pairs.png",
    include_background=False,
    max_cols=10,           
    figsize_per_img=3.0   
):
  
    ds = VOCSegmentation(root=root, image_set=image_set, transform=None)

    wanted_ids = list(range(21)) if include_background else list(range(1, 21))
    found = {cid: None for cid in wanted_ids}  #(PIL image, color_mask, idx)
    remaining = set(wanted_ids)

    print(f"Scanning {len(ds)} samples to find one example per class...")
    for i in range(len(ds)):
        img, mask = ds[i]  # img: PIL.Image, mask: np.ndarray (H,W)
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask, dtype=np.uint8)

        uniq = np.unique(mask)
        uniq = uniq[(uniq != 255)]  # drop ignore label

     
        to_claim = [cid for cid in uniq if cid in remaining]
        for cid in to_claim:
            color_mask = VOCSegmentation.decode_target(mask)  # (H,W,3) uint8
            found[cid] = (img, color_mask, i)
            remaining.discard(cid)

        if not remaining:
            break

    missing = [cid for cid in wanted_ids if found[cid] is None]
    if missing:
        print("WARNING: Could not find examples for class ids:", missing,
              "(", [VOC_CLASSES[c] for c in missing], ")")

   
    pairs = []
    for cid in wanted_ids:
        if found[cid] is None:
            continue
        img, color_mask, idx = found[cid]
        title = f"{VOC_CLASSES[cid]} (id={cid}, idx={idx})"
        pairs.append((img, color_mask, title))

    if not pairs:
        raise RuntimeError("No class examples found. Check your dataset root and split.")

    
    tiles = []
    titles = []
    for (img, cmask, t) in pairs:
        tiles.append(img)           
        tiles.append(cmask)         
        titles.append(t)            
        titles.append("")           

    n_tiles = len(tiles)
    cols = max_cols
    rows = math.ceil(n_tiles / cols)

  
    fig = plt.figure(figsize=(figsize_per_img * cols, figsize_per_img * rows))
    for k, tile in enumerate(tiles, start=1):
        ax = plt.subplot(rows, cols, k)
        if isinstance(tile, np.ndarray):
            ax.imshow(tile)
        else:
            ax.imshow(tile)  
        ax.set_axis_off()
        if titles[k - 1]:
            ax.set_title(titles[k - 1], fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved montage: {save_path}")
    plt.show()


if __name__ == "__main__":
    ROOT = "/home/cropthecoder/Downloads/18794_HW3_F25/hw3_semantic_segmentation/datasets/data/VOCtrainval_11-May-2012"

    make_one_per_class_montage(
        root=ROOT,
        image_set="train",
        save_path="voc_one_per_class_pairs.png",
        include_background=False, 
        max_cols=10,
        figsize_per_img=3.0
    )