
#modified from main.py

import os, random, argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import network
from datasets import VOCSegmentation
from utils import ext_transforms as et
from metrics import StreamSegMetrics
import utils

def build_val_loader(data_root, crop_val=False, crop_size=513, val_batch_size=1):
    if crop_val:
        val_tf = et.ExtCompose([
            et.ExtResize(crop_size),
            et.ExtCenterCrop(crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    else:
        val_tf = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    val_dst = VOCSegmentation(root=data_root, image_set='val', transform=val_tf)
    from torch.utils import data
    val_loader = data.DataLoader(val_dst, batch_size=val_batch_size, shuffle=False, num_workers=2)
    return val_dst, val_loader

def load_model(model_name, num_classes, output_stride, ckpt_path, device):
    model = network.modeling.__dict__[model_name](num_classes=num_classes, output_stride=output_stride)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
  
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        from collections import OrderedDict
        cleaned = OrderedDict((k.replace("module.",""), v) for k,v in state.items())
        model.load_state_dict(cleaned, strict=True)
    model = nn.DataParallel(model).to(device)
    model.eval()
    return model

@torch.no_grad()
def evaluate_and_collect(model, loader, device, num_classes, pick_ids):
    metrics = StreamSegMetrics(num_classes)
    denorm = utils.Denormalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    samples = []  

    for i, (images, labels) in enumerate(loader):
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        out = model(images)
        logits = out["out"] if isinstance(out, dict) else out
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        targets = labels.cpu().numpy()

        metrics.update(targets, preds)

        if i in pick_ids:
            img = images[0].detach().cpu().numpy()
            img = (denorm(img) * 255).transpose(1,2,0).astype(np.uint8)
            gt = loader.dataset.decode_target(targets[0].astype(np.uint8)).astype(np.uint8)
            pr = loader.dataset.decode_target(preds[0].astype(np.uint8)).astype(np.uint8)
            samples.append((img, gt, pr))

    return metrics.get_results(), samples

def save_four_column_grid(samples_m1, samples_m2, out_path, title_m1, title_m2):
    assert len(samples_m1) == len(samples_m2)
    k = len(samples_m1)
    fig, axes = plt.subplots(k, 4, figsize=(16, 3*k))
    if k == 1:
        axes = np.expand_dims(axes, 0)

    for r in range(k):
        rgb, gt, pred1 = samples_m1[r]
        _,  _, pred2 = samples_m2[r]

        axes[r,0].imshow(rgb);  axes[r,0].set_title("RGB Image"); axes[r,0].axis("off")
        axes[r,1].imshow(gt);   axes[r,1].set_title("Ground Truth"); axes[r,1].axis("off")
        axes[r,2].imshow(pred1);axes[r,2].set_title(f"{title_m1} Prediction"); axes[r,2].axis("off")
        axes[r,3].imshow(pred2);axes[r,3].set_title(f"{title_m2} Prediction"); axes[r,3].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"[Saved] {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./datasets/data")
    ap.add_argument("--num_classes", type=int, default=21)
    ap.add_argument("--output_stride", type=int, default=16, choices=[8,16])
    ap.add_argument("--model1", type=str, default="deeplabv3plus_resnet50")
    ap.add_argument("--model2", type=str, default="deeplabv3_resnet50")
    ap.add_argument("--ckpt1", type=str, required=True)
    ap.add_argument("--ckpt2", type=str, required=True)
    ap.add_argument("--gpu_id", type=str, default="0")
    ap.add_argument("--num_images", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--crop_val", action="store_true", default=False)
    ap.add_argument("--crop_size", type=int, default=513)
    ap.add_argument("--val_batch_size", type=int, default=1)
    ap.add_argument("--out_grid", type=str, default="results/compare_grid.png")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    val_dst, val_loader = build_val_loader(
        args.data_root, crop_val=args.crop_val, crop_size=args.crop_size, val_batch_size=args.val_batch_size
    )
    print("Val set size:", len(val_dst))

    #pick 5 by random
    random.seed(args.seed)
    pick_ids = sorted(random.sample(range(len(val_loader)), k=min(args.num_images, len(val_loader))))

   
    m1 = load_model(args.model1, args.num_classes, args.output_stride, args.ckpt1, device)
    m2 = load_model(args.model2, args.num_classes, args.output_stride, args.ckpt2, device)

    score1, samples1 = evaluate_and_collect(m1, val_loader, device, args.num_classes, pick_ids)
    score2, samples2 = evaluate_and_collect(m2, val_loader, device, args.num_classes, pick_ids)


    def fmt(s):
        return f"mIoU={s.get('Mean IoU',0):.4f}, Acc={s.get('Overall Acc',0):.4f}"
    print(f"[{args.model1}] {fmt(score1)}")
    print(f"[{args.model2}] {fmt(score2)}")


    aligned1 = []
    aligned2 = []
    for (rgb, gt, p1), (_, _, p2) in zip(samples1, samples2):
        aligned1.append((rgb, gt, p1))
        aligned2.append((rgb, gt, p2))

    save_four_column_grid(aligned1, aligned2, args.out_grid, args.model1, args.model2)

if __name__ == "__main__":
    main()
