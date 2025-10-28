# eval_compare.py
# Compare two segmentation models and (optionally) SAM on the same 5 samples.
# Saves a grid: | RGB | GT | Model1 | Model2 | SAM |

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

# ---- Optional SAM imports (safe) ----
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    _HAS_SAM = True
except Exception:
    _HAS_SAM = False


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

    # PyTorch 2.6+: weights_only defaults to True; allow full state dicts
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)

    # handle DP prefix if present
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
    samples = []  # (rgb_uint8, gt_color, pred_color)

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
            rgb = (denorm(img) * 255).transpose(1,2,0).astype(np.uint8)
            gt = loader.dataset.decode_target(targets[0].astype(np.uint8)).astype(np.uint8)
            pr = loader.dataset.decode_target(preds[0].astype(np.uint8)).astype(np.uint8)
            samples.append((rgb, gt, pr))

    return metrics.get_results(), samples


def sam_overlay_rgb(rgb_uint8, sam_masks, alpha=0.55):
    """
    Overlays all SAM masks onto an RGB image (HxWx3 uint8).
    """
    if len(sam_masks) == 0:
        return rgb_uint8

    overlay = rgb_uint8.copy().astype(np.float32)
    H, W, _ = overlay.shape

    rng = np.random.RandomState(123)  # deterministic colors
    for m in sam_masks:
        seg = m.get("segmentation", None)
        if seg is None or seg.shape[:2] != (H, W):
            continue
        color = rng.randint(0, 256, size=(1, 1, 3), dtype=np.uint8).astype(np.float32)
        mask = seg.astype(bool)
        overlay[mask] = alpha * color + (1 - alpha) * overlay[mask]

    return np.clip(overlay, 0, 255).astype(np.uint8)


@torch.no_grad()
def run_sam_on_picks(val_loader, pick_ids, sam_generator):
    """
    Runs SAM (promptless) on the *same* validation indices used for the model samples.
    NOTE: Set val_batch_size=1 for correct alignment.
    Returns list of overlay images (HxWx3 uint8).
    """
    overlays = []
    denorm = utils.Denormalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    picked = set(pick_ids)

    for i, (images, labels) in enumerate(val_loader):
        if i not in picked:
            continue

        img_t = images[0].detach().cpu().numpy()
        rgb = (denorm(img_t) * 255).transpose(1,2,0).astype(np.uint8)

        masks = sam_generator.generate(rgb)   # SAM expects uint8 RGB
        overlays.append(sam_overlay_rgb(rgb, masks, alpha=0.55))

    return overlays


def save_grid_with_optional_sam(samples_m1, samples_m2, sam_imgs, out_path, title_m1, title_m2):
    """
    samples_m1 / samples_m2: list of (rgb, gt, pred)
    sam_imgs: list of SAM overlay rgb images OR None
    """
    assert len(samples_m1) == len(samples_m2)
    k = len(samples_m1)
    use_sam = sam_imgs is not None and len(sam_imgs) == k

    cols = 5 if use_sam else 4
    fig, axes = plt.subplots(k, cols, figsize=(4*cols, 3*k))
    if k == 1:
        axes = np.expand_dims(axes, 0)

    for r in range(k):
        rgb, gt, pred1 = samples_m1[r]
        _,  _, pred2 = samples_m2[r]

        axes[r,0].imshow(rgb);   axes[r,0].set_title("RGB"); axes[r,0].axis("off")
        axes[r,1].imshow(gt);    axes[r,1].set_title("Ground Truth"); axes[r,1].axis("off")
        axes[r,2].imshow(pred1); axes[r,2].set_title(f"{title_m1}"); axes[r,2].axis("off")
        axes[r,3].imshow(pred2); axes[r,3].set_title(f"{title_m2}"); axes[r,3].axis("off")

        if use_sam:
            axes[r,4].imshow(sam_imgs[r]); axes[r,4].set_title("SAM"); axes[r,4].axis("off")

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
    # --- SAM options ---
    ap.add_argument("--with_sam", action="store_true", help="Also run SAM on the same samples")
    ap.add_argument("--sam_model_type", type=str, default="vit_h", choices=["vit_h","vit_l","vit_b"])
    ap.add_argument("--sam_ckpt", type=str, default="", help="Path to SAM checkpoint .pth")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    val_dst, val_loader = build_val_loader(
        args.data_root, crop_val=args.crop_val, crop_size=args.crop_size, val_batch_size=args.val_batch_size
    )
    print("Val set size:", len(val_dst))

    # Pick K validated indices (batch index) — keep val_batch_size=1 for alignment
    random.seed(args.seed)
    k = min(args.num_images, len(val_loader))
    pick_ids = sorted(random.sample(range(len(val_loader)), k=k))

    # Load models
    m1 = load_model(args.model1, args.num_classes, args.output_stride, args.ckpt1, device)
    m2 = load_model(args.model2, args.num_classes, args.output_stride, args.ckpt2, device)

    # Evaluate & collect samples
    score1, samples1 = evaluate_and_collect(m1, val_loader, device, args.num_classes, pick_ids)
    score2, samples2 = evaluate_and_collect(m2, val_loader, device, args.num_classes, pick_ids)

    # Print metrics
    def fmt(s):
        return f"mIoU={s.get('Mean IoU',0):.4f}, Acc={s.get('Overall Acc',0):.4f}"
    print(f"[{args.model1}] {fmt(score1)}")
    print(f"[{args.model2}] {fmt(score2)}")

    # Align RGB/GT rows across models (use RGB,GT from samples1)
    aligned1, aligned2 = [], []
    for (rgb, gt, p1), (_, _, p2) in zip(samples1, samples2):
        aligned1.append((rgb, gt, p1))
        aligned2.append((rgb, gt, p2))

    # Optional: SAM
    sam_overlays = None
    if args.with_sam:
        if not _HAS_SAM:
            print("[WARN] segment_anything not available; skipping SAM.")
        elif not args.sam_ckpt:
            print("[WARN] --with_sam set but --sam_ckpt is empty; skipping SAM.")
        else:
            print("[SAM] Loading checkpoint:", args.sam_ckpt)
            sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_ckpt)
            sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
            sam.eval()
            sam_gen = SamAutomaticMaskGenerator(sam)
            if args.val_batch_size != 1:
                print("[WARN] For SAM overlays, set --val_batch_size=1 to align indices.")
            sam_overlays = run_sam_on_picks(val_loader, pick_ids, sam_gen)

    # Save grid (4 or 5 columns depending on SAM)
    save_grid_with_optional_sam(aligned1, aligned2, sam_overlays, args.out_grid, args.model1, args.model2)


if __name__ == "__main__":
    main()

# !python eval_compare.py \
#   --data_root ./datasets/data \
#   --model1 deeplabv3plus_resnet50 \
#   --ckpt1 checkpoints/best_deeplabv3plus_resnet50_VOC_os16.pth \
#   --model2 deeplabv3_resnet50 \
#   --ckpt2 checkpoints/best_deeplabv3_resnet50_VOC_os16.pth \
#   --val_batch_size 1 \
#   --num_images 5 \
#   --with_sam \
#   --sam_model_type vit_h \
#   --sam_ckpt /content/cvhw3/hw3_semantic_segmentation/datasets/HOME/weights/sam_vit_h_4b8939.pth \
#   --out_grid results/compare_grid_with_sam.png


# !python eval_compare.py \
#   --data_root ./datasets/data \
#   --model1 deeplabv3plus_resnet50 \
#   --ckpt1 checkpoints/best_deeplabv3plus_resnet50_VOC_os16.pth \
#   --model2 deeplabv3_resnet50 \
#   --ckpt2 checkpoints/best_deeplabv3_resnet50_VOC_os16.pth \
#   --val_batch_size 1 \
#   --num_images 5 \
#   --out_grid results/compare_grid.png
