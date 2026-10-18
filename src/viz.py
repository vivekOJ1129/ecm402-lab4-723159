"""All figures. Every function writes a PNG and returns its path."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch
import torch.nn.functional as F
from torchvision import transforms
from data import SOBEL_X, SOBEL_Y, GAUSS, CLASSES, augment_ops
from models import hand_table

DPI = 130


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def conv2d_true(img, k):
    """True 2-D convolution of Eq. (1): F.conv2d is cross-correlation, so flip the kernel first."""
    k = torch.flip(k, dims=(0, 1))
    return F.conv2d(img[None, None].float(), k[None, None], padding=1)[0, 0]


def fig_filtering(img, path):
    img = img if img.dim() == 2 else img[0]
    items = [("Original", img), ("Sobel-x (vertical edges)", conv2d_true(img, SOBEL_X)),
             ("Sobel-y (horizontal edges)", conv2d_true(img, SOBEL_Y)), ("Gaussian blur 3x3", conv2d_true(img, GAUSS))]
    fig, ax = plt.subplots(1, 4, figsize=(11, 3))
    for a, (t, im) in zip(ax, items):
        a.imshow(im, cmap="gray"); a.set_title(t, fontsize=9); a.axis("off")
    return _save(fig, path)


def fig_augmentation(pil_ds, path, n=6):
    tf = transforms.Compose(augment_ops())
    torch.manual_seed(0)
    fig, ax = plt.subplots(3, n, figsize=(1.6 * n, 5))
    for j in range(n):
        im, y = pil_ds[j]
        ax[0, j].imshow(im, cmap="gray"); ax[0, j].set_title(CLASSES[y], fontsize=7)
        for i in (1, 2):
            ax[i, j].imshow(tf(im), cmap="gray")
        for i in range(3):
            ax[i, j].axis("off")
    fig.suptitle("Row 1: original | rows 2-3: random augmentations (h-flip + small affine)", fontsize=8)
    return _save(fig, path)


def fig_history(h, title, path):
    ep = np.arange(1, len(h["val_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ep, h["train_loss"], label="train loss"); ax[0].plot(ep, h["val_loss"], label="val loss")
    b = int(np.argmin(h["val_loss"]))
    ax[0].axvline(b + 1, ls="--", c="gray", label=f"min val loss (epoch {b+1})")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy"); ax[0].legend(fontsize=8); ax[0].set_title(title, fontsize=9)
    ax[1].plot(ep, 100 * np.array(h["val_acc"]), marker="o", c="tab:green")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation accuracy (%)"); ax[1].grid(alpha=.3)
    return _save(fig, path)


def fig_kernels(model, path):
    W = model.features[0].weight.detach().cpu()[:, 0].numpy()
    C, cols = len(W), 8
    rows = int(np.ceil(C / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.2 + 0.4))
    for i, a in enumerate(np.atleast_2d(ax).ravel()):
        a.axis("off")
        if i < C:
            m = np.abs(W[i]).max() + 1e-8
            a.imshow(W[i], cmap="RdBu_r", vmin=-m, vmax=m); a.set_title(str(i), fontsize=6)
    fig.suptitle("Learned first-layer 3x3 kernels (red = +, blue = -)", fontsize=9)
    return _save(fig, path)


def fig_featuremaps(model, x, path):
    model.eval()
    n = 3 if model.cfg["use_bn"] else 2
    dev = next(model.parameters()).device
    with torch.no_grad():
        fm = model.features[:n](x.to(dev))[0].cpu().numpy()
    C, cols = len(fm), 8
    rows = int(np.ceil((C + 1) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.2 + 0.4))
    for i, a in enumerate(np.atleast_2d(ax).ravel()):
        a.axis("off")
        if i == 0:
            a.imshow(x[0, 0].numpy(), cmap="gray"); a.set_title("input", fontsize=6)
        elif i <= C:
            a.imshow(fm[i - 1], cmap="viridis"); a.set_title(f"f{i-1}", fontsize=6)
    fig.suptitle("First-layer feature maps (after Conv+BN+ReLU) for one validation image", fontsize=9)
    return _save(fig, path)


def fig_block_diagram(model, path):
    rows, _, _ = hand_table(model.cfg)
    boxes = [("Input", f"1x{model.cfg['in_size']}x{model.cfg['in_size']}", "", "#dddddd")]
    for r in rows:
        n = r[0]
        if n.startswith(("BN", "ReLU", "Flatten")):
            continue
        col = "#9ec9f5" if n.startswith("Conv") else "#f7c59f" if "Pool" in n else "#e6b8e6" if n == "Dropout" else "#b8e0b0"
        extra = "\n+BN+ReLU" if n.startswith("Conv") and model.cfg["use_bn"] else ("\n(ReLU)" if n == "FC1" else "")
        boxes.append((n + extra, r[6], f"{r[-1]:,} p" if r[-1] else "", col))
    fig, ax = plt.subplots(figsize=(1.75 * len(boxes), 2.3))
    ax.set_xlim(-0.2, 1.75 * len(boxes)); ax.set_ylim(-0.3, 1.9); ax.axis("off")
    for i, (t, shp, p, c) in enumerate(boxes):
        x0 = i * 1.75
        ax.add_patch(FancyBboxPatch((x0, 0.2), 1.35, 1.4, boxstyle="round,pad=0.03", fc=c, ec="k"))
        ax.text(x0 + 0.675, 1.1, t, ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(x0 + 0.675, 0.6, shp + ("\n" + p if p else ""), ha="center", va="center", fontsize=7)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x0 + 1.72, 0.9), xytext=(x0 + 1.38, 0.9), arrowprops=dict(arrowstyle="->"))
    return _save(fig, path)


def fig_depth(abl, path):
    d = [1, 2, 3, 4]
    acc = [100 * abl[f"depth_{k}"]["best_val_acc"] for k in d]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(d, acc, marker="o")
    for k, a in zip(d, acc):
        ax.annotate(f"{abl[f'depth_{k}']['n_params']:,} p", (k, a), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
    ax.set_xticks(d); ax.set_xlabel("conv blocks"); ax.set_ylabel("best val accuracy (%)"); ax.grid(alpha=.3)
    return _save(fig, path)


def fig_aug(off, on, path):
    ho, hn = off["hist"], on["hist"]
    ep = np.arange(1, len(ho["val_acc"]) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].plot(ep, 100 * np.array(ho["val_acc"]), label="no aug"); ax[0].plot(ep, 100 * np.array(hn["val_acc"]), label="aug")
    ax[0].set_title("validation accuracy (%)", fontsize=9); ax[0].legend()
    gap = lambda h: 100 * (np.array(h["train_acc_clean"]) - np.array(h["val_acc"]))
    ax[1].plot(ep, gap(ho), label="no aug"); ax[1].plot(ep, gap(hn), label="aug")
    ax[1].set_title("train - val accuracy gap (pp; train measured un-augmented)", fontsize=9); ax[1].legend()
    ax[2].plot(ep, ho["train_loss"], "--", c="C0", label="train, no aug"); ax[2].plot(ep, ho["val_loss"], c="C0", label="val, no aug")
    ax[2].plot(ep, hn["train_loss"], "--", c="C1", label="train, aug"); ax[2].plot(ep, hn["val_loss"], c="C1", label="val, aug")
    ax[2].set_title("loss", fontsize=9); ax[2].legend(fontsize=7)
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=.3)
    return _save(fig, path)


def fig_tl(recs, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for k, r in recs.items():
        ax[0].plot(range(1, len(r["hist"]["val_acc"]) + 1), 100 * np.array(r["hist"]["val_acc"]), marker="o", label=k)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("val accuracy (%)"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    ax[1].bar(range(len(recs)), [r["t_epoch"] for r in recs.values()])
    ax[1].set_xticks(range(len(recs))); ax[1].set_xticklabels(list(recs), rotation=25, ha="right", fontsize=7)
    ax[1].set_ylabel("seconds / epoch")
    return _save(fig, path)


def fig_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=60, ha="right", fontsize=7); ax.set_yticklabels(CLASSES, fontsize=7)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6, color="w" if cm[i, j] > cm.max() / 2 else "k")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=9)
    return _save(fig, path)


def fig_pairs(imgs, labels, preds, a, b, path):
    sets = [(f"{CLASSES[a]} (correct)", (labels == a) & (preds == a)), (f"{CLASSES[b]} (correct)", (labels == b) & (preds == b)),
            (f"true {CLASSES[a]} -> predicted {CLASSES[b]}", (labels == a) & (preds == b)),
            (f"true {CLASSES[b]} -> predicted {CLASSES[a]}", (labels == b) & (preds == a))]
    fig, ax = plt.subplots(4, 5, figsize=(7.5, 6.8))
    for i, (t, m) in enumerate(sets):
        idx = np.where(m)[0][:5]
        for j in range(5):
            ax[i, j].axis("off")
            if j < len(idx):
                ax[i, j].imshow(imgs[idx[j]], cmap="gray")
        ax[i, 0].set_title(t, fontsize=8, loc="left")
    return _save(fig, path)


def fig_gradcam(cases, path):
    names = list(cases[0]["cams"])
    ncol = 1 + len(names)
    fig, ax = plt.subplots(len(cases), ncol, figsize=(2.7 * ncol, 2.9 * len(cases)))
    ax = np.atleast_2d(ax)
    for i, c in enumerate(cases):
        ax[i, 0].imshow(c["img28"], cmap="gray"); ax[i, 0].set_title(c["title"], fontsize=7); ax[i, 0].axis("off")
        for j, n in enumerate(names):
            ax[i, j + 1].imshow(c["img28"], cmap="gray", extent=(0, 1, 1, 0))
            ax[i, j + 1].imshow(c["cams"][n], cmap="jet", alpha=0.5, extent=(0, 1, 1, 0))
            ax[i, j + 1].set_title(f"Grad-CAM {n}", fontsize=7); ax[i, j + 1].axis("off")
    return _save(fig, path)


def fig_denoise(out, path):
    sig = list(out)
    fig, ax = plt.subplots(4 * len(sig), 4, figsize=(6, 6 * len(sig)))
    rows = ["clean", "noisy", "CNN denoised", "3x3 Gaussian"]
    for si, s in enumerate(sig):
        for r in range(4):
            for j in range(4):
                a = ax[4 * si + r, j]
                a.imshow(out[s]["examples"][r][j, 0].numpy(), cmap="gray", vmin=0, vmax=1); a.set_xticks([]); a.set_yticks([])
                if j == 0:
                    a.set_ylabel(f"s={s}\n{rows[r]}", fontsize=7)
    return _save(fig, path)
