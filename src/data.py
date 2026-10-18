"""Task 1 - data pipeline: transforms, splits, loaders, classical kernels."""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
FM_MEAN, FM_STD = 0.2860, 0.3530                      # Fashion-MNIST train statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
SOBEL_Y = SOBEL_X.t().contiguous()
GAUSS = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.0


def get_device():
    """Task 1(d): cuda if available else cpu; used everywhere."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def augment_ops():
    # Horizontal flip: garments are (almost) left-right symmetric, so labels are invariant.
    # Small affine (rot <=10 deg, shift <=10%, scale 0.9-1.1): photographed/folded items vary slightly.
    # NOT used: vertical flip / large rotation (upside-down or sideways garments never occur and a
    # 90-degree rotated Trouser looks like a wide Pullover/Bag silhouette -> destroys class evidence).
    return [transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))]


def eval_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((FM_MEAN,), (FM_STD,))])


def train_transform():
    return transforms.Compose(augment_ops() + [transforms.ToTensor(),
                                               transforms.Normalize((FM_MEAN,), (FM_STD,))])


def resnet_transform(size=64):
    return transforms.Compose([transforms.Resize(size),
                               transforms.Grayscale(num_output_channels=3),
                               transforms.ToTensor(),
                               transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def make_splits(seed=SEED, quick=False):
    """80/20 train/val split of the 60k training images with a fixed seed (random_split)."""
    g = torch.Generator().manual_seed(seed)
    tr, va = random_split(range(60000), [48000, 12000], generator=g)
    tr, va = list(tr.indices), list(va.indices)
    if quick:
        tr, va = tr[:4800], va[:1200]
    return tr, va


def make_loaders(augment=True, batch_size=128, quick=False, num_workers=2, seed=SEED):
    """Augmentation is applied to the TRAIN split only; val / clean-train use eval transform."""
    tr_idx, va_idx = make_splits(seed, quick)
    ds_tr = datasets.FashionMNIST(DATA_DIR, train=True, download=True,
                                  transform=train_transform() if augment else eval_transform())
    ds_ev = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=eval_transform())
    g = torch.Generator().manual_seed(seed)
    pin = torch.cuda.is_available()
    nw = num_workers if augment else 0
    n_gap = 2000 if quick else 10000
    return {
        "train": DataLoader(Subset(ds_tr, tr_idx), batch_size=batch_size, shuffle=True,
                            generator=g, num_workers=nw, pin_memory=pin),
        "val": DataLoader(Subset(ds_ev, va_idx), batch_size=512, pin_memory=pin),
        # fixed non-augmented slice of the train split, used for the train-val accuracy gap
        "train_eval": DataLoader(Subset(ds_ev, tr_idx[:n_gap]), batch_size=512, pin_memory=pin),
    }


def stratified_split(targets, n_train, n_val, seed=SEED):
    rng = np.random.RandomState(seed)
    t = np.asarray(targets)
    ptr, pva = n_train // 10, n_val // 10
    tr, va = [], []
    for c in range(10):
        idx = np.where(t == c)[0]
        rng.shuffle(idx)
        tr += idx[:ptr].tolist()
        va += idx[ptr:ptr + pva].tolist()
    return tr, va


def make_tl_loaders(n_train=10000, n_val=2000, size=64, batch_size=32, seed=SEED, num_workers=2):
    """Task 4: stratified fixed subset (seed reported), disjoint train/val, 64x64 3-channel."""
    ds = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=resnet_transform(size))
    tr, va = stratified_split(ds.targets.numpy(), n_train, n_val, seed)
    g = torch.Generator().manual_seed(seed)
    pin = torch.cuda.is_available()
    return {"train": DataLoader(Subset(ds, tr), batch_size=batch_size, shuffle=True, generator=g,
                                num_workers=num_workers, pin_memory=pin),
            "val": DataLoader(Subset(ds, va), batch_size=128, num_workers=num_workers, pin_memory=pin)}


def make_test_loader(kind="custom", quick=False, size=64, num_workers=0):
    """Official 10k test set (touched only in Task 5/6)."""
    tf = eval_transform() if kind == "custom" else resnet_transform(size)
    ds = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=tf)
    if quick:
        ds = Subset(ds, list(range(1000)))
    return DataLoader(ds, batch_size=256, num_workers=num_workers)
