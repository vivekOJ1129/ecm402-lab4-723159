"""Bonus - CNN encoder-decoder denoiser vs 3x3 Gaussian blur."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import data as D
from train import set_seed


class DenoiseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(True),
                                 nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(True))
        self.dec = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True), nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        return self.dec(self.enc(x))


def psnr(a, b):
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return float((10 * torch.log10(1.0 / mse)).mean())


def _conv(x, k):
    return F.conv2d(x, k[None, None].to(x), padding=1)


def edge_mag(x):
    return float(((_conv(x, D.SOBEL_X) ** 2 + _conv(x, D.SOBEL_Y) ** 2).sqrt()).mean())


def run(device, cfg, seed=D.SEED):
    tr_idx, _ = D.make_splits(seed, cfg["quick"])
    tr = datasets.FashionMNIST(D.DATA_DIR, train=True, download=True)
    te = datasets.FashionMNIST(D.DATA_DIR, train=False, download=True)
    Xtr = tr.data[tr_idx].float().div(255).unsqueeze(1)
    Xte = te.data.float().div(255).unsqueeze(1)
    if cfg["quick"]:
        Xte = Xte[:1000]
    epochs = 1 if cfg["quick"] else cfg["dn_epochs"]
    out = {}
    for sigma in (0.1, 0.3):
        set_seed(seed)
        net = DenoiseNet().to(device)
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        g = torch.Generator().manual_seed(seed)
        for _ in range(epochs):
            net.train()
            perm = torch.randperm(len(Xtr), generator=g)
            for i in range(0, len(Xtr), 128):
                clean = Xtr[perm[i:i + 128]].to(device)
                noisy = (clean + sigma * torch.randn_like(clean)).clamp(0, 1)
                loss = F.mse_loss(net(noisy), clean)
                opt.zero_grad(); loss.backward(); opt.step()
        g2 = torch.Generator().manual_seed(123)
        Xn = (Xte + sigma * torch.randn(Xte.shape, generator=g2)).clamp(0, 1)
        net.eval()
        with torch.no_grad():
            den = torch.cat([net(Xn[i:i + 500].to(device)).cpu() for i in range(0, len(Xn), 500)])
        bl = _conv(Xn, D.GAUSS)
        out[sigma] = {"psnr_noisy": psnr(Xn, Xte), "psnr_cnn": psnr(den, Xte), "psnr_blur": psnr(bl, Xte),
                      "edge_clean": edge_mag(Xte), "edge_noisy": edge_mag(Xn),
                      "edge_cnn": edge_mag(den), "edge_blur": edge_mag(bl),
                      "examples": (Xte[:4], Xn[:4], den[:4], bl[:4])}
        print(f"[denoise sigma={sigma}] PSNR noisy {out[sigma]['psnr_noisy']:.2f} "
              f"cnn {out[sigma]['psnr_cnn']:.2f} blur {out[sigma]['psnr_blur']:.2f}", flush=True)
    return out
