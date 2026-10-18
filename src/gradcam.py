"""Task 6 - Grad-CAM from scratch with forward hook + tensor gradient hook."""
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model, self.act, self.grad = model, None, None
        self._h = target_layer.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        self.act = out.detach()
        out.register_hook(lambda g: setattr(self, "grad", g.detach()))

    def __call__(self, x, class_idx=None):
        self.model.eval()
        x = x.clone().requires_grad_(True)              # guarantees a graph even if the layer is frozen
        logits = self.model(x)
        probs = F.softmax(logits, 1)[0].detach().cpu()
        if class_idx is None:
            class_idx = int(logits.argmax(1))
        self.model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()
        w = self.grad.mean(dim=(2, 3), keepdim=True)       # alpha_k = GAP of dy/dA_k
        cam = F.relu((w * self.act).sum(1, keepdim=True))   # ReLU(sum_k alpha_k A_k)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), class_idx, probs

    def close(self):
        self._h.remove()


def fg_stats(cam, img28, thr=0.1):
    """(fraction of CAM mass on garment pixels, fraction of image area covered by the garment)."""
    mask = torch.from_numpy((img28 > thr).astype(np.float32))[None, None]
    mask = F.interpolate(mask, size=cam.shape, mode="nearest")[0, 0].numpy() > 0.5
    return float(cam[mask].sum() / (cam.sum() + 1e-8)), float(mask.mean())
