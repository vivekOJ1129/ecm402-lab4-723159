"""Task 4 - ResNet-18 transfer learning (64x64 inputs, 10k/2k stratified subset)."""
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import SEED
from models import STRATEGIES, make_resnet18, param_groups, count_params, verify_resnet_input
from train import fit, make_record, set_seed

LABELS = {"feature_extraction": "Feature extraction (backbone frozen)",
          "gradual_unfreeze": "Gradual unfreezing (layer4 + fc)",
          "diff_lr": "Differential LR (all layers, backbone 0.1x)"}


def run_transfer(device, cfg, L):
    info = {}
    info["feat"], info["out"] = verify_resnet_input(make_resnet18("feature_extraction", pretrained=False), 64)
    recs = {}

    def go(strat, key, sched):
        set_seed(SEED)
        model = make_resnet18(strat)
        n = count_params(model)
        opt = torch.optim.Adam(param_groups(model, strat, cfg["tl_lr"]))
        s = CosineAnnealingLR(opt, T_max=cfg["tl_epochs"]) if sched else None
        hist = fit(model, L["train"], L["val"], device, cfg["tl_epochs"], opt, scheduler=s, name=key)
        recs[key] = make_record(key, "resnet", {"strategy": strat}, n, hist)

    for s in STRATEGIES:
        go(s, s, False)
    best = max(STRATEGIES, key=lambda k: recs[k]["best_val_acc"])
    go(best, best + "+cosine", True)                          # Task 4(d): scheduler on best strategy
    selected = max(recs, key=lambda k: recs[k]["best_val_acc"])
    for k in recs:                                            # free memory: keep only the selected weights
        if k != selected:
            recs[k]["state"] = None
    return {"recs": recs, "info": info, "best_strategy": best, "selected": selected}
