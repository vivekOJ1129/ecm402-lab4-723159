"""Task 3 - controlled ablations: one factor at a time, seed/optimizer/lr/epochs fixed."""
import torch
from data import SEED
from models import CustomCNN, MLP, mlp_hidden_for, count_params, receptive_field
from train import fit, make_record, set_seed


def train_cnn(device, cfg, mk, name, aug=False, **model_kw):
    set_seed(SEED)
    loaders = mk(augment=aug)
    model = CustomCNN(**model_kw).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    hist = fit(model, loaders["train"], loaders["val"], device, cfg["epochs"], opt,
               train_eval_loader=loaders["train_eval"], name=name)
    return make_record(name, "cnn", model.cfg, count_params(model), hist)


def train_mlp(device, cfg, mk, name, target_params):
    set_seed(SEED)
    loaders = mk(augment=False)
    h = mlp_hidden_for(target_params)
    model = MLP(hidden=h).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    hist = fit(model, loaders["train"], loaders["val"], device, cfg["epochs"], opt,
               train_eval_loader=loaders["train_eval"], name=name)
    return make_record(name, "mlp", {"hidden": h}, count_params(model), hist)


def run_ablations(device, cfg, mk, base):
    R = {"baseline": base, "pool_max": base, "depth_2": base}
    R["mlp"] = train_mlp(device, cfg, mk, "MLP", base["n_params"])                       # (a)
    R["pool_avg"] = train_cnn(device, cfg, mk, "AvgPool", n_blocks=2, pool="avg")         # (b)
    R["pool_stride"] = train_cnn(device, cfg, mk, "Stride-2 conv", n_blocks=2, pool="stride")
    for d in (1, 3, 4):                                                                   # (c)
        R[f"depth_{d}"] = train_cnn(device, cfg, mk, f"depth {d}", n_blocks=d, pool="max")
    cands = [R[k] for k in ("baseline", "pool_avg", "pool_stride", "depth_1", "depth_3", "depth_4")]
    best = max(cands, key=lambda r: r["best_val_acc"])                                    # (d)
    R["aug_off"] = best                                                                    # identical run, no aug
    kw = {k: best["cfg"][k] for k in ("n_blocks", "pool")}
    R["aug_on"] = train_cnn(device, cfg, mk, "best arch + augmentation", aug=True, **kw)
    R["aug_arch"] = kw
    return R


def summary_rows(R):
    spec = [("(a) MLP vs CNN", "MLP baseline", "mlp"), ("(a) MLP vs CNN", "CNN 2 blocks, MaxPool", "baseline"),
            ("(b) Pooling", "MaxPool", "pool_max"), ("(b) Pooling", "AvgPool", "pool_avg"),
            ("(b) Pooling", "No pool (stride-2 conv)", "pool_stride"),
            ("(c) Depth", "1 conv block", "depth_1"), ("(c) Depth", "2 conv blocks", "depth_2"),
            ("(c) Depth", "3 conv blocks", "depth_3"), ("(c) Depth", "4 conv blocks", "depth_4"),
            ("(d) Augmentation", "best arch, no aug", "aug_off"), ("(d) Augmentation", "best arch, with aug", "aug_on")]
    rows = []
    for exp, var, k in spec:
        r = R[k]
        rf = receptive_field(r["cfg"])[1] if r["kind"] == "cnn" else "-"
        rows.append([exp, var, r["n_params"], f"{100*r['best_val_acc']:.2f}", r["best_epoch"],
                     f"{r['t_epoch']:.1f}", rf])
    return ["Experiment", "Variant", "Params", "Val acc %", "Best ep", "s/epoch", "RF"], rows
