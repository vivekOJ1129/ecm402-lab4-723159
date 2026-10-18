"""Training / validation loops and metrics."""
import random, time, platform, os
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def hardware_info(device):
    if device.type == "cuda":
        dev = f"GPU: {torch.cuda.get_device_name(0)}"
    else:
        dev = (f"CPU-only ({platform.processor() or platform.machine()}, "
               f"{os.cpu_count()} logical cores, torch threads={torch.get_num_threads()})")
    return {"device": dev, "torch": torch.__version__, "python": platform.python_version(),
            "os": platform.platform()}


@torch.no_grad()
def evaluate(model, loader, device, return_preds=False):
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    loss, correct, n, P, Y = 0.0, 0, 0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss += crit(out, y).item()
        pred = out.argmax(1)
        correct += (pred == y).sum().item(); n += y.numel()
        if return_preds:
            P.append(pred.cpu()); Y.append(y.cpu())
    res = {"loss": loss / n, "acc": correct / n}
    if return_preds:
        res["preds"] = torch.cat(P).numpy(); res["labels"] = torch.cat(Y).numpy()
    return res


def _freeze_bn(model):
    """BN layers whose parameters are frozen must also stop updating running statistics."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def fit(model, train_loader, val_loader, device, epochs, optimizer, scheduler=None,
        train_eval_loader=None, name="", verbose=True):
    """Trains, keeps the epoch with best VALIDATION accuracy, reloads it before returning."""
    crit = nn.CrossEntropyLoss()
    model.to(device)
    H = {k: [] for k in ("train_loss", "train_acc", "train_acc_clean", "val_loss", "val_acc",
                         "epoch_time", "lr")}
    best_acc, best_ep, best_state = -1.0, 0, None
    for ep in range(1, epochs + 1):
        model.train(); _freeze_bn(model)
        H["lr"].append(optimizer.param_groups[-1]["lr"])
        t0 = time.time(); tot, corr, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x); loss = crit(out, y)
            loss.backward(); optimizer.step()
            tot += loss.item() * y.size(0); corr += (out.argmax(1) == y).sum().item(); n += y.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        H["epoch_time"].append(time.time() - t0)          # training pass only
        if scheduler is not None:
            scheduler.step()
        v = evaluate(model, val_loader, device)
        H["train_loss"].append(tot / n); H["train_acc"].append(corr / n)
        H["val_loss"].append(v["loss"]); H["val_acc"].append(v["acc"])
        if train_eval_loader is not None:
            H["train_acc_clean"].append(evaluate(model, train_eval_loader, device)["acc"])
        if v["acc"] > best_acc:
            best_acc, best_ep = v["acc"], ep
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        if verbose:
            print(f"[{name}] ep {ep:02d}/{epochs} train_loss {tot/n:.4f} val_loss {v['loss']:.4f} "
                  f"val_acc {v['acc']*100:.2f}% ({H['epoch_time'][-1]:.1f}s)", flush=True)
    model.load_state_dict(best_state)
    H["best_val_acc"], H["best_epoch"], H["best_state"] = best_acc, best_ep, best_state
    return H


def make_record(name, kind, cfg, n_params, hist):
    state = hist.pop("best_state")
    return {"name": name, "kind": kind, "cfg": cfg, "n_params": n_params, "hist": hist, "state": state,
            "best_val_acc": hist["best_val_acc"], "best_epoch": hist["best_epoch"],
            "t_epoch": float(np.mean(hist["epoch_time"]))}


def confusion_matrix(y, p, k=10):
    return np.bincount(y * k + p, minlength=k * k).reshape(k, k)       # rows = true, cols = pred


def macro_f1(cm):
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1); rec = tp / np.maximum(cm.sum(1), 1)
    return float(np.mean(2 * prec * rec / np.maximum(prec + rec, 1e-12)))
