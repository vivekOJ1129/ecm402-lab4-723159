"""Reproduces every number / figure / table of Lab 4:  python src/main.py   (add --quick for a ~5 min smoke test)."""
import argparse, os, sys, time
from functools import partial
import numpy as np
import torch
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
import models as M
import viz as V
import report as REP
import ablations as A
import transfer as T
import denoise as DN
from gradcam import GradCAM, fg_stats
from train import set_seed, hardware_info, evaluate, confusion_matrix, macro_f1


def kernel_stats(model):
    W = model.features[0].weight.detach().cpu()[:, 0]
    flat = W.reshape(len(W), -1)
    cos = lambda ref: ((flat * ref.reshape(1, -1)).sum(1) / (flat.norm(dim=1) * ref.norm() + 1e-8)).abs()
    sx, sy, gs = cos(D.SOBEL_X), cos(D.SOBEL_Y), cos(D.GAUSS)
    sign = torch.tensor([[1., -1., 1.], [-1., 1., -1.], [1., -1., 1.]])
    dc, nyq = W.sum((1, 2)), (W * sign).sum((1, 2))
    return {"n": len(W), "edge": int(((sx > 0.7) | (sy > 0.7)).sum()), "blob": int((gs > 0.7).sum()),
            "lowpass": int((dc.abs() > nyq.abs()).sum()), "dc_ratio": float((dc.abs() / (flat.norm(dim=1) + 1e-8)).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny smoke test (results go to results_quick/, numbers meaningless)")
    ap.add_argument("--epochs", type=int, default=15, help="epochs for custom CNN / ablations (>=10 required)")
    ap.add_argument("--tl-epochs", type=int, default=6, help="epochs per fine-tuning strategy (5-8)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--skip-bonus", action="store_true")
    a = ap.parse_args()
    q = a.quick
    cfg = dict(quick=q, epochs=1 if q else a.epochs, lr=1e-3, batch_size=128, tl_epochs=1 if q else a.tl_epochs,
               tl_lr=1e-3, tl_n_train=500 if q else 10000, tl_n_val=200 if q else 2000, dn_epochs=3, workers=a.workers)
    device = D.get_device()
    hw = hardware_info(device)
    print(hw, flush=True)
    out = os.path.join(D.ROOT, "results_quick" if q else "results")
    FIG, TAB = os.path.join(out, "figures"), os.path.join(out, "tables")
    CK = os.path.join(D.ROOT, "checkpoints")
    for d in (FIG, TAB, CK):
        os.makedirs(d, exist_ok=True)
    f = lambda n: os.path.join(FIG, n)
    P = {}
    set_seed(D.SEED)

    # ---------------------------------------------------------------- Task 1
    raw = datasets.FashionMNIST(D.DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
    P["t1a"] = V.fig_filtering(raw[0][0], f("t1a_filtering.png"))
    P["t1b"] = V.fig_augmentation(datasets.FashionMNIST(D.DATA_DIR, train=True, download=True), f("t1b_augmentation.png"))

    # ---------------------------------------------------------------- Task 2 (baseline, no augmentation)
    mk = partial(D.make_loaders, batch_size=cfg["batch_size"], quick=q, num_workers=cfg["workers"])
    base = A.train_cnn(device, cfg, mk, "baseline CNN", n_blocks=2, pool="max")
    model = M.build_from_rec(base).to(device)
    rows, total, shapes = M.hand_table(model.cfg)
    trace, rf_last = M.receptive_field(model.cfg)
    hand = {"rows": rows, "total": total, "counted": M.count_params(model), "rf_trace": trace, "rf_last": rf_last,
            "shapes_ok": shapes == M.measured_out_shapes(model)}
    print(f"[Task 2b] hand total {total} vs counted {hand['counted']}; shapes_ok={hand['shapes_ok']}", flush=True)
    REP.save_table(os.path.join(TAB, "t2_layer_table"), ["Layer", "Input", "Kernel", "Stride", "Pad", "Filters", "Output", "Params"], rows)
    P["t2_block"] = V.fig_block_diagram(model, f("t2_block_diagram.png"))
    P["t2_hist"] = V.fig_history(base["hist"], "Custom CNN (no augmentation)", f("t2_training_curves.png"))
    P["t2_kernels"] = V.fig_kernels(model, f("t2_first_layer_kernels.png"))
    x0 = mk(augment=False)["val"].dataset[0][0][None]
    P["t2_fmaps"] = V.fig_featuremaps(model, x0, f("t2_feature_maps.png"))
    kstats = kernel_stats(model)

    # ---------------------------------------------------------------- Task 3
    abl = A.run_ablations(device, cfg, mk, base)
    hdr, arows = A.summary_rows(abl)
    REP.save_table(os.path.join(TAB, "t3_ablation_summary"), hdr, arows)
    P["t3_depth"] = V.fig_depth(abl, f("t3_depth.png"))
    P["t3_aug"] = V.fig_aug(abl["aug_off"], abl["aug_on"], f("t3_augmentation.png"))

    # ---------------------------------------------------------------- Task 4
    L = D.make_tl_loaders(cfg["tl_n_train"], cfg["tl_n_val"], batch_size=32, num_workers=cfg["workers"])
    tl = T.run_transfer(device, cfg, L)
    REP.save_table(os.path.join(TAB, "t4_transfer_summary"), ["Run", "Trainable params", "Best val acc %", "Best epoch", "s/epoch", "Val acc per epoch %"],
                   [[k, r["n_params"], f"{100*r['best_val_acc']:.2f}", r["best_epoch"], f"{r['t_epoch']:.1f}",
                     " ".join(f"{100*v:.1f}" for v in r["hist"]["val_acc"])] for k, r in tl["recs"].items()])
    P["t4_tl"] = V.fig_tl(tl["recs"], f("t4_transfer.png"))

    # ---------------------------------------------------------------- Task 5: all models frozen -> test ONCE
    cands = [abl[k] for k in ("baseline", "pool_avg", "pool_stride", "depth_1", "depth_3", "depth_4", "aug_on")]
    best_cnn = max(cands, key=lambda r: r["best_val_acc"])                 # selected on VALIDATION accuracy
    rn_rec = tl["recs"][tl["selected"]]                                     # selected on VALIDATION accuracy
    torch.save(best_cnn["state"], os.path.join(CK, "custom_cnn.pt"))
    test_c = D.make_test_loader("custom", q)
    test_r = D.make_test_loader("resnet", q, num_workers=cfg["workers"])

    def ev(rec, loader):
        m = M.build_from_rec(rec).to(device)
        r = evaluate(m, loader, device, return_preds=True)
        cm = confusion_matrix(r["labels"], r["preds"])
        return m, r, cm, macro_f1(cm)

    cnn_m, cnn_r, cnn_cm, cnn_f1 = ev(best_cnn, test_c)
    rn_m, rn_r, rn_cm, rn_f1 = ev(rn_rec, test_r)
    _, mlp_r, _, _ = ev(abl["mlp"], test_c)                                  # only to fill the efficiency table
    better = "resnet" if rn_r["acc"] > cnn_r["acc"] else "cnn"
    cm_b, r_b = (rn_cm, rn_r) if better == "resnet" else (cnn_cm, cnn_r)
    pairs = sorted(((int(cm_b[i, j] + cm_b[j, i]), i, j) for i in range(10) for j in range(i + 1, 10)), reverse=True)
    pairs = [(i, j, n) for n, i, j in pairs[:5]]
    final = {"cnn": {"rec": best_cnn, "acc": cnn_r["acc"], "f1": cnn_f1}, "resnet": {"rec": rn_rec, "acc": rn_r["acc"], "f1": rn_f1},
             "mlp": {"rec": abl["mlp"], "acc": mlp_r["acc"]}, "better": better, "pairs": pairs}
    print(f"[Task 5] TEST  CNN {cnn_r['acc']*100:.2f}% (F1 {cnn_f1:.4f}) | ResNet {rn_r['acc']*100:.2f}% (F1 {rn_f1:.4f})", flush=True)
    P["t5_cm"] = V.fig_confusion(cm_b, f"Confusion matrix - {better} (official test set)", f("t5_confusion.png"))
    raw_te = datasets.FashionMNIST(D.DATA_DIR, train=False, download=True).data.numpy() / 255.0
    if q:
        raw_te = raw_te[:1000]
    P["t5_pairs"] = V.fig_pairs(raw_te, r_b["labels"], r_b["preds"], pairs[0][0], pairs[0][1], f("t5_confused_pairs.png"))
    REP.save_table(os.path.join(TAB, "t5_efficiency"), ["Model", "Trainable params", "s/epoch", "Test acc %"],
                   [["MLP baseline", abl["mlp"]["n_params"], f"{abl['mlp']['t_epoch']:.1f}", f"{100*mlp_r['acc']:.2f}"],
                    ["Custom CNN", best_cnn["n_params"], f"{best_cnn['t_epoch']:.1f}", f"{100*cnn_r['acc']:.2f}"],
                    [f"ResNet-18 ({tl['selected']})", rn_rec["n_params"], f"{rn_rec['t_epoch']:.1f}", f"{100*rn_r['acc']:.2f}"]])

    # ---------------------------------------------------------------- Task 6: Grad-CAM on the frozen ResNet-18
    ds_tl = datasets.FashionMNIST(D.DATA_DIR, train=False, download=True, transform=D.resnet_transform())
    lab, prd = rn_r["labels"], rn_r["preds"]

    def cam_case(name, idx):
        x = ds_tl[int(idx)][0][None].to(device)
        img28, cams, st = raw_te[int(idx)], {}, {}
        for ln, layer in (("layer4[-1]", rn_m.layer4[-1]), ("layer3[-1]", rn_m.layer3[-1])):
            gc = GradCAM(rn_m, layer)
            cam, cls, probs = gc(x)
            gc.close()
            cams[ln], st[ln] = cam, fg_stats(cam, img28)
        return dict(name=name, idx=int(idx), true=int(lab[idx]), pred=int(cls), conf=float(probs[cls]),
                    fg4=st["layer4[-1]"][0], fg3=st["layer3[-1]"][0], area=st["layer4[-1]"][1], cams=cams, img28=img28,
                    title=f"{name}: true {D.CLASSES[int(lab[idx])]} / pred {D.CLASSES[int(cls)]}")

    cases, fail_in_pair = [], False
    for name, c in (("Coat", 4), ("Shirt", 6)):
        ok = np.where((lab == c) & (prd == c))[0]
        if len(ok):
            cases.append(cam_case(name, ok[0]))
    pa, pb, _ = pairs[0]
    m1, m2 = (lab == pa) & (prd == pb), (lab == pb) & (prd == pa)
    fidx = np.where(m1 if m1.sum() >= m2.sum() else m2)[0]
    if len(fidx):
        fail_in_pair = True
    else:
        fidx = np.where(lab != prd)[0]
    if len(fidx):
        cases.append(cam_case("failure", fidx[0]))
    P["t6_gc"] = V.fig_gradcam(cases, f("t6_gradcam.png"))
    gcam = {"cases": cases, "fail_in_pair": fail_in_pair}

    # ---------------------------------------------------------------- Task 8 (bonus)
    bonus = None
    if not a.skip_bonus:
        bonus = DN.run(device, cfg)
        P["t8"] = V.fig_denoise(bonus, f("t8_denoising.png"))

    # ---------------------------------------------------------------- report + README
    R = dict(hw=hw, cfg=cfg, paths=P, hand=hand, base=base, kstats=kstats, abl=abl, abl_hdr=hdr, abl_rows=arows,
             tl=tl, final=final, gcam=gcam, bonus=bonus)
    REP.save_table(os.path.join(TAB, "runtimes"), ["Model", "Trainable params", "s/epoch"], REP.runtime_rows(R))
    pdf = os.path.join(out, "report_quick.pdf") if q else os.path.join(D.ROOT, "report.pdf")
    REP.build_report(R, pdf)
    if not q:
        REP.update_readme(os.path.join(D.ROOT, "README.md"), R)
    print(f"Done. Report: {pdf}", flush=True)


if __name__ == "__main__":
    main()
