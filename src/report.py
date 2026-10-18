"""Builds report.pdf, results/tables/*, and the README hardware block from the REAL results dict."""
import csv, os, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from data import SEED, CLASSES
from models import receptive_field
from transfer import LABELS

W, H = 8.27, 11.69
pp = lambda x: f"{100 * x:.2f}%"
dp = lambda x: f"{100 * x:+.2f} pp"


def fmt_table(headers, rows):
    rows = [[str(c) for c in r] for r in rows]
    w = [max(len(str(h)), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = lambda r: " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r))
    return "\n".join([line(headers), "-+-".join("-" * x for x in w)] + [line(r) for r in rows])


def save_table(base, headers, rows):
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(headers); wr.writerows(rows)
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n|" + "---|" * len(headers) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(c) for c in r) + " |\n")


class Report:
    def __init__(self, path):
        self.pdf, self.fig, self.y = PdfPages(path), None, 0
        self.new_page()

    def new_page(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig); plt.close(self.fig)
        self.fig, self.y = plt.figure(figsize=(W, H)), 0.955

    def _need(self, h):
        if self.y - h < 0.04:
            self.new_page()

    def heading(self, t, size=12):
        self._need(0.05); self.y -= 0.008
        self.fig.text(0.07, self.y, t, fontsize=size, fontweight="bold", va="top"); self.y -= 0.032

    def para(self, t, size=8.6, width=108):
        lh = size * 1.5 / 72 / H
        for line in textwrap.wrap(t, width):
            self._need(lh); self.fig.text(0.07, self.y, line, fontsize=size, va="top"); self.y -= lh
        self.y -= lh * 0.6

    def mono(self, t, size=6.6):
        lh = size * 1.4 / 72 / H
        for line in t.split("\n"):
            self._need(lh); self.fig.text(0.07, self.y, line, fontsize=size, family="monospace", va="top"); self.y -= lh
        self.y -= lh * 0.6

    def image(self, path, caption="", wfrac=0.86):
        im = plt.imread(path)
        asp = im.shape[0] / im.shape[1]
        hf = wfrac * W * asp / H
        if hf > 0.86:
            wfrac, hf = wfrac * 0.86 / hf, 0.86
        self._need(hf + 0.03)
        ax = self.fig.add_axes([(1 - wfrac) / 2, self.y - hf, wfrac, hf]); ax.imshow(im); ax.axis("off")
        self.y -= hf + 0.004
        if caption:
            self.fig.text(0.5, self.y, caption, fontsize=7, style="italic", ha="center", va="top"); self.y -= 0.022

    def close(self):
        self.pdf.savefig(self.fig); plt.close(self.fig); self.pdf.close()


def runtime_rows(R):
    rows = [["Custom CNN baseline", R["base"]["n_params"], f"{R['base']['t_epoch']:.1f}"]]
    for k in ("mlp", "pool_avg", "pool_stride", "depth_1", "depth_3", "depth_4", "aug_on"):
        rows.append([R["abl"][k]["name"], R["abl"][k]["n_params"], f"{R['abl'][k]['t_epoch']:.1f}"])
    for k, r in R["tl"]["recs"].items():
        rows.append([f"ResNet-18 {k} (64x64, 10k subset)", r["n_params"], f"{r['t_epoch']:.1f}"])
    return rows


def update_readme(path, R):
    hw, cfg = R["hw"], R["cfg"]
    block = [f"- Mode: {'QUICK SMOKE TEST (numbers meaningless - rerun without --quick)' if cfg['quick'] else 'full run'}",
             f"- Device: {hw['device']}", f"- PyTorch {hw['torch']}, Python {hw['python']}, {hw['os']}",
             f"- Seed: {SEED} (train/val split, stratified TL subset, weight init, noise); custom epochs: {cfg['epochs']}, "
             f"TL epochs: {cfg['tl_epochs']}, TL subset: {cfg['tl_n_train']} train / {cfg['tl_n_val']} val",
             "", "Training time per epoch (mean over epochs, training pass only):", "",
             "| Model | Trainable params | s/epoch |", "|---|---|---|"]
    block += [f"| {a} | {b:,} | {c} |" for a, b, c in runtime_rows(R)]
    s, e = "<!-- AUTO:START -->", "<!-- AUTO:END -->"
    txt = open(path, encoding="utf-8").read()
    i, j = txt.index(s), txt.index(e) + len(e)
    open(path, "w", encoding="utf-8").write(txt[:i] + s + "\n" + "\n".join(block) + "\n" + e + txt[j:])


def build_report(R, out_pdf):
    P, cfg, hw, A, TL, FN, G = R["paths"], R["cfg"], R["hw"], R["abl"], R["tl"], R["final"], R["gcam"]
    base, bh = R["base"], R["base"]["hist"]
    rep = Report(out_pdf)
    rep.heading("ECM-402 Lab 4: CNNs for Multi-Dimensional Signals - Report", 15)
    rep.para(f"Hardware: {hw['device']}; PyTorch {hw['torch']}. Seed {SEED} everywhere. Mode: "
             f"{'QUICK SMOKE TEST - numbers below are NOT meaningful' if cfg['quick'] else 'full run'}. "
             "All numbers in this report are produced by `python src/main.py`. Protocol: every model-selection decision "
             "(architecture, augmentation, fine-tuning strategy, best epoch) uses the validation split only; the official test set "
             "is evaluated once, in Task 5, after all models were frozen (Task 6 only analyses the already-frozen model).")

    # ---------------- Task 1
    rep.heading("1. Data pipeline and the filtering view")
    rep.image(P["t1a"], "Fig 1. Original image and three classical 3x3 kernels applied with Eq. (1).")
    rep.para("(a) Eq. (1) is a sliding weighted sum of the neighbourhood of each pixel; F.conv2d computes cross-correlation, so the kernels "
             "were flipped to implement true convolution. The Gaussian kernel has all-positive taps summing to 1 (DC gain 1, ~0 at Nyquist): "
             "a LOW-pass filter that blurs. Sobel-x and Sobel-y have taps summing to 0 (DC gain 0): they suppress flat regions and respond "
             "to intensity changes, i.e. HIGH-pass (derivative) along the differentiating axis, with [1 2 1] smoothing along the other axis. "
             "Sobel-x highlights vertical edges (garment sides), Sobel-y horizontal edges (hem, shoulders).")
    rep.image(P["t1b"], "Fig 2. Augmentation pipeline examples.")
    rep.para("(b) Pipeline: RandomHorizontalFlip(0.5) + RandomAffine(rotation<=10 deg, shift<=10%, scale 0.9-1.1) + ToTensor + Normalize"
             "(0.2860, 0.3530). Horizontal flip is valid because garments are close to left-right symmetric, so the label is invariant. "
             "Small rotation/shift/scale mimic small pose and framing variations. A BAD idea: vertical flip or large rotation - upside-down or "
             "sideways garments never occur in the data, and a ~90 degree rotated Trouser (tall, narrow) becomes a wide shape resembling a "
             "Pullover or Bag, so the transform would erase exactly the silhouette cue that separates classes.")
    rep.para("(c) 60,000 training images are split 48,000 / 12,000 (80/20) with torch.utils.data.random_split and a fixed generator seed. "
             "The validation set is needed because all model selection must be done on data that is not the test set; otherwise the test "
             "accuracy becomes an optimistic, selected-on number and stops measuring generalisation. Augmentation is applied to the train "
             "split only: the validation set must represent the (un-augmented) test distribution and give a deterministic, comparable "
             "metric across runs; augmenting it would add random noise to the metric and measure performance on images that never occur at test time.")
    rep.para("(d) get_device() returns cuda if available else cpu; models and tensors are moved with it, so the code runs unchanged on both.")

    # ---------------- Task 2
    rep.heading("2. Custom CNN")
    c = base["cfg"]
    rep.para(f"(a) Architecture: {c['n_blocks']} x [Conv3x3 (stride 1, pad 1) + BatchNorm + ReLU + MaxPool2x2] -> Flatten -> FC({c['fc_hidden']}) "
             f"-> ReLU -> Dropout({c['dropout']}) -> FC(10). Channels 32 then 64. Adam, lr {cfg['lr']}, batch {cfg['batch_size']}, "
             f"{cfg['epochs']} epochs, CrossEntropyLoss, no augmentation (this run is the reference for all ablations).")
    hd = ["Layer", "Input", "Kernel", "Stride", "Pad", "Filters", "Output", "Params"]
    rep.mono(fmt_table(hd, R["hand"]["rows"]))
    hnd = R["hand"]
    rep.para(f"(b) Hand-computed total = {hnd['total']:,}; programmatic count sum(p.numel() for p in model.parameters() if p.requires_grad) = "
             f"{hnd['counted']:,}; difference = {hnd['total'] - hnd['counted']}. Per-layer output shapes measured with forward hooks "
             f"{'match' if hnd['shapes_ok'] else 'DO NOT MATCH'} the hand-computed shapes. Pitfalls checked: BatchNorm contributes 2C trainable "
             "parameters (gamma, beta; running mean/var are buffers, not parameters), each Conv/FC bias adds one parameter per output channel, "
             "and MaxPool floors odd sizes (7 -> 3), which determines the Flatten size.")
    rep.image(P["t2_block"], "Fig 3. Block diagram of the custom CNN (shapes and parameter counts).")
    tr = "; ".join(f"{n}: RF={r}, jump={j}" for n, r, j in hnd["rf_trace"])
    rep.para(f"(c) Receptive field (r_out = r_in + (k-1)*j_in, j_out = j_in*stride): {tr}. A unit in the last conv layer sees "
             f"{hnd['rf_last']}x{hnd['rf_last']} input pixels ({100 * hnd['rf_last'] ** 2 / 784:.0f}% of the 28x28 image). That is NOT large enough to "
             "see a whole garment, so no single conv unit can recognise the global silhouette; the Flatten + fully-connected head must integrate "
             "the 7x7 map of local features across the entire image. A deeper stack (or global pooling) would be needed for conv units alone to do so.")
    vl, trl = np.array(bh["val_loss"]), np.array(bh["train_loss"])
    b = int(vl.argmin())
    if b == len(vl) - 1:
        ov = "Validation loss was still improving at the last epoch, so there is no sign of overfitting yet (more epochs may help)."
    elif vl[-1] > vl[b] * 1.03 and trl[-1] < trl[b]:
        ov = (f"After epoch {b+1}, train loss keeps falling ({trl[b]:.4f} -> {trl[-1]:.4f}) while validation loss rises "
              f"({vl[b]:.4f} -> {vl[-1]:.4f}): the model overfits after that point. The checkpoint with best validation ACCURACY (epoch {base['best_epoch']}) is kept.")
    else:
        ov = (f"After epoch {b+1} validation loss ({vl[b]:.4f} -> {vl[-1]:.4f}) and train loss ({trl[b]:.4f} -> {trl[-1]:.4f}) do not diverge "
              "markedly, so overfitting after that point is mild in this run.")
    rep.image(P["t2_hist"], "Fig 4. Training/validation loss and validation accuracy.")
    rep.para(f"(d) Best validation accuracy {pp(base['best_val_acc'])} (epoch {base['best_epoch']}). Validation loss is lowest at epoch {b+1}. {ov}")
    rep.image(P["t2_kernels"], "Fig 5. All 32 learned first-layer kernels.")
    rep.image(P["t2_fmaps"], "Fig 6. First-layer feature maps for one validation image.")
    k = R["kstats"]
    rep.para(f"(e) Of {k['n']} first-layer kernels, {k['edge']} have |cosine similarity| > 0.7 with Sobel-x or Sobel-y (oriented edge/derivative detectors), "
             f"{k['blob']} with the Gaussian (blob / smoothing detectors), and {k['lowpass']} have |DC gain| > |Nyquist gain| (low-pass-like) vs "
             f"{k['n'] - k['lowpass']} that are high-pass-like. Mean |DC gain| / ||w|| = {k['dc_ratio']:.2f} (a Sobel kernel has 0, a 3x3 Gaussian ~2.7, the maximum possible is 3). "
             "Compare Fig 5 with Fig 1: the learned bank contains both smoothing-like and derivative-like kernels, but also mixed ones that no one would hand-design. "
             "These filters were found by gradient descent, not specified by an engineer.")

    # ---------------- Task 3
    rep.heading("3. Ablation study (one factor changed at a time; seed, optimizer, lr, epochs fixed)")
    rep.mono(fmt_table(R["abl_hdr"], R["abl_rows"]))
    rep.para("Val acc = best validation accuracy over epochs; RF = receptive field of last conv unit; s/epoch = training pass only. "
             "Rows (b) MaxPool, (c) 2 blocks and (a) CNN are the same run (the baseline).")
    mlp, cnn = A["mlp"], A["baseline"]
    rep.para(f"(a) MLP ({mlp['n_params']:,} params, hidden width {mlp['cfg']['hidden']}) reaches {pp(mlp['best_val_acc'])}; the CNN ({cnn['n_params']:,} params) "
             f"reaches {pp(cnn['best_val_acc'])} ({dp(cnn['best_val_acc'] - mlp['best_val_acc'])}). "
             + ("At a comparable parameter budget the CNN is better: locality and weight sharing give a useful inductive bias (the same 3x3 detector is reused at every position) "
                "instead of spending parameters on independent weights per pixel." if cnn['best_val_acc'] > mlp['best_val_acc'] else
                "In this run the MLP is not worse than the CNN, so on this small, centred, low-resolution dataset the inductive bias of convolution gave no measurable benefit at this budget."))
    mx, av, st = A["pool_max"], A["pool_avg"], A["pool_stride"]
    rfs = {k: receptive_field(A[k]["cfg"])[1] for k in ("pool_max", "pool_avg", "pool_stride")}
    best_pool = max(("pool_max", "pool_avg", "pool_stride"), key=lambda q: A[q]["best_val_acc"])
    rep.para(f"(b) MaxPool {pp(mx['best_val_acc'])} ({mx['n_params']:,} p, {mx['t_epoch']:.1f}s/ep, RF {rfs['pool_max']}), AvgPool {pp(av['best_val_acc'])} "
             f"({av['n_params']:,} p, {av['t_epoch']:.1f}s/ep, RF {rfs['pool_avg']}), stride-2 conv without pooling {pp(st['best_val_acc'])} "
             f"({st['n_params']:,} p, {st['t_epoch']:.1f}s/ep, RF {rfs['pool_stride']}). Best on validation: {best_pool}. Evidence in the numbers: "
             "(i) DOWNSAMPLING - pooling/stride shrink the feature map so the FC layer is much smaller and the cost per epoch drops (see params and s/epoch); "
             "(ii) RECEPTIVE-FIELD GROWTH - the analytic RF column shows pooling and striding both enlarge what each deeper unit sees; "
             "(iii) LOCAL TRANSLATION INVARIANCE - cannot be isolated from accuracy alone (it would need a shifted-input test), so these numbers give "
             "no direct evidence for it.")
    dacc = [A[f"depth_{d}"]["best_val_acc"] for d in (1, 2, 3, 4)]
    gains = np.diff(dacc) * 100
    mono = bool(all(g >= 0 for g in gains))
    dim = next((i + 2 for i, g in enumerate(gains) if g < 0.3), None)
    rep.image(P["t3_depth"], "Fig 7. Validation accuracy vs number of conv blocks (labels: parameter counts).")
    rep.para(f"(c) Validation accuracy for 1/2/3/4 blocks: {', '.join(pp(a) for a in dacc)}. Accuracy is "
             f"{'monotonically non-decreasing' if mono else 'NOT monotonic'} in this run; gains between successive depths: "
             f"{', '.join(f'{g:+.2f} pp' for g in gains)}. "
             + (f"Returns diminish from {dim} blocks on (gain < 0.3 pp). " if dim else "No depth step gained < 0.3 pp. ")
             + "Why stacking more eventually stops helping on 28x28: (1) spatial resolution runs out - with 2x2 pooling 28 -> 14 -> 7 -> 3 -> 1, "
             "so the 4th block works on a 3x3 map and its 3x3 kernels see mostly zero padding; (2) optimisation gets harder without skip connections "
             "(vanishing/exploding gradients, more hyper-parameter sensitivity) and the data is simple enough that extra capacity mostly overfits. "
             "The FC head also changes size with depth, so parameter count is a confound (see table).")
    rep.image(P["t3_aug"], "Fig 8. Augmentation: val accuracy, generalisation gap, and loss.")
    off, on = A["aug_off"], A["aug_on"]
    g_off = float(np.mean(100 * (np.array(off["hist"]["train_acc_clean"][-3:]) - np.array(off["hist"]["val_acc"][-3:]))))
    g_on = float(np.mean(100 * (np.array(on["hist"]["train_acc_clean"][-3:]) - np.array(on["hist"]["val_acc"][-3:]))))
    tl_off, tl_on = off["hist"]["train_loss"][-1], on["hist"]["train_loss"][-1]
    rep.para(f"(d) Best architecture (chosen on validation): {A['aug_arch']['n_blocks']} blocks, {A['aug_arch']['pool']} downsampling. "
             f"Best val acc {pp(off['best_val_acc'])} without vs {pp(on['best_val_acc'])} with augmentation ({dp(on['best_val_acc'] - off['best_val_acc'])}). "
             f"Mean train-val accuracy gap over the last 3 epochs: {g_off:.2f} pp without vs {g_on:.2f} pp with augmentation. "
             f"Final training loss {tl_off:.4f} -> {tl_on:.4f}. Observation: training loss {'widened (higher)' if tl_on > tl_off else 'did not widen'} and the gap "
             f"{'narrowed' if g_on < g_off else 'did NOT narrow'} - i.e. the expected effect {'is' if (tl_on > tl_off and g_on < g_off) else 'is not fully'} observed in this run.")

    # ---------------- Task 4
    rep.heading("4. Transfer learning with ResNet-18")
    rep.para(f"Compute budget respected: inputs 64x64, stratified fixed subset of {cfg['tl_n_train']} train / {cfg['tl_n_val']} val images (seed {SEED}, disjoint), "
             f"{cfg['tl_epochs']} epochs per strategy, batch 32, Adam lr {cfg['tl_lr']} (classifier; backbone 0.1x for differential LR). Hardware: {hw['device']}.")
    rep.para("(a) Pipeline: Resize(64) -> Grayscale(3 channels) -> ToTensor -> Normalize(ImageNet mean/std). ImageNet statistics are required because the pretrained "
             "weights, especially the first conv and all BatchNorm running statistics, were fitted to inputs with that distribution; feeding differently "
             "scaled inputs would shift every activation away from what the frozen/pretrained features expect.")
    fi = TL["info"]
    rep.para(f"(b) models.resnet18(weights='IMAGENET1K_V1') with fc replaced by Linear(512, 10). Verified at 64x64: layer4 feature map shape {fi['feat']}, "
             f"logits shape {fi['out']}. It runs at 64x64 because of the AdaptiveAvgPool2d layer before fc, which pools any HxW map to 1x1 so the "
             "fc input is always 512 regardless of input size.")
    hdr = ["Run", "Trainable params", "Val acc per epoch (%)", "Best", "s/epoch"]
    rows = [[k, f"{r['n_params']:,}", " ".join(f"{100*a:.1f}" for a in r["hist"]["val_acc"]), f"{100*r['best_val_acc']:.2f}", f"{r['t_epoch']:.1f}"]
            for k, r in TL["recs"].items()]
    rep.mono(fmt_table(hdr, rows))
    rep.image(P["t4_tl"], "Fig 9. Fine-tuning strategies: validation accuracy per epoch and time per epoch.")
    S = ("feature_extraction", "gradual_unfreeze", "diff_lr")
    ref = max(TL["recs"][s]["best_val_acc"] for s in S)

    def ep_to(r):
        return next((i + 1 for i, a in enumerate(r["hist"]["val_acc"]) if a >= ref - 0.01), None)
    fast = {s: ep_to(TL["recs"][s]) for s in S}
    fastest = min(S, key=lambda s: fast[s] if fast[s] else 10 ** 9)
    top = max(S, key=lambda s: TL["recs"][s]["best_val_acc"])
    rep.para(f"(c) Epochs needed to get within 1 pp of the best strategy's accuracy: {fast}. Fastest: {fastest}. Highest validation accuracy: {top} "
             f"({pp(TL['recs'][top]['best_val_acc'])}). Trade-off: unfreezing more layers adds capacity and lets features adapt from ImageNet RGB photos to "
             "grayscale clothing, but with only 10k images it also adds variance - more parameters to fit, higher risk of overfitting and of destroying useful "
             "pretrained features with large early updates, and each epoch costs more time. A frozen backbone has few trainable parameters and cannot overfit as easily, "
             "so unfreezing more layers does not automatically win.")
    bs = TL["best_strategy"]; sc = TL["recs"][bs + "+cosine"]; cn = TL["recs"][bs]
    rep.para(f"(d) CosineAnnealingLR on the best strategy ({bs}): val acc {pp(cn['best_val_acc'])} (constant LR) vs {pp(sc['best_val_acc'])} (cosine), "
             f"{dp(sc['best_val_acc'] - cn['best_val_acc'])}. Decaying the LR matters in fine-tuning because the starting weights are already good: a large LR late in "
             "training keeps kicking the weights around the optimum, while annealing to ~0 lets them settle in a minimum and refine rather than overwrite the pretrained features.")
    rep.para(f"Selected transfer model (highest validation accuracy among all runs): {TL['selected']}.")

    # ---------------- Task 5
    rep.heading("5. Final evaluation (official test set, evaluated once)")
    ch, rn, ml = FN["cnn"], FN["resnet"], FN["mlp"]
    rep.para(f"(a) Best custom CNN (selected on validation accuracy: {ch['rec']['name']}) and best transfer model (selected on validation accuracy: {TL['selected']}) "
             "were frozen and evaluated once on the 10,000 official test images.")
    rep.mono(fmt_table(["Model", "Test acc", "Macro-F1"], [["Custom CNN", pp(ch["acc"]), f"{ch['f1']:.4f}"], ["ResNet-18", pp(rn["acc"]), f"{rn['f1']:.4f}"]]))
    rep.image(P["t5_cm"], f"Fig 10. Confusion matrix of the better model ({FN['better']}); this choice is for display only, not for model selection.")
    (a, b_, n1), (a2, b2, n2) = FN["pairs"][0], FN["pairs"][1]
    upper = {0, 2, 4, 6}
    expl = ("These are all upper-body garments with nearly the same silhouette (torso + two sleeves); what distinguishes them - collar, buttons, zipper, sleeve length, "
            "hem - is a few pixels wide or lost entirely at 28x28, so the evidence is weak and the classes overlap in appearance."
            if {a, b_} <= upper else
            "Inspect Fig 11: at 28x28 the shapes of these two classes overlap and the distinguishing detail spans only a few pixels.")
    rep.image(P["t5_pairs"], "Fig 11. Examples of the most confused pair (correct and misclassified).")
    rep.para(f"(b) Most confused pair: {CLASSES[a]} <-> {CLASSES[b_]} ({n1} errors in both directions); second: {CLASSES[a2]} <-> {CLASSES[b2]} ({n2}). {expl}")
    hdr = ["Model", "Trainable params", "s/epoch", "Test acc", "Test acc per M params"]
    erow = [[nm, f"{r['n_params']:,}", f"{r['t_epoch']:.1f}", pp(a_), f"{100 * a_ / (r['n_params'] / 1e6):.2f}"]
            for nm, r, a_ in (("MLP baseline", ml["rec"], ml["acc"]), ("Custom CNN", ch["rec"], ch["acc"]), (f"ResNet-18 ({TL['selected']})", rn["rec"], rn["acc"]))]
    rep.mono(fmt_table(hdr, erow))
    rep.para(f"(c) Note: the MLP was evaluated on the test set once only to fill this table, not for selection. ResNet s/epoch is for the {cfg['tl_n_train']}-image subset "
             f"at 64x64 while CNN/MLP use all 48,000 images at 28x28, so times are not directly comparable. Accuracy per parameter strongly favours the small CNN "
             f"({ch['rec']['n_params']:,} params). ResNet-18 vs custom CNN test accuracy: {pp(rn['acc'])} vs {pp(ch['acc'])} ({dp(rn['acc'] - ch['acc'])}). "
             + ("The ~11M-parameter pretrained model " + ("does not beat" if rn['acc'] <= ch['acc'] else "outperforms") +
                " the purpose-built CNN here, so on a task this small the extra cost is hard to justify unless that margin matters.") )

    # ---------------- Task 6
    rep.heading("6. Explainability with Grad-CAM")
    rep.para("(a) Grad-CAM implemented from scratch (forward hook on the target layer + tensor hook for dy_c/dA; alpha_k = spatial mean of the gradient; "
             "CAM = ReLU(sum_k alpha_k A_k); bilinear upsampling; min-max normalisation). Target layer for ResNet-18: model.layer4[-1] (block output, after its "
             "final BN + residual add + ReLU), as recommended; layer3[-1] is shown additionally for higher resolution. At 64x64 input the layer4 map is only 2x2, "
             "so each CAM cell covers half the image: heatmaps from layer4 are very coarse.")
    rep.image(P["t6_gc"], "Fig 12. Grad-CAM for a correct Coat, a correct Shirt and a misclassified image (class = predicted class).")
    hdr = ["Case", "True", "Pred", "Conf", "CAM mass on garment (L4)", "same (L3)", "Garment area"]
    rows = [[c["name"], CLASSES[c["true"]], CLASSES[c["pred"]], f"{100 * c['conf']:.1f}%", f"{100 * c['fg4']:.0f}%", f"{100 * c['fg3']:.0f}%", f"{100 * c['area']:.0f}%"] for c in G["cases"]]
    rep.mono(fmt_table(hdr, rows))
    good = [c for c in G["cases"] if c["name"] != "failure"]
    txt = "; ".join(f"{c['name']}: {100 * c['fg4']:.0f}% of the CAM mass lies on garment pixels vs {100 * c['area']:.0f}% garment area (ratio {c['fg4'] / max(c['area'], 1e-6):.2f}, L4)" for c in good)
    rep.para(f"(b) Quantitative check: {txt}. A ratio > 1 means the model attends to the garment more than chance; ratio ~1 means attention is no better than uniform. "
             "Whether the highlighted part is a sleeve, collar, hem or the whole silhouette must be read from Fig 12; with a 2x2 layer4 map only a coarse half/quadrant "
             "of the garment can be resolved, and the layer3 map (4x4) gives somewhat finer localisation.")
    fc = next((c for c in G["cases"] if c["name"] == "failure"), None)
    if fc:
        rep.para(f"(c) Failure case: test image #{fc['idx']}, true {CLASSES[fc['true']]}, predicted {CLASSES[fc['pred']]} (confidence {100 * fc['conf']:.1f}%). "
                 f"{100 * fc['fg4']:.0f}% of its Grad-CAM mass (L4) lies on garment pixels (garment covers {100 * fc['area']:.0f}% of the image), so the model "
                 f"{'did look at the garment' if fc['fg4'] > 0.6 else 'spread attention onto background'} rather than at nothing meaningful. "
                 f"This (true, predicted) pair {'IS' if G['fail_in_pair'] else 'is not'} the most confused pair from Task 5(b) ({CLASSES[a]} <-> {CLASSES[b_]}). "
                 "Where exactly the heat falls (shared silhouette vs a misleading detail) is shown in the last row of Fig 12. What it adds beyond the confusion matrix: "
                 "the matrix says THAT these classes are confused; the CAM shows what evidence the model used for the wrong class.")
    else:
        rep.para("(c) No misclassified test image was found, so no failure case could be analysed.")
    rep.para("(d) Caveat: Grad-CAM is a coarse, class-sensitivity visualisation of one late layer (low-resolution map, upsampled and normalised per image); a heatmap on the "
             "garment shows where gradient signal is large, not that the model uses sleeves or collars - it can attend to the right region yet rely on texture, brightness "
             "or correlated cues, and it is not a causal test (no intervention such as occluding the region).")

    # ---------------- Task 7
    rep.heading("7. Discussion")
    rep.para(f"(a) Augmentation changed the train-val gap from {g_off:.2f} pp to {g_on:.2f} pp (mean of last 3 epochs) and final best validation accuracy by {dp(on['best_val_acc'] - off['best_val_acc'])} "
             f"({pp(off['best_val_acc'])} -> {pp(on['best_val_acc'])}); training loss moved {tl_off:.4f} -> {tl_on:.4f}.")
    rep.para(f"(b) Best transfer strategy on validation: {top} ({pp(TL['recs'][top]['best_val_acc'])}). With only 10k images and 64x64 inputs, pretrained features are good generic "
             "edge/texture detectors, so adapting few parameters is data-efficient, while unfreezing everything has more variance; the observed ranking in Task 4 is what the data "
             "gave, and differences of a few tenths of a point between strategies are within run-to-run noise (single seed).")
    rep.para(f"(c) Most confused: {CLASSES[a]} <-> {CLASSES[b_]} and {CLASSES[a2]} <-> {CLASSES[b2]}. {expl} The model relies mainly on global shape/silhouette, which is "
             "insufficient when classes share it.")
    rep.para("(d) Grad-CAM: see the garment/area ratios in Task 6(b); heatmaps are coarse because layer4 is 2x2 at 64x64, so 'meaningful regions' can only be judged at the level of garment "
             "halves/quadrants.")
    rep.para("(e) The failure case shows what the model attended to when it was wrong (see Task 6(c)), which the confusion matrix, an aggregate of counts, cannot show.")
    rep.para(f"(f) Both Sobel and the first conv layer compute Eq. (1) with a 3x3 kernel - a weighted sum of a pixel neighbourhood, i.e. a linear shift-invariant FIR filter - "
             f"and the Sobel kernels are special cases of what the layer can represent ({R['kstats']['edge']} of {R['kstats']['n']} learned kernels are Sobel-like). The difference: "
             "the learned kernels are optimised jointly with the rest of the network for the classification loss, so they include oriented, mixed smoothing/derivative and "
             "non-zero-DC kernels no engineer specified, and they feed a ReLU non-linearity and deeper layers that compose them into larger-scale features - a fixed linear "
             "hand-designed bank cannot adapt to the data or be followed by learned non-linear composition.")

    # ---------------- Bonus
    if R.get("bonus"):
        rep.heading("8. Bonus: CNN denoising")
        rep.image(P["t8"], "Fig 13. Clean / noisy / CNN-denoised / 3x3 Gaussian examples for sigma = 0.1 and 0.3.")
        hdr = ["sigma", "PSNR noisy", "PSNR CNN", "PSNR Gaussian", "edge mag clean", "noisy", "CNN", "Gaussian"]
        rows = [[s, f"{d['psnr_noisy']:.2f}", f"{d['psnr_cnn']:.2f}", f"{d['psnr_blur']:.2f}", f"{d['edge_clean']:.4f}", f"{d['edge_noisy']:.4f}", f"{d['edge_cnn']:.4f}", f"{d['edge_blur']:.4f}"]
                for s, d in R["bonus"].items()]
        rep.mono(fmt_table(hdr, rows))
        rep.para("Test-set PSNR in dB (higher is better); 'edge mag' = mean Sobel gradient magnitude, compared with the clean images. A fixed 3x3 Gaussian is a linear low-pass filter: "
                 "it removes noise but cannot tell noise from edges, so it attenuates both (edge magnitude falls below clean). The CNN is non-linear and learned from clean/noisy pairs, "
                 "so it can smooth flat regions yet preserve garment contours, keeping edge magnitude closer to the clean value (compare the columns above).")
    rep.close()
