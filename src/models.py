"""Models: custom CNN, MLP baseline, ResNet-18 setup, analytic bookkeeping (Task 2b/2c)."""
import torch
import torch.nn as nn
from torchvision import models

CHANNELS = (32, 64, 128, 256)


def conv_out(s, k=3, st=1, p=1):
    return (s + 2 * p - k) // st + 1


class CustomCNN(nn.Module):
    """[Conv3x3(+BN)+ReLU+(Pool)] x n_blocks -> Flatten -> FC -> ReLU -> Dropout -> FC."""

    def __init__(self, n_blocks=2, pool="max", fc_hidden=128, dropout=0.3, use_bn=True,
                 num_classes=10, in_size=28):
        super().__init__()
        assert pool in ("max", "avg", "stride") and 1 <= n_blocks <= 4
        self.cfg = dict(n_blocks=n_blocks, pool=pool, fc_hidden=fc_hidden, dropout=dropout,
                        use_bn=use_bn, num_classes=num_classes, in_size=in_size)
        layers, cin, s = [], 1, in_size
        for c in CHANNELS[:n_blocks]:
            st = 2 if pool == "stride" else 1
            layers.append(nn.Conv2d(cin, c, 3, stride=st, padding=1))
            s = conv_out(s, 3, st, 1)
            if use_bn:
                layers.append(nn.BatchNorm2d(c))
            layers.append(nn.ReLU(inplace=True))
            if pool in ("max", "avg"):
                layers.append(nn.MaxPool2d(2) if pool == "max" else nn.AvgPool2d(2))
                s //= 2
            cin = c
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(cin * s * s, fc_hidden),
                                        nn.ReLU(inplace=True), nn.Dropout(dropout),
                                        nn.Linear(fc_hidden, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class MLP(nn.Module):
    def __init__(self, hidden=256, in_dim=784, num_classes=10, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                 nn.Linear(hidden, num_classes))

    def forward(self, x):
        return self.net(x)


def mlp_hidden_for(target_params, in_dim=784, k=10):
    """Hidden width h whose 784-h-h-10 MLP has a parameter count closest to target_params."""
    return min(range(16, 4096), key=lambda h: abs(in_dim * h + h + h * h + h + h * k + k - target_params))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------- analytic bookkeeping
def hand_table(cfg):
    """Layer table computed from formulas only (k*k*Cin*Cout + Cout, 2C for BN, in*out+out)."""
    rows, shapes = [], []
    s, cin = cfg["in_size"], 1
    for i, c in enumerate(CHANNELS[:cfg["n_blocks"]], 1):
        st = 2 if cfg["pool"] == "stride" else 1
        o = conv_out(s, 3, st, 1)
        rows.append([f"Conv{i}", f"{cin}x{s}x{s}", "3x3", st, 1, c, f"{c}x{o}x{o}", 9 * cin * c + c])
        shapes.append((c, o, o))
        if cfg["use_bn"]:
            rows.append([f"BN{i}", f"{c}x{o}x{o}", "-", "-", "-", "-", f"{c}x{o}x{o}", 2 * c])
        rows.append([f"ReLU{i}", f"{c}x{o}x{o}", "-", "-", "-", "-", f"{c}x{o}x{o}", 0])
        s = o
        if cfg["pool"] in ("max", "avg"):
            o = s // 2
            rows.append([("MaxPool" if cfg["pool"] == "max" else "AvgPool") + str(i), f"{c}x{s}x{s}",
                         "2x2", 2, 0, "-", f"{c}x{o}x{o}", 0])
            shapes.append((c, o, o))
            s = o
        cin = c
    flat, h, k = cin * s * s, cfg["fc_hidden"], cfg["num_classes"]
    rows.append(["Flatten", f"{cin}x{s}x{s}", "-", "-", "-", "-", f"{flat}", 0])
    rows.append(["FC1", f"{flat}", "-", "-", "-", h, f"{h}", flat * h + h]); shapes.append((h,))
    rows.append(["ReLU", f"{h}", "-", "-", "-", "-", f"{h}", 0])
    rows.append(["Dropout", f"{h}", "-", "-", "-", "-", f"{h}", 0])
    rows.append(["FC2", f"{h}", "-", "-", "-", k, f"{k}", h * k + k]); shapes.append((k,))
    return rows, sum(r[-1] for r in rows), shapes


def measured_out_shapes(model):
    shapes, hooks = [], []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d, nn.Linear)):
            hooks.append(m.register_forward_hook(lambda mod, i, o: shapes.append(tuple(o.shape[1:]))))
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        model(torch.zeros(1, 1, model.cfg["in_size"], model.cfg["in_size"], device=dev))
    for h in hooks:
        h.remove()
    return shapes


def receptive_field(cfg):
    """r_out = r_in + (k-1)*j_in ; j_out = j_in*stride. Returns (trace, RF of last conv unit)."""
    r, j, trace, last = 1, 1, [], 1
    for i in range(1, cfg["n_blocks"] + 1):
        st = 2 if cfg["pool"] == "stride" else 1
        r += 2 * j; j *= st
        trace.append((f"Conv{i} (3x3,s{st})", r, j)); last = r
        if cfg["pool"] in ("max", "avg"):
            r += 1 * j; j *= 2
            trace.append((f"Pool{i} (2x2,s2)", r, j))
    return trace, last


# ---------------------------------------------------------------- ResNet-18
STRATEGIES = ("feature_extraction", "gradual_unfreeze", "diff_lr")


def make_resnet18(strategy="feature_extraction", num_classes=10, pretrained=True):
    m = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    for p in m.parameters():
        p.requires_grad = False
    if strategy == "feature_extraction":
        train = [m.fc]
    elif strategy == "gradual_unfreeze":
        train = [m.layer4, m.fc]
    elif strategy == "diff_lr":
        train = [m]
    else:
        raise ValueError(strategy)
    for mod in train:
        for p in mod.parameters():
            p.requires_grad = True
    return m


def param_groups(model, strategy, lr):
    if strategy == "diff_lr":
        backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
        return [{"params": backbone, "lr": 0.1 * lr}, {"params": list(model.fc.parameters()), "lr": lr}]
    return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]


def verify_resnet_input(model, size=64):
    """Returns (layer4 feature-map shape, logits shape) for one size x size RGB input."""
    model.eval()
    body = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                         model.layer1, model.layer2, model.layer3, model.layer4)
    with torch.no_grad():
        x = torch.zeros(1, 3, size, size)
        return tuple(body(x).shape), tuple(model(x).shape)


def build_from_rec(rec):
    if rec["kind"] == "cnn":
        m = CustomCNN(**rec["cfg"])
    elif rec["kind"] == "mlp":
        m = MLP(**rec["cfg"])
    else:
        m = make_resnet18(rec["cfg"]["strategy"], pretrained=False)
    m.load_state_dict(rec["state"])
    return m
