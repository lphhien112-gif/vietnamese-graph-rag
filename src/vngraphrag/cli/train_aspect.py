"""CLI: HUẤN LUYỆN BiLSTM aspect classifier ngay trong repo (không cần notebook).

    python -m vngraphrag.cli.train_aspect            # 5 epoch
    python -m vngraphrag.cli.train_aspect --epochs 8

Lưu `artifacts/aspect_clf.pt` -> deploy qua `POST /classify` và làm giàu KG.
Dùng cùng kiến trúc `_build_model` mà `AspectClassifier.load` nạp -> khớp tuyệt đối.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from ..config import Config
from ..core import ASPECTS, load_visfd, preprocess_vietnamese

MAXLEN = 40
EMB, HID = 100, 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    from ..core.aspect_clf import ASPECT_CLF_FILE, _build_model

    cfg = Config.load(args.config)
    train = load_visfd(cfg.data_dir, "Train.csv")
    dev = load_visfd(cfg.data_dir, "Dev.csv")
    print(f"Train {len(train)} / Dev {len(dev)} — tiền xử lý...")
    train["clean"] = train["comment"].map(preprocess_vietnamese)
    dev["clean"] = dev["comment"].map(preprocess_vietnamese)

    freq = Counter(w for t in train["clean"] for w in t.split())
    itos = ["<pad>", "<unk>"] + [w for w, _ in freq.most_common(8000)]
    stoi = {w: i for i, w in enumerate(itos)}

    def enc(t: str) -> list[int]:
        ids = [stoi.get(w, 1) for w in t.split()[:MAXLEN]]
        return ids + [0] * (MAXLEN - len(ids))

    def multihot(asps):
        v = torch.zeros(len(ASPECTS))
        for a in asps:
            if a in ASPECTS:
                v[ASPECTS.index(a)] = 1.0
        return v

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xtr = torch.tensor([enc(t) for t in train["clean"]]).to(device)
    ytr = torch.stack([multihot(a) for a in train["aspects"]]).to(device)
    xdv = torch.tensor([enc(t) for t in dev["clean"]]).to(device)
    ydv = torch.stack([multihot(a) for a in dev["aspects"]]).to(device)

    model = _build_model(len(itos), EMB, HID, len(ASPECTS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    bs = 128
    f1 = 0.0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            lossf(model(xtr[idx]), ytr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = (torch.sigmoid(model(xdv)) > 0.5).cpu().bool()
        g = ydv.cpu().bool()
        tp = (g & pred).sum().item()
        fp = (pred & ~g).sum().item()
        fn = (g & ~pred).sum().item()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

        # Per-aspect F1 + macro-F1
        per_f1 = []
        for ai, asp in enumerate(ASPECTS):
            a_g = g[:, ai]
            a_p = pred[:, ai]
            a_tp = (a_g & a_p).sum().item()
            a_fp = (a_p & ~a_g).sum().item()
            a_fn = (a_g & ~a_p).sum().item()
            a_pr = a_tp / (a_tp + a_fp) if a_tp + a_fp else 0.0
            a_rc = a_tp / (a_tp + a_fn) if a_tp + a_fn else 0.0
            a_f1 = 2 * a_pr * a_rc / (a_pr + a_rc) if a_pr + a_rc else 0.0
            per_f1.append(a_f1)
        macro_f1 = sum(per_f1) / len(per_f1)

        print(f"epoch {ep + 1}/{args.epochs} — micro-F1={f1:.4f}  macro-F1={macro_f1:.4f}")
        if ep == args.epochs - 1:
            print(f"\n{'Aspect':12} {'F1':>7}")
            for asp, af1 in zip(ASPECTS, per_f1):
                print(f"  {asp:12} {af1:.4f}")

    out = Path(cfg.artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "itos": itos,
            "aspects": ASPECTS,
            "maxlen": MAXLEN,
            "emb": EMB,
            "hid": HID,
        },
        out / ASPECT_CLF_FILE,
    )
    print(f"Saved -> {out / ASPECT_CLF_FILE}  (micro-F1={f1:.4f}) — deploy: make api")


if __name__ == "__main__":
    main()
