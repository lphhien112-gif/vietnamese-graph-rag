"""Aspect classifier (BiLSTM) — HUẤN LUYỆN ở notebook §6, DEPLOY ở đây.

Nạp lại từ `artifacts/aspect_clf.pt` (state_dict + vocab + label space + hyperparams),
suy luận trên CPU. torch nạp lười để import core vẫn nhẹ (CI không cần torch).
"""

from __future__ import annotations

from pathlib import Path

ASPECT_CLF_FILE = "aspect_clf.pt"


def _build_model(vocab: int, emb: int, hid: int, n_aspect: int):
    """Dựng đúng kiến trúc BiLSTM như notebook §6 (tên layer phải khớp state_dict)."""
    import torch.nn as nn

    class BiLSTMAspect(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, emb, padding_idx=0)
            self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hid * 2, n_aspect)

        def forward(self, x):
            o, _ = self.lstm(self.emb(x))
            return self.fc(o.mean(1))

    return BiLSTMAspect()


class AspectClassifier:
    """Wrapper deploy: .predict(texts) -> list[set[str]] aspect dự đoán."""

    def __init__(self, model, stoi: dict, aspects: list[str], maxlen: int, threshold: float = 0.5):
        self.model = model
        self.stoi = stoi
        self.aspects = aspects
        self.maxlen = maxlen
        self.threshold = threshold

    def _encode(self, text: str) -> list[int]:
        # tiền xử lý GIỐNG lúc train (notebook §6 dùng clean_text) để vocab khớp
        from .data import preprocess_vietnamese

        toks = preprocess_vietnamese(text).split()[: self.maxlen]
        ids = [self.stoi.get(w, 1) for w in toks]
        return ids + [0] * (self.maxlen - len(ids))

    def predict(self, texts: list[str]) -> list[set]:
        import torch

        x = torch.tensor([self._encode(t) for t in texts])
        with torch.no_grad():
            probs = torch.sigmoid(self.model(x)).cpu().numpy()
        return [{self.aspects[i] for i, p in enumerate(row) if p >= self.threshold} for row in probs]

    @classmethod
    def load(cls, artifacts_dir: str | Path, threshold: float = 0.5) -> AspectClassifier | None:
        p = Path(artifacts_dir) / ASPECT_CLF_FILE
        if not p.exists():
            return None
        import torch

        ckpt = torch.load(p, map_location="cpu")
        model = _build_model(len(ckpt["itos"]), ckpt["emb"], ckpt["hid"], len(ckpt["aspects"]))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        stoi = {w: i for i, w in enumerate(ckpt["itos"])}
        return cls(model, stoi, ckpt["aspects"], ckpt["maxlen"], threshold)
