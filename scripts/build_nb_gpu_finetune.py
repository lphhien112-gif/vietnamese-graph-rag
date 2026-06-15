"""Dựng notebook GPU: vá các gap CẦN GPU mà báo cáo nêu (Hạn chế/Hướng phát triển):
  1. Fine-tune PhoBERT cho phân loại aspect — class-weighted, sửa STORAGE F1=0.
  2. SimCSE (contrastive) fine-tune encoder PhoBERT cho retrieval — đo MRR trước/sau.

Notebook CHỈ in log/kết quả để người dùng tự đánh giá (không có cell nhận xét).
Ghi ra: notebooks/gpu_finetune_improvements.ipynb
Chạy:  python scripts/build_nb_gpu_finetune.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("""# GPU fine-tune — fix STORAGE F1=0 (classifier) + SimCSE encoder cho retrieval
Notebook chạy trên **GPU** (Kaggle T4 / Colab). Mỗi phần **in kết quả** để đánh giá.""")

# ── Bootstrap ────────────────────────────────────────────────────────────────
md("""## 0. Bootstrap (Kaggle/Colab/local tự nhận diện) — cần GPU""")
code('''import os, subprocess, sys
ON_KAGGLE = os.path.exists('/kaggle')
REPO = 'https://github.com/lphhien112-gif/vietnamese-graph-rag.git'
if ON_KAGGLE:
    subprocess.run('pip install -q underthesea transformers', shell=True)
    if not os.path.isdir('/kaggle/working/vietnamese-graph-rag'):
        subprocess.run(f'cd /kaggle/working && git clone -q {REPO}', shell=True)
    os.chdir('/kaggle/working/vietnamese-graph-rag')
elif not os.path.isdir('src/vngraphrag'):
    # Colab hoặc nơi khác: clone nếu chưa có repo
    if not os.path.isdir('vietnamese-graph-rag'):
        subprocess.run(f'git clone -q {REPO}', shell=True)
    os.chdir('vietnamese-graph-rag')
sys.path.insert(0, 'src')

import torch
print('torch', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('DEVICE =', DEVICE)
assert torch.cuda.is_available(), 'Notebook này cần GPU. Bật GPU accelerator rồi Run All.'
''')

# ── Phần 1: dữ liệu ──────────────────────────────────────────────────────────
md("""## 1. Dữ liệu UIT-ViSFD + phân bố nhãn (STORAGE hiếm cỡ nào)""")
code('''import numpy as np, pandas as pd
from vngraphrag.core import ASPECTS, load_visfd, preprocess_vietnamese
from underthesea import word_tokenize

train = load_visfd('data/raw', 'Train.csv')
dev   = load_visfd('data/raw', 'Dev.csv')
print('Train', len(train), '| Dev', len(dev), '| aspects', ASPECTS)

def multihot(asps):
    v = np.zeros(len(ASPECTS), dtype='float32')
    for a in asps:
        if a in ASPECTS: v[ASPECTS.index(a)] = 1.0
    return v

Ytr = np.stack([multihot(a) for a in train['aspects']])
Ydv = np.stack([multihot(a) for a in dev['aspects']])
print('\\nPhân bố nhãn train (số review có aspect):')
for ai,a in enumerate(ASPECTS):
    print(f'  {a:12} {int(Ytr[:,ai].sum()):5}  ({100*Ytr[:,ai].mean():.1f}%)')
print(f'\\nSTORAGE chỉ {int(Ytr[:,ASPECTS.index("STORAGE")].sum())} review -> lớp hiếm nhất')
''')

# ── Phần 2: PhoBERT classifier class-weighted ────────────────────────────────
md("""## 2. Fine-tune PhoBERT phân loại aspect — class-weighted (vá STORAGE F1=0)""")
code('''from transformers import AutoModel, AutoTokenizer
import torch.nn as nn

MODEL = 'vinai/phobert-base-v2'
MAXLEN = 128
tok = AutoTokenizer.from_pretrained(MODEL)

def seg_batch(texts):
    return [word_tokenize(str(t)[:256], format='text') for t in texts]

def encode(texts):
    enc = tok(seg_batch(list(texts)), padding='max_length', truncation=True,
              max_length=MAXLEN, return_tensors='pt')
    return enc['input_ids'], enc['attention_mask']

print('Tokenizing (PhoBERT)...')
Xtr_ids, Xtr_mask = encode(train['comment'])
Xdv_ids, Xdv_mask = encode(dev['comment'])
ytr = torch.tensor(Ytr); ydv = torch.tensor(Ydv)
print('train ids', tuple(Xtr_ids.shape), '| dev ids', tuple(Xdv_ids.shape))

class PhoBERTAspect(nn.Module):
    def __init__(self, n_aspect):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL)
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_aspect)
    def forward(self, ids, mask):
        h = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.fc(self.drop(pooled))

model = PhoBERTAspect(len(ASPECTS)).to(DEVICE)

# pos_weight = (#âm / #dương) mỗi lớp -> upweight lớp hiếm (STORAGE)
pos = ytr.sum(0); neg = ytr.shape[0] - pos
pos_weight = (neg / pos.clamp(min=1)).to(DEVICE)
print('pos_weight mỗi aspect:', {a: round(float(w),1) for a,w in zip(ASPECTS, pos_weight)})
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
''')

code('''def per_aspect_f1(g, p):
    g = g.bool().cpu().numpy(); p = p.bool().cpu().numpy()
    tpT=fpT=fnT=0; per={}
    for ai,a in enumerate(ASPECTS):
        tp=int((g[:,ai]&p[:,ai]).sum()); fp=int((~g[:,ai]&p[:,ai]).sum()); fn=int((g[:,ai]&~p[:,ai]).sum())
        tpT+=tp; fpT+=fp; fnT+=fn
        pr=tp/(tp+fp) if tp+fp else 0.0; rc=tp/(tp+fn) if tp+fn else 0.0
        per[a]=2*pr*rc/(pr+rc) if pr+rc else 0.0
    mp=tpT/(tpT+fpT) if tpT+fpT else 0.0; mr=tpT/(tpT+fnT) if tpT+fnT else 0.0
    micro=2*mp*mr/(mp+mr) if mp+mr else 0.0
    macro=sum(per.values())/len(per)
    return micro, macro, per

@torch.no_grad()
def evaluate():
    model.eval(); preds=[]
    for i in range(0, len(Xdv_ids), 64):
        ids=Xdv_ids[i:i+64].to(DEVICE); mask=Xdv_mask[i:i+64].to(DEVICE)
        preds.append((torch.sigmoid(model(ids,mask))>0.5).cpu())
    p=torch.cat(preds)
    return per_aspect_f1(ydv, p)

EPOCHS=4; BS=16
for ep in range(EPOCHS):
    model.train(); perm=torch.randperm(len(Xtr_ids)); tot=0.0
    for i in range(0, len(Xtr_ids), BS):
        idx=perm[i:i+BS]
        ids=Xtr_ids[idx].to(DEVICE); mask=Xtr_mask[idx].to(DEVICE); y=ytr[idx].to(DEVICE)
        opt.zero_grad(); loss=loss_fn(model(ids,mask), y); loss.backward(); opt.step()
        tot+=loss.item()*len(idx)
    micro,macro,per=evaluate()
    print(f'epoch {ep+1}/{EPOCHS}  loss={tot/len(Xtr_ids):.4f}  micro-F1={micro:.4f}  macro-F1={macro:.4f}  STORAGE-F1={per["STORAGE"]:.4f}')
''')

md("""### 2.1 Kết quả per-aspect: PhoBERT (mới) vs BiLSTM baseline""")
code('''micro,macro,per = evaluate()
# BiLSTM baseline (từ artifacts/metrics.json của repo) để so sánh
BILSTM = {'micro':0.7907,'macro':0.6776,'per':{'SCREEN':0.5079,'CAMERA':0.8313,'BATTERY':0.9042,
 'PERFORMANCE':0.7916,'STORAGE':0.0,'DESIGN':0.6369,'PRICE':0.7899,'GENERAL':0.809,
 'FEATURES':0.7629,'SER&ACC':0.7425}}
print(f'{"Aspect":12} {"BiLSTM-F1":>10} {"PhoBERT-F1":>11} {"Δ":>8}')
for a in ASPECTS:
    b=BILSTM['per'][a]; n=per[a]
    print(f'  {a:12} {b:10.4f} {n:11.4f} {n-b:+8.4f}')
print('-'*45)
print(f'  {"micro-F1":12} {BILSTM["micro"]:10.4f} {micro:11.4f} {micro-BILSTM["micro"]:+8.4f}')
print(f'  {"macro-F1":12} {BILSTM["macro"]:10.4f} {macro:11.4f} {macro-BILSTM["macro"]:+8.4f}')
print(f'  {"STORAGE-F1":12} {BILSTM["per"]["STORAGE"]:10.4f} {per["STORAGE"]:11.4f} {per["STORAGE"]-0.0:+8.4f}')
''')

code('''import os
os.makedirs('artifacts', exist_ok=True)
torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},
            'aspects':ASPECTS,'maxlen':MAXLEN,'model':MODEL}, 'artifacts/phobert_aspect.pt')
print('Saved -> artifacts/phobert_aspect.pt  (micro-F1=%.4f macro-F1=%.4f)' % (micro,macro))
''')

# ── Phần 3: SimCSE encoder cho retrieval ─────────────────────────────────────
md("""## 3. SimCSE (contrastive) fine-tune encoder PhoBERT cho retrieval — MRR trước/sau""")
code('''from vngraphrag.core import DocumentIndex
from vngraphrag.cli.evaluate import EVAL_QUERIES

index = DocumentIndex.load('artifacts', MODEL)
assert index is not None, 'Chưa có index. Chạy `make index` trước (cần artifacts/doc_vectors.npy + records.json).'
records = index.records
corpus = [r['raw'] for r in records]
gold = [r['gold'] for r in records]
print('corpus', len(corpus), '| eval queries', len(EVAL_QUERIES))

class SimCSEEncoder(nn.Module):
    def __init__(self, name):
        super().__init__(); self.bert=AutoModel.from_pretrained(name)
    def forward(self, ids, mask):
        h=self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        m=mask.unsqueeze(-1).float(); return (h*m).sum(1)/m.sum(1).clamp(min=1e-9)

@torch.no_grad()
def encode_corpus(enc_model, texts, bs=64):
    enc_model.eval(); out=[]
    for i in range(0, len(texts), bs):
        ids=tok(seg_batch(texts[i:i+bs]), padding=True, truncation=True, max_length=MAXLEN, return_tensors='pt')
        v=enc_model(ids['input_ids'].to(DEVICE), ids['attention_mask'].to(DEVICE)).cpu().numpy()
        out.append(v)
    M=np.vstack(out).astype('float32'); M/=np.clip(np.linalg.norm(M,axis=1,keepdims=True),1e-9,None)
    return M

def mrr_p5(encoder_model):
    D=encode_corpus(encoder_model, corpus); Q=encode_corpus(encoder_model, [q for q,_ in EVAL_QUERIES])
    mrr=p5=0.0
    for (_,asp),qv in zip(EVAL_QUERIES, Q):
        order=np.argsort(D@qv)[::-1]
        p5+=sum(1 for i in order[:5] if asp in gold[i])/5
        for r,i in enumerate(order,1):
            if asp in gold[i]: mrr+=1.0/r; break
    n=len(EVAL_QUERIES); return mrr/n, p5/n

base = SimCSEEncoder(MODEL).to(DEVICE)
mrr_b, p5_b = mrr_p5(base)
print(f'[BASELINE PhoBERT thuần]  MRR={mrr_b:.4f}  P@5={p5_b:.4f}')
''')

code('''# SimCSE không giám sát: encode mỗi câu HAI lần (dropout khác nhau) -> cặp dương, in-batch negatives.
import torch.nn.functional as F
rng = np.random.default_rng(42)
sample = [corpus[i] for i in rng.choice(len(corpus), size=min(8000,len(corpus)), replace=False)]
enc_model = base  # fine-tune tiếp từ baseline
enc_model.train()
opt2 = torch.optim.AdamW(enc_model.parameters(), lr=3e-5)
TEMP=0.05; BS=64; EPOCHS_SC=1
for ep in range(EPOCHS_SC):
    perm=rng.permutation(len(sample)); tot=0.0; nb=0
    for i in range(0, len(sample), BS):
        batch=[sample[j] for j in perm[i:i+BS]]
        enc=tok(seg_batch(batch), padding=True, truncation=True, max_length=MAXLEN, return_tensors='pt')
        ids=enc['input_ids'].to(DEVICE); mask=enc['attention_mask'].to(DEVICE)
        z1=F.normalize(enc_model(ids,mask),dim=1); z2=F.normalize(enc_model(ids,mask),dim=1)
        sim=(z1@z2.T)/TEMP
        labels=torch.arange(z1.size(0), device=DEVICE)
        loss=F.cross_entropy(sim, labels)
        opt2.zero_grad(); loss.backward(); opt2.step()
        tot+=loss.item(); nb+=1
    print(f'[SimCSE] epoch {ep+1}/{EPOCHS_SC}  loss={tot/max(nb,1):.4f}')

mrr_a, p5_a = mrr_p5(enc_model)
print(f'\\n[BEFORE] MRR={mrr_b:.4f}  P@5={p5_b:.4f}')
print(f'[AFTER ] MRR={mrr_a:.4f}  P@5={p5_a:.4f}')
print(f'[Δ     ] MRR={mrr_a-mrr_b:+.4f}  P@5={p5_a-p5_b:+.4f}')
torch.save(enc_model.state_dict(), 'artifacts/phobert_simcse.pt')
print('Saved -> artifacts/phobert_simcse.pt')
''')

nb["cells"] = cells
out = ROOT / "notebooks" / "gpu_finetune_improvements.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(out))
print(f"✅ Đã ghi {out}  ({len(cells)} cells)")
