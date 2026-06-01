"""Dựng lại notebook Part 2 (Graph RAG) cho CHẠY ĐƯỢC trên Python 3.14.

Tái dùng package src/vngraphrag: Knowledge Graph, HybridRetriever, GraphRAGPipeline,
evaluate (ablation). LLM gọi qua proxy OpenAI-compatible (.env). Cell Gradio chỉ DỰNG UI
(không launch) để nbconvert không treo; hướng dẫn launch ghi trong markdown.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("""# 🛒 Vietnamese Graph RAG — Part 2: Knowledge Graph + RAG
**ĐH KHTN, ĐHQG-HCM — NLP Final Project**

Tái hiện **số liệu thật** của báo cáo:
- **Module 4** — Knowledge Graph hợp nhất (UIT-ViSFD + Shopee) + trực quan.
- **TN3** — Ablation retrieval (bi-encoder → +MaxSim → +Graph boost) bằng P@k/MRR (không rò rỉ metric).
- **TN4** — Demo end-to-end: Retrieve → Graph context → sinh câu trả lời (LLM qua proxy OpenAI-compatible).

> Tái dùng package `src/vngraphrag`. Cần đã `make index` (build index PhoBERT + KG) và có `.env` (OPENAI_API_KEY + OPENAI_BASE_URL).""")

md("""## ⚙️ Bootstrap (Kaggle / local tự nhận diện)
Trên **Kaggle**: cài thư viện + clone repo + build index/KG + train BiLSTM nếu thiếu (GPU T4) +
lấy khoá LLM từ **Add-ons → Secrets** (`OPENAI_API_KEY`, tuỳ chọn `OPENAI_BASE_URL` nếu dùng proxy).
Trên **local**: bỏ qua (đã có repo + `.env`).""")
code('''import os, subprocess
ON_KAGGLE = os.path.exists('/kaggle')
REPO = 'https://github.com/lphhien112-gif/vietnamese-graph-rag.git'
if ON_KAGGLE:
    subprocess.run('pip install -q underthesea transformers scikit-learn scipy networkx matplotlib openai gradio', shell=True)
    if not os.path.isdir('/kaggle/working/vietnamese-graph-rag'):
        subprocess.run(f'cd /kaggle/working && git clone -q {REPO}', shell=True)
    os.chdir('/kaggle/working/vietnamese-graph-rag')
    for f, cmd in [('artifacts/doc_vectors.npy','vngraphrag.cli.build_index'),
                   ('artifacts/aspect_clf.pt','vngraphrag.cli.train_aspect')]:
        if not os.path.exists(f):
            print('Đang chạy', cmd, '...'); subprocess.run(f'python -m {cmd}', shell=True)
    try:
        from kaggle_secrets import UserSecretsClient
        sec = UserSecretsClient()
        os.environ['OPENAI_API_KEY'] = sec.get_secret('OPENAI_API_KEY')
        try: os.environ['OPENAI_BASE_URL'] = sec.get_secret('OPENAI_BASE_URL')
        except Exception: pass
    except Exception as e:
        print('⚠️ Chưa có secret OPENAI_API_KEY — phần sinh câu trả lời sẽ fallback:', e)
print('Môi trường:', 'Kaggle' if ON_KAGGLE else 'local')''')

md("## 0. Thiết lập")
code("""import sys, os, json
from pathlib import Path
import numpy as np
ROOT = Path.cwd(); ROOT = ROOT if (ROOT/'src').exists() else ROOT.parent
os.chdir(ROOT)                       # path tương đối (artifacts/, data/raw) trỏ đúng repo root
sys.path.insert(0, str(ROOT/'src'))
from vngraphrag.config import Config
cfg = Config.load(ROOT/'config.yaml')
print('artifacts_dir =', cfg.artifacts_dir, '| model =', cfg.llm.model)""")

md("## 1. Knowledge Graph hợp nhất + thống kê")
code("""from vngraphrag.core import load_visfd, load_shopee, load_kg
from vngraphrag.core.aspect_clf import AspectClassifier
from collections import Counter
G = load_kg(Path(cfg.artifacts_dir)/'kg.pkl')
types = Counter(d.get('type') for _,d in G.nodes(data=True))
rels  = Counter(d.get('relation') for *_,d in G.edges(data=True))
print(f'KG: {G.number_of_nodes()} node, {G.number_of_edges()} cạnh')
print('node types:', dict(types)); print('edge relations:', dict(rels))""")

md("### 1.1 Truy vấn KG: phân bố cảm xúc theo aspect")
code("""from vngraphrag.core import graph_query
for asp in ['CAMERA','BATTERY','SER&ACC']:
    sd = graph_query(G, asp); tot = sum(v['count'] for v in sd.values())
    print(f'\\n{asp}:')
    for node, inf in sorted(sd.items(), key=lambda x:-x[1]['count']):
        print(f"  {inf['sentiment']:9} {inf['count']:5}  ({100*inf['count']/tot:.0f}%)")""")

md("### 1.2 Trực quan KG (brand → aspect → sentiment)")
code("""import matplotlib.pyplot as plt, networkx as nx, math
keep=[n for n,d in G.nodes(data=True) if d.get('type') in ('brand','aspect','sentiment')]
H=G.subgraph(keep)
col={'aspect':'#4C9BE0','sentiment':'#9FD89F','brand':'#F2C14E'}
fig,ax=plt.subplots(figsize=(14,9)); pos=nx.spring_layout(H,k=0.9,seed=7,iterations=120)
nx.draw_networkx_nodes(H,pos,node_color=[col[H.nodes[n]['type']] for n in H],
                       node_size=[400+60*H.degree(n) for n in H],ax=ax)
nx.draw_networkx_edges(H,pos,width=[0.4+math.log1p(H[u][v]['weight'])*0.6 for u,v in H.edges()],alpha=0.3,ax=ax,arrows=True)
nx.draw_networkx_labels(H,pos,font_size=8,ax=ax); ax.axis('off')
ax.set_title('Knowledge Graph: brand → aspect → sentiment'); plt.tight_layout(); plt.show()""")

md("""## 2. TN3 — Ablation retrieval (không rò rỉ metric)
Trên cùng Top-50 candidate, đổi trọng số 3 thành phần: bi-encoder · MaxSim · graph boost.
Graph boost dùng aspect *quan sát trong nội dung*; chấm điểm bằng *nhãn vàng* (độc lập → không rò rỉ).""")
code("""from vngraphrag.cli.evaluate import run_eval, CONFIGS, EVAL_QUERIES
import pandas as pd
print(f'{len(EVAL_QUERIES)} truy vấn · {len(CONFIGS)} cấu hình')
results = run_eval(cfg)
eval_df = pd.DataFrame(results).T[['P@5','P@10','MRR']]
display(eval_df)""")

code("""ax = eval_df.plot(kind='bar', figsize=(9,5), ylim=(0.75,0.90), rot=0)
ax.set_title('TN3 — Ablation: bi-encoder → +MaxSim → +Graph (Graph RAG)')
ax.set_ylabel('Điểm'); ax.grid(axis='y', alpha=0.3)
for c in ax.containers: ax.bar_label(c, fmt='%.3f', fontsize=7)
plt.tight_layout(); plt.show()
print('Graph RAG đạt MRR cao nhất =', max(r['MRR'] for r in results.values()))""")

md("""## 3. TN4 — Demo end-to-end (Retrieve → Graph context → LLM)
Pipeline đầy đủ; LLM gọi qua proxy OpenAI-compatible (đọc từ .env).""")
code("""from vngraphrag.rag.pipeline import GraphRAGPipeline
pipe = GraphRAGPipeline.from_artifacts(cfg)
print('LLM khả dụng:', pipe.generator.available, '| model:', pipe.generator.model)

for q in ["Camera điện thoại chụp đêm có tốt không?",
          "Pin Samsung dùng có lâu không?",
          "Dịch vụ bảo hành và nhân viên tư vấn thế nào?"]:
    r = pipe.answer(q)
    print('\\n❓', q)
    print('📊 KG context:', r['graph_context'].strip().replace(chr(10),' | '))
    print('💬', r['answer'])
    print(f"⏱️ {r['latency_ms']}ms")""")

md("## 4. Export artifacts (đã có sẵn từ `make index`)")
code("""import os
for f in ['doc_vectors.npy','records.json','manifest.json','kg.pkl','aspect_clf.pt','metrics.json']:
    p = Path(cfg.artifacts_dir)/f
    print(f"  {'✅' if p.exists() else '❌'} {f}" + (f"  ({os.path.getsize(p)//1024} KB)" if p.exists() else ''))""")

md("""## 5. 🎛️ Giao diện Web (Gradio)
Cell dưới chỉ **dựng** UI (không `launch()` để nbconvert không treo). Khi chạy tương tác,
bỏ comment dòng `demo.launch(share=True)` để mở link demo `*.gradio.live`.""")
code("""try:
    from app.ui import build_ui
    demo = build_ui()
    print('✅ Đã dựng Gradio UI. Để mở: demo.launch(share=True)')
    # demo.launch(share=True)   # <- bỏ comment khi chạy tương tác
except Exception as e:
    print('Gradio chưa sẵn sàng:', e)""")

nb['cells'] = cells
nb.metadata['kernelspec'] = {'display_name':'Python 3','language':'python','name':'python3'}
p = ROOT / 'notebooks' / 'kaggle_part2_graph_rag.ipynb'
nbf.write(nb, str(p))
print('Wrote', p, '—', len(cells), 'cells')
