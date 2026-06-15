"""Chụp ảnh Experiment Dashboard (app/experiments_ui.py) cho báo cáo, bằng Playwright.

Tự launch Gradio (non-blocking) ở cổng 7865 rồi chụp:
  report/figures/dashboard_results.png   — tab 'Kết quả thí nghiệm' (bảng + 4 biểu đồ)
  report/figures/dashboard_compare.png   — tab 'So sánh trực tiếp' (No-RAG/Vanilla/Graph-RAG, best-effort)

Chạy: python scripts/shot_dashboard.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "scripts"))
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
PORT = 7865
URL = f"http://127.0.0.1:{PORT}"


def main() -> int:
    from playwright.sync_api import sync_playwright

    from experiments_ui import build  # type: ignore

    demo = build()
    demo.launch(server_name="127.0.0.1", server_port=PORT, share=False, prevent_thread_lock=True)
    time.sleep(2)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1300, "height": 1500}, device_scale_factor=2)
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_selector("h1", timeout=30000)
        time.sleep(4)  # chờ matplotlib plots render

        # ── Tab 1: Kết quả thí nghiệm ──
        out1 = FIG / "dashboard_results.png"
        page.screenshot(path=str(out1), full_page=True)
        print(f"✅ {out1}")

        # ── Tab 2: So sánh trực tiếp (best-effort: cần LLM + load PhoBERT) ──
        try:
            page.get_by_role("tab", name="So sánh trực tiếp (live)").click()
            time.sleep(1.5)
            page.get_by_role("button", name="So sánh 3 chế độ").click()
            # chờ 3 ô output có nội dung (tối đa ~150s vì lần đầu load PhoBERT)
            filled = False
            for _ in range(150):
                time.sleep(1)
                vals = page.locator("textarea").evaluate_all("els => els.map(e => e.value)")
                non_empty = [v for v in vals if v and v.strip()]
                if len(non_empty) >= 4:  # 1 input + 3 output
                    filled = True
                    break
            time.sleep(1.0)
            out2 = FIG / "dashboard_compare.png"
            page.screenshot(path=str(out2), full_page=True)
            print(f"{'✅' if filled else '⚠ (chưa đủ output)'} {out2}")
        except Exception as e:
            print(f"⚠ tab so sánh lỗi: {str(e)[:120]}")

        browser.close()
    demo.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
