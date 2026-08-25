"""2주차 A2 — 누수 확인 (freeze 세팅).
천장 = raw CLIP 특징 suff (그 모달을 선형으로 최대한 뽑은 값 = 고정 인코더의 상한).
정렬후 = proj를 '순수 정렬만'(InfoNCE+정렬, 분류 없음, 압축 없이 dim=512)으로 학습 후 suff.
누수 = 천장 − 정렬후. 양수면 '정렬이 약한 모달의 essence를 깎는다'.

실행:
  python diagnose_a2.py --mode real --feat_dir feats --seeds 0,1,2
  python diagnose_a2.py --mode synthetic --seeds 0,1     # 로직 스모크
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_features import load_feats, synth_feats
from utils import info_nce, align_cos, multilabel_probe


class Proj(nn.Module):
    def __init__(self, in_dim=512, dim=512):
        super().__init__()
        self.pi = nn.Linear(in_dim, dim)
        self.pt = nn.Linear(in_dim, dim)

    def forward(self, zi, zt):
        return self.pi(zi), self.pt(zt)


def get(mode, feat_dir, split):
    if mode == "synthetic":
        return synth_feats({"train": 1500, "test": 600}[split], seed={"train": 0, "test": 2}[split])
    return load_feats(feat_dir, split)


def train_align_only(proj, zi, zt, device, epochs, lr, batch, temp, beta):
    opt = torch.optim.Adam(proj.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(zi, zt), batch_size=batch, shuffle=True)
    for _ in range(epochs):
        for bi, bt in loader:
            bi, bt = bi.to(device), bt.to(device)
            pi, pt = proj(bi, bt)
            loss = info_nce(pi, pt, temp) + beta * align_cos(pi, pt)   # 순수 정렬만(분류 없음)
            opt.zero_grad(); loss.backward(); opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthetic", "real"], default="real")
    ap.add_argument("--feat_dir", default="feats")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--dim", type=int, default=512, help="proj 차원(압축 배제 위해 512=입력과 동일)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--out", default="results/a2_leak.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in args.seeds.split(",")]

    zi_tr, zt_tr, y_tr = get(args.mode, args.feat_dir, "train")
    zi_te, zt_te, y_te = get(args.mode, args.feat_dir, "test")
    print("=== A2 누수 확인 | mode={} | train={} test={} | dev={} ===".format(
        args.mode, len(y_tr), len(y_te), device))

    ceil, aligned = {"image": [], "text": []}, {"image": [], "text": []}
    for s in seeds:
        # 천장: raw CLIP 특징 probe
        ceil["image"].append(multilabel_probe(zi_tr, y_tr, zi_te, y_te, device=device, seed=s)["f1_macro"])
        ceil["text"].append(multilabel_probe(zt_tr, y_tr, zt_te, y_te, device=device, seed=s)["f1_macro"])
        # 정렬후: 순수 정렬 proj 학습 → probe
        torch.manual_seed(s); np.random.seed(s)
        proj = Proj(zi_tr.size(1), args.dim).to(device)
        train_align_only(proj, zi_tr, zt_tr, device, args.epochs, args.lr, args.batch, args.temp, args.beta)
        proj.eval()
        with torch.no_grad():
            pi_tr, pt_tr = proj(zi_tr.to(device), zt_tr.to(device))
            pi_te, pt_te = proj(zi_te.to(device), zt_te.to(device))
        aligned["image"].append(multilabel_probe(pi_tr, y_tr, pi_te, y_te, device=device, seed=s)["f1_macro"])
        aligned["text"].append(multilabel_probe(pt_tr, y_tr, pt_te, y_te, device=device, seed=s)["f1_macro"])
        print("  seed {}: image 천장 {:.3f} / 정렬후 {:.3f} | text 천장 {:.3f} / 정렬후 {:.3f}".format(
            s, ceil["image"][-1], aligned["image"][-1], ceil["text"][-1], aligned["text"][-1]))

    def ms(a):
        return float(np.mean(a)), float(np.std(a))
    print("\n--- A2 (F1-macro, mean±std, {}seed) ---".format(len(seeds)))
    res = {}
    for m in ("image", "text"):
        cm, cs = ms(ceil[m]); am, as_ = ms(aligned[m])
        leak = cm - am
        res[m] = {"ceiling": cm, "aligned": am, "leak": leak}
        print("  {:5s}: 천장 {:.3f}±{:.3f}  정렬후 {:.3f}±{:.3f}  누수(천장-정렬후) {:+.3f}".format(
            m, cm, cs, am, as_, leak))
    print("\n판정: 약한 모달(image)의 누수가 유의(+)하면 '정렬이 약한 모달 essence를 깎는다' 지지.")
    print("      ~0이면 freeze 세팅에선 정렬이 안 깎음 → 약함은 CLIP 표현 자체의 성질(A3 방향 재검토).")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    print(">>> saved", args.out)


if __name__ == "__main__":
    main()
