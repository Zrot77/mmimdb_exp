"""1주차 — MM-IMDb 베이스라인 파이프라인 + Sanity Check(S4 게이트).
frozen CLIP 특징 위에 (proj heads + 정렬 + 다중라벨 BCE) 학습 → full/결손 F1 + frozen probe.

실행:
  python train_week1.py --mode synthetic --smoke          # 데이터·CLIP 없이 로직 점검
  python train_week1.py --mode real --feat_dir feats --epochs 30
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data_features import get_split, apply_missing, train_missing_mask
from utils import (info_nce, supcon_multilabel, align_cos, f1_macro_micro,
                   multilabel_probe, N_GENRE)


class MMHead(nn.Module):
    """frozen CLIP 특징(512) 위의 학습 헤드: 모달별 proj + fusion 분류(concat)."""
    def __init__(self, in_dim=512, dim=256, n_cls=N_GENRE):
        super().__init__()
        self.pi = nn.Linear(in_dim, dim)
        self.pt = nn.Linear(in_dim, dim)
        self.cls = nn.Linear(2 * dim, n_cls)

    def proj(self, zi, zt):
        return self.pi(zi), self.pt(zt)

    def classify(self, pi, pt):
        return self.cls(torch.cat([pi, pt], dim=1))


def evaluate(net, zi, zt, y, device):
    """full/image-missing/text-missing 각각 F1(macro,micro)."""
    net.eval()
    out = {}
    with torch.no_grad():
        pi, pt = net.proj(zi.to(device), zt.to(device))
        for regime in ("full", "image", "text"):
            pim, ptm = apply_missing(pi, pt, regime)
            fm, fmi = f1_macro_micro(net.classify(pim, ptm), y.to(device))
            out[regime] = {"f1_macro": float(fm), "f1_micro": float(fmi)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--feat_dir", default="feats")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--beta", type=float, default=0.3, help="정렬(1-cos) 가중치")
    ap.add_argument("--w_align", type=float, default=1.0, help="contrastive 가중치")
    ap.add_argument("--contrastive", choices=["infonce", "supcon"], default="infonce")
    ap.add_argument("--eta", type=float, default=0.3, help="학습 시 결손률")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/week1.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 3
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    zi_tr, zt_tr, y_tr = get_split(args.mode, args.feat_dir, "train", args.seed)
    zi_te, zt_te, y_te = get_split(args.mode, args.feat_dir, "test", args.seed)
    print("=== Week1 | mode={} | train={} test={} | dev={} ===".format(
        args.mode, len(y_tr), len(y_te), device))

    net = MMHead(zi_tr.size(1), args.dim, N_GENRE).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(zi_tr, zt_tr, y_tr), batch_size=args.batch, shuffle=True)

    for ep in range(args.epochs):
        net.train(); tot = 0.0
        for zi, zt, y in loader:
            zi, zt, y = zi.to(device), zt.to(device), y.to(device)
            pi, pt = net.proj(zi, zt)
            if args.contrastive == "supcon":
                lc = supcon_multilabel(pi, pt, args.temp, y)
            else:
                lc = info_nce(pi, pt, args.temp)
            pim, ptm = train_missing_mask(pi, pt, args.eta)      # 결손 로버스트 분류
            lbce = F.binary_cross_entropy_with_logits(net.classify(pim, ptm), y)
            loss = lbce + args.w_align * lc + args.beta * align_cos(pi, pt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())
        if (ep + 1) % 10 == 0 or ep == 0 or args.smoke:
            print("  epoch {:>3d}  loss={:.4f}".format(ep + 1, tot / len(loader)))

    # ---- 평가: full/결손 F1 ----
    ev = evaluate(net, zi_te, zt_te, y_te, device)
    # ---- frozen probe(suff): 학습된 proj 고정 후 각 모달 z의 다중라벨 probe ----
    net.eval()
    with torch.no_grad():
        pi_tr, pt_tr = net.proj(zi_tr.to(device), zt_tr.to(device))
        pi_te, pt_te = net.proj(zi_te.to(device), zt_te.to(device))
    suff_img = multilabel_probe(pi_tr, y_tr, pi_te, y_te, device=device, seed=args.seed)
    suff_txt = multilabel_probe(pt_tr, y_tr, pt_te, y_te, device=device, seed=args.seed)

    # ---- S4 sanity 게이트 ----
    full_fm = ev["full"]["f1_macro"]
    checks = {
        "full_F1_not_collapsed": full_fm > 0.15,                       # 붕괴 아님(랜덤≈0)
        "multilabel_metrics_ok": all("f1_macro" in ev[r] for r in ev),
        "missing_degrades": (ev["image"]["f1_macro"] < full_fm - 1e-4
                             and ev["text"]["f1_macro"] < full_fm - 1e-4),
        "frozen_probe_runs": ("f1_macro" in suff_img and "f1_macro" in suff_txt),
    }

    print("\n--- 평가 (F1 macro / micro) ---")
    for r in ("full", "image", "text"):
        print("  {:6s}: macro {:.3f}  micro {:.3f}".format(r, ev[r]["f1_macro"], ev[r]["f1_micro"]))
    print("--- suff (frozen probe, F1-macro) ---")
    print("  z_image {:.3f}  z_text {:.3f}".format(suff_img["f1_macro"], suff_txt["f1_macro"]))
    print("--- S4 sanity ---")
    for k, v in checks.items():
        print("  [{}] {}".format("PASS" if v else "FAIL", k))
    gate = all(checks.values())
    print("STEP4 게이트:", "통과" if gate else "실패(원인 수정 후 재확인)")

    res = {"config": vars(args), "eval": ev,
           "suff": {"z_image": suff_img, "z_text": suff_txt},
           "sanity": checks, "gate_pass": gate}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    print(">>> saved", args.out)


if __name__ == "__main__":
    main()
