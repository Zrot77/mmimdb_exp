"""v9 — 처방→결손 붕괴 연결 검증.
한 파이프라인(frozen CLIP 특징 + proj + fusion 분류기)에서 처방을 넣고
(a) suff(z) 회복 여부와 (b) 결손 성능(text/image 결손) 개선 여부를 동시에 측정.
→ H-연결(누수와 붕괴가 같은 뿌리) vs H-분리(붕괴는 내재적 약함) 판정.

처방(--presc):
  none  : 정렬(InfoNCE)+BCE만 (baseline; eta=0이라 결손 붕괴 보임)
  noise : 정렬 직전 feature(pi,pt)에 Gaussian noise 상시 주입 (우리 이중해리 개입의 처방 승격)
  gamma : 각 모달 표현에 label head(BCE)로 sufficiency 직접 강제
  mmr   : 결손 표현의 분류 logit을 full과 같게 강제(결손 직접 겨냥, 비교군)

실행:
  python train_v9.py --mode synthetic --smoke
  python train_v9.py --mode real --feat_dir feats --presc noise --presc_sigma 0.3 --seed 1
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data_features import get_split, train_missing_mask
from utils import info_nce, align_cos, f1_all, multilabel_probe, N_GENRE


class MMHeadV9(nn.Module):
    def __init__(self, in_dim=512, dim=256, n_cls=N_GENRE):
        super().__init__()
        self.pi = nn.Linear(in_dim, dim)
        self.pt = nn.Linear(in_dim, dim)
        self.cls = nn.Linear(2 * dim, n_cls)
        self.head_i = nn.Linear(dim, n_cls)   # gamma 처방용
        self.head_t = nn.Linear(dim, n_cls)

    def proj(self, zi, zt):
        return self.pi(zi), self.pt(zt)

    def classify(self, pi, pt):
        return self.cls(torch.cat([pi, pt], dim=1))


def evaluate(net, zi, zt, y, device):
    """full/image결손/text결손/both결손 각각 F1(macro/micro/weighted)."""
    net.eval()
    out = {}
    with torch.no_grad():
        pi, pt = net.proj(zi.to(device), zt.to(device))
        z0i, z0t = torch.zeros_like(pi), torch.zeros_like(pt)
        regimes = {"full": (pi, pt), "image": (z0i, pt), "text": (pi, z0t), "both": (z0i, z0t)}
        yt = y.to(device)
        for r, (a, b) in regimes.items():
            out[r] = f1_all(net.classify(a, b), yt)
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
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--w_align", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=0.0, help="결손학습률(v9 baseline은 0 → 붕괴 노출)")
    ap.add_argument("--presc", choices=["none", "noise", "gamma", "combo", "mmr"], default="none")
    ap.add_argument("--presc_sigma", type=float, default=0.3, help="noise 처방 σ")
    ap.add_argument("--gamma", type=float, default=1.0, help="gamma 처방 가중치")
    ap.add_argument("--mmr_w", type=float, default=1.0, help="mmr 처방 가중치")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/v9.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 5
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    zi_tr, zt_tr, y_tr = get_split(args.mode, args.feat_dir, "train", args.seed)
    zi_te, zt_te, y_te = get_split(args.mode, args.feat_dir, "test", args.seed)
    print("=== v9 | presc={} | mode={} train={} test={} | dev={} ===".format(
        args.presc, args.mode, len(y_tr), len(y_te), device))

    net = MMHeadV9(zi_tr.size(1), args.dim, N_GENRE).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(zi_tr, zt_tr, y_tr), batch_size=args.batch, shuffle=True)

    for ep in range(args.epochs):
        net.train(); tot = 0.0
        for zi, zt, y in loader:
            zi, zt, y = zi.to(device), zt.to(device), y.to(device)
            pi, pt = net.proj(zi, zt)
            if args.presc in ("noise", "combo"):                   # 처방 A: feature noise
                pi = pi + args.presc_sigma * torch.randn_like(pi)
                pt = pt + args.presc_sigma * torch.randn_like(pt)
            lc = info_nce(pi, pt, args.temp)
            pim, ptm = train_missing_mask(pi, pt, args.eta)
            loss = (F.binary_cross_entropy_with_logits(net.classify(pim, ptm), y)
                    + args.w_align * lc + args.beta * align_cos(pi, pt))
            if args.presc in ("gamma", "combo"):                   # 처방 B: sufficiency head
                loss = loss + args.gamma * (
                    F.binary_cross_entropy_with_logits(net.head_i(pi), y)
                    + F.binary_cross_entropy_with_logits(net.head_t(pt), y))
            if args.presc == "mmr":                                # 처방 C: 결손=full 강제
                full = net.classify(pi, pt).detach()
                zi0, zt0 = torch.zeros_like(pi), torch.zeros_like(pt)
                loss = loss + args.mmr_w * (
                    F.mse_loss(net.classify(zi0, pt), full)        # image 결손 → full
                    + F.mse_loss(net.classify(pi, zt0), full))     # text 결손 → full
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())
        if (ep + 1) % 10 == 0 or ep == 0 or args.smoke:
            print("  epoch {:>3d}  loss={:.4f}".format(ep + 1, tot / len(loader)))

    ev = evaluate(net, zi_te, zt_te, y_te, device)
    net.eval()
    with torch.no_grad():
        pi_tr, pt_tr = net.proj(zi_tr.to(device), zt_tr.to(device))
        pi_te, pt_te = net.proj(zi_te.to(device), zt_te.to(device))
    si = multilabel_probe(pi_tr, y_tr, pi_te, y_te, device=device, seed=args.seed)["f1_macro"]
    stx = multilabel_probe(pt_tr, y_tr, pt_te, y_te, device=device, seed=args.seed)["f1_macro"]

    print("\n--- 결손 성능 (F1 macro / micro / weighted) ---")
    for r in ("full", "image", "text", "both"):
        e = ev[r]
        print("  {:6s}: {:.3f} / {:.3f} / {:.3f}".format(r, e["macro"], e["micro"], e["weighted"]))
    print("--- suff (frozen probe) ---  z_image {:.3f}  z_text {:.3f}".format(si, stx))

    res = {"presc": args.presc, "config": vars(args), "eval": ev,
           "suff": {"z_image": si, "z_text": stx}}
    json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    print(">>> saved", args.out)


if __name__ == "__main__":
    main()
