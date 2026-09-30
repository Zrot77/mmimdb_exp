"""v19 TLP (test-time logit prompting) — frozen γ8 모델 위 추론 시점 적응.
원 논문(arXiv:2609.02039, Chen & Guo) 충실 이식 + 멀티라벨 적응:
  - logit prompt pool P={complete, missT, missI} 각 ℝ^C, zero-init, ẑ = z + P[cond] (backbone freeze, P만 최적화)
  - L_u: 예측을 최대불확실(멀티라벨=클래스별 0.5)로 당김, MSE
  - L_c: 결손 샘플을 배치 내 최근접 complete 샘플 예측에 정렬(예측공간 최근접 k=1, KL)
  - L = L_u + (1/C) L_c, AdamW lr1e-2, K 스텝, source-free transductive(라벨·학습데이터 불필요)
  - 멀티라벨 적응: softmax→sigmoid, uniform→0.5, KL→Bernoulli-KL, 예측→임계 0.5
게이트: γ8+TLP vs γ8(P=0)을 결손율 η 스윕에서 비교. §6 착시 점검용 complete-subset F1도 병기.

실행: python tlp_v19.py --ckpt models/v19_g8_s1.pt --raw_root ~/Yechan/mmimdb_data/mmimdb --feat_dir feats \
        --missing text --K 5 --seed 1 --out results/v19_tlp_text_K5_s1.json
"""
import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from train_v12 import V12Model, make_loader, extract_proj, bernoulli_kl
from utils import N_GENRE


def f1_macro(logits, y, thr=0.5):
    pred = (torch.sigmoid(logits) > thr).int().cpu().numpy()
    return float(f1_score(y.int().cpu().numpy(), pred, average="macro", zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--feat_dir", default="feats")
    ap.add_argument("--etas", default="0.3,0.5,0.7")
    ap.add_argument("--K", type=int, default=5, help="TLP 최적화 스텝(원 논문 1 또는 5)")
    ap.add_argument("--lam_u", type=float, default=1.0, help="L_u(불확실성) 가중치. 0=L_c만(멀티라벨 격리 검증)")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--tlp_batch", type=int, default=32)
    ap.add_argument("--missing", default="text", choices=["text", "image", "both"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/v19_tlp.json")
    cli = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cli.seed); np.random.seed(cli.seed)
    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)

    ck = torch.load(cli.ckpt, map_location="cpu")
    cfg = ck["config"]
    la = SimpleNamespace(mode="real", backbone=cfg["backbone"], siglip_name=cfg.get("siglip_name"),
                         feat_dir=cli.feat_dir, raw_root=cli.raw_root, batch=64, workers=4, limit=0)
    model = V12Model(cfg["method"], cfg["backbone"], cfg["unfreeze_last"], cfg["dim"],
                     siglip_name=cfg.get("siglip_name")).to(device)
    model.load_state_dict(ck["state"]); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print("모델 로드:", cli.ckpt, "| method", cfg["method"], "| backbone", cfg["backbone"])

    te = make_loader(la, "test", False)
    pi, pt, y = extract_proj(model, te, device)
    pi, pt, y = pi.to(device), pt.to(device), y.to(device)
    z0 = torch.zeros_like(pi)
    with torch.no_grad():
        z_c = model.classify(pi, pt)      # complete
        z_mt = model.classify(pi, z0)     # text 결손(이미지가 carry)
        z_mi = model.classify(z0, pt)     # image 결손(텍스트가 carry)
    Zc = torch.stack([z_c, z_mt, z_mi], 0)          # (3,N,C)
    C, N = N_GENRE, y.size(0)
    ar = torch.arange(N, device=device)

    results = {}
    for eta in [float(x) for x in cli.etas.split(",")]:
        g = torch.Generator().manual_seed(cli.seed)
        r = torch.rand(N, generator=g).to(device)
        cond = torch.zeros(N, dtype=torch.long, device=device)   # 0=complete 1=missT 2=missI
        if cli.missing == "text":
            cond[r < eta] = 1
        elif cli.missing == "image":
            cond[r < eta] = 2
        else:
            cond[r < eta / 2] = 1
            cond[(r >= eta / 2) & (r < eta)] = 2
        miss, comp = cond > 0, cond == 0

        def logits_with(P):
            return Zc[cond, ar] + P[cond]

        with torch.no_grad():                                    # baseline (γ8, P=0)
            bl = logits_with(torch.zeros(3, C, device=device))
            b_all = f1_macro(bl, y)
            b_miss = f1_macro(bl[miss], y[miss]) if miss.any() else float("nan")
            b_comp = f1_macro(bl[comp], y[comp]) if comp.any() else float("nan")

        P = nn.Parameter(torch.zeros(3, C, device=device))       # TLP
        opt = torch.optim.AdamW([P], lr=cli.lr)
        for _ in range(cli.K):
            perm = ar[torch.randperm(N, device=device)]
            for bs in range(0, N, cli.tlp_batch):
                b = perm[bs:bs + cli.tlp_batch]
                cb = cond[b]
                q = torch.sigmoid(Zc[cb, b] + P[cb])
                Lu = ((q - 0.5) ** 2).mean()
                mm, cm = cb > 0, cb == 0
                Lc = torch.zeros((), device=device)
                if mm.any() and cm.any():
                    qc, qm = q[cm], q[mm]
                    d = torch.cdist(qm.detach(), qc.detach())
                    anchor = qc.detach()[d.argmin(1)]
                    Lc = bernoulli_kl(anchor, qm)
                loss = cli.lam_u * Lu + (1.0 / C) * Lc
                opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            tl = logits_with(P.detach())
            t_all = f1_macro(tl, y)
            t_miss = f1_macro(tl[miss], y[miss]) if miss.any() else float("nan")
            t_comp = f1_macro(tl[comp], y[comp]) if comp.any() else float("nan")
        results["eta_%.1f" % eta] = {"baseline_all": b_all, "tlp_all": t_all,
                                     "baseline_miss": b_miss, "tlp_miss": t_miss,
                                     "baseline_comp": b_comp, "tlp_comp": t_comp}
        print("η={:.1f} | miss결손 {:.3f}→{:.3f} (Δ{:+.3f}) | all {:.3f}→{:.3f} | comp {:.3f}→{:.3f}".format(
            eta, b_miss, t_miss, t_miss - b_miss, b_all, t_all, b_comp, t_comp))

    json.dump({"ckpt": cli.ckpt, "missing": cli.missing, "K": cli.K, "seed": cli.seed, "results": results},
              open(cli.out, "w"), indent=2, default=float)
    print(">>> saved", cli.out)


if __name__ == "__main__":
    main()
