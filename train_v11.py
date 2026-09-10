"""v11 (정식) — 처방 라운드 기준선, 파인튜닝 L0 세팅 (캐시 아님, 충실판).
이미지 타워 L0 파인튜닝(마지막 3블록) + 텍스트 고정 앵커(캐시 특징) + proj + fusion 분류기.
처방(noise/gamma/combo)을 넣고 결손 강건 성능(full/image/text/both × F1 macro/micro/weighted) 측정.
주 지표 = 결손 강건 성능, 보조 = suff. 과적합 통제 = val full-micro F1 조기중단 + weight decay.
차등 lr: CLIP 백본 1e-5, 랜덤 초기화 head(proj/분류기) 1e-3 (안 그러면 분류기 학습 안 됨).

실행:
  python train_v11.py --mode synthetic --backbone stub --smoke
  python train_v11.py --mode real --backbone clip --raw_root ~/Yechan/mmimdb_data/mmimdb \
      --feat_dir feats --presc gamma --gamma 1.0 --epochs 8 --early_stop --wd 1e-4 --seed 1
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_ft import ImageTextFT, SyntheticFT
from train_ft import StubVision
from utils import info_nce, align_cos, f1_all, multilabel_probe, N_GENRE

CLIP_NAME = "openai/clip-vit-base-patch32"


class V11Model(nn.Module):
    def __init__(self, backbone="clip", unfreeze_last=3, dim=256, n_cls=N_GENRE, clip_name=CLIP_NAME):
        super().__init__()
        self.backbone = backbone
        if backbone == "clip":
            from transformers import CLIPModel
            self.clip = CLIPModel.from_pretrained(clip_name)
            for p in self.clip.parameters():
                p.requires_grad_(False)
            for blk in self.clip.vision_model.encoder.layers[-unfreeze_last:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            for m in (self.clip.visual_projection, self.clip.vision_model.post_layernorm):
                for p in m.parameters():
                    p.requires_grad_(True)
        else:
            self.vision = StubVision(512)
        self.pi = nn.Linear(512, dim)
        self.pt = nn.Linear(512, dim)
        self.cls = nn.Linear(2 * dim, n_cls)
        self.head_i = nn.Linear(dim, n_cls)
        self.head_t = nn.Linear(dim, n_cls)

    def encode_image(self, px):
        return self.clip.get_image_features(pixel_values=px) if self.backbone == "clip" else self.vision(px)

    def proj(self, zi, zt):
        return self.pi(zi), self.pt(zt)

    def classify(self, a, b):
        return self.cls(torch.cat([a, b], dim=1))


def make_loader(args, split, shuffle):
    if args.mode == "synthetic":
        ds = SyntheticFT(n={"train": 400, "dev": 200, "test": 200}[split], seed={"train": 0, "dev": 1, "test": 2}[split])
    else:
        from transformers import CLIPProcessor
        ds = ImageTextFT(args.feat_dir, args.raw_root, split, CLIPProcessor.from_pretrained(CLIP_NAME))
    return DataLoader(ds, batch_size=args.batch, shuffle=shuffle, num_workers=args.workers)


@torch.no_grad()
def extract_proj(model, loader, device):
    """clean proj 특징 (pi, pt) + 라벨 — suff·결손 평가용 (노이즈 없이)."""
    model.eval()
    pis, pts, ys = [], [], []
    for px, txt, y in loader:
        zi = model.encode_image(px.to(device))
        pi, pt = model.proj(zi, txt.to(device))
        pis.append(pi.cpu()); pts.append(pt.cpu()); ys.append(y)
    return torch.cat(pis), torch.cat(pts), torch.cat(ys)


def eval_missing(model, loader, device):
    """test에서 full/image결손/text결손/both결손 × F1(macro/micro/weighted)."""
    pi, pt, y = extract_proj(model, loader, device)
    pi, pt, y = pi.to(device), pt.to(device), y.to(device)
    z0i, z0t = torch.zeros_like(pi), torch.zeros_like(pt)
    reg = {"full": (pi, pt), "image": (z0i, pt), "text": (pi, z0t), "both": (z0i, z0t)}
    with torch.no_grad():
        return {r: f1_all(model.classify(a, b), y) for r, (a, b) in reg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--feat_dir", default="feats")
    ap.add_argument("--raw_root", default=None)
    ap.add_argument("--backbone", choices=["clip", "stub"], default="clip")
    ap.add_argument("--unfreeze_last", type=int, default=3)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--presc", choices=["none", "noise", "gamma", "combo"], default="none")
    ap.add_argument("--presc_sigma", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--w_align", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5, help="CLIP 백본 파인튜닝 lr")
    ap.add_argument("--head_lr", type=float, default=1e-3, help="proj/분류기(랜덤 초기화) lr — 백본보다 크게")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--early_stop", action="store_true")
    ap.add_argument("--eval_every", type=int, default=4)
    ap.add_argument("--probe_subset", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/v11ft.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 2
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr = make_loader(args, "train", True)
    va = make_loader(args, "dev", False)
    te = make_loader(args, "test", False)
    model = V11Model(args.backbone, args.unfreeze_last, args.dim).to(device)
    # 차등 lr: 랜덤 초기화 head(proj·분류기)는 백본보다 크게 — 안 그러면 분류기가 학습이 안 됨
    head_params = [p for m in (model.pi, model.pt, model.cls, model.head_i, model.head_t) for p in m.parameters()]
    hid = set(id(p) for p in head_params)
    bb_params = [p for p in model.parameters() if p.requires_grad and id(p) not in hid]
    opt = torch.optim.Adam([{"params": bb_params, "lr": args.lr},
                            {"params": head_params, "lr": args.head_lr}], weight_decay=args.wd)
    print("=== v11(FT) | presc={} sig={} gam={} | 백본 {} (lr {}) + head {} (lr {}) | dev={} ===".format(
        args.presc, args.presc_sigma, args.gamma,
        sum(p.numel() for p in bb_params), args.lr, sum(p.numel() for p in head_params), args.head_lr, device))

    def val_metric():
        return eval_missing(model, va, device)["full"]["micro"]   # 주 지표(결손 F1)에 맞춘 ES

    best = {"val": -1.0, "state": None, "ep": args.epochs}
    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for px, txt, y in tr:
            px, txt, y = px.to(device), txt.to(device), y.to(device)
            zi = model.encode_image(px)
            pi, pt = model.proj(zi, txt)
            if args.presc in ("noise", "combo"):
                pi = pi + args.presc_sigma * torch.randn_like(pi)
                pt = pt + args.presc_sigma * torch.randn_like(pt)
            lc = info_nce(pi, pt, args.temp)
            loss = (F.binary_cross_entropy_with_logits(model.classify(pi, pt), y)
                    + args.w_align * lc + args.beta * align_cos(pi, pt))
            if args.presc in ("gamma", "combo"):
                loss = loss + args.gamma * (F.binary_cross_entropy_with_logits(model.head_i(pi), y)
                                            + F.binary_cross_entropy_with_logits(model.head_t(pt), y))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())
        print("  epoch {:>2d}  loss={:.4f}".format(ep + 1, tot / max(1, len(tr))))
        if args.early_stop and ((ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs):
            vs = val_metric()
            if vs > best["val"]:
                best.update(val=vs, ep=ep + 1,
                            state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            print("    [val] ep {} full_micro={:.3f} (best {:.3f}@{})".format(ep + 1, vs, best["val"], best["ep"]))
    if args.early_stop and best["state"] is not None:
        model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})
        print("  => 조기중단 선택: epoch {}".format(best["ep"]))

    ev = eval_missing(model, te, device)
    pi_tr, pt_tr, y_tr = extract_proj(model, tr, device)
    pi_te, pt_te, y_te = extract_proj(model, te, device)
    si = multilabel_probe(pi_tr, y_tr, pi_te, y_te, device=device, seed=args.seed)["f1_macro"]
    stx = multilabel_probe(pt_tr, y_tr, pt_te, y_te, device=device, seed=args.seed)["f1_macro"]

    print("\n--- 결손 성능 (F1 macro / micro / weighted) ---")
    for r in ("full", "image", "text", "both"):
        e = ev[r]
        print("  {:6s}: {:.3f} / {:.3f} / {:.3f}".format(r, e["macro"], e["micro"], e["weighted"]))
    print("--- suff --- z_image {:.3f}  z_text {:.3f}".format(si, stx))

    res = {"presc": args.presc, "config": vars(args), "eval": ev, "suff": {"z_image": si, "z_text": stx}}
    json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    print(">>> saved", args.out)


if __name__ == "__main__":
    main()
