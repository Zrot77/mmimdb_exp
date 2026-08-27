"""2주차 (A) — 학습되는 이미지 인코더로 nuisance-shortcut 진단.
CLIP 이미지 타워 (부분)파인튜닝 + 고정 텍스트 앵커. 목적함수별 suff(z_image) 측정.

목적함수(--objective):
  infonce    : 학습 이미지 ↔ 고정 텍스트 정렬 (라벨 없음)   → suff [정렬·학습]
  supervised : 이미지-only BCE (라벨)                      → suff [천장·학습]
  supcon     : 같은 장르 positive(FN 제거) + 텍스트 앵커     → suff
(+ --img_aug noise 로 nuisance 억제 처방; A3)

실행:
  python train_ft.py --mode synthetic --backbone stub --objective infonce --smoke   # 로직 스모크
  python train_ft.py --mode real --feat_dir feats --raw_root ~/Yechan/mmimdb_data/mmimdb \
      --backbone clip --objective infonce --epochs 5
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
from utils import info_nce, supcon_multilabel, multilabel_probe, N_GENRE

CLIP_NAME = "openai/clip-vit-base-patch32"


class StubVision(nn.Module):
    """스모크용 소형 이미지 인코더(가변 HxW → 512)."""
    def __init__(self, dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, dim))

    def forward(self, px):
        return self.net(px)


class FTModel(nn.Module):
    def __init__(self, backbone="clip", unfreeze_last=3, n_cls=N_GENRE, clip_name=CLIP_NAME):
        super().__init__()
        self.backbone = backbone
        if backbone == "clip":
            from transformers import CLIPModel
            self.clip = CLIPModel.from_pretrained(clip_name)
            for p in self.clip.parameters():
                p.requires_grad_(False)
            layers = self.clip.vision_model.encoder.layers
            for blk in layers[-unfreeze_last:]:               # 마지막 N 블록만 학습
                for p in blk.parameters():
                    p.requires_grad_(True)
            for p in self.clip.visual_projection.parameters():
                p.requires_grad_(True)
            for p in self.clip.vision_model.post_layernorm.parameters():
                p.requires_grad_(True)
        else:
            self.vision = StubVision(512)
        self.cls = nn.Linear(512, n_cls)

    def encode_image(self, px):
        if self.backbone == "clip":
            return self.clip.get_image_features(pixel_values=px)
        return self.vision(px)


def make_loader(args, split, shuffle):
    if args.mode == "synthetic":
        ds = SyntheticFT(n={"train": 400, "dev": 200, "test": 200}[split],
                         seed={"train": 0, "dev": 1, "test": 2}[split])
    else:
        from transformers import CLIPProcessor
        proc = CLIPProcessor.from_pretrained(CLIP_NAME)
        ds = ImageTextFT(args.feat_dir, args.raw_root, split, proc)
    return DataLoader(ds, batch_size=args.batch, shuffle=shuffle, num_workers=args.workers)


@torch.no_grad()
def extract_z(model, loader, device):
    model.eval()
    zs, ys = [], []
    for px, txt, y in loader:
        zs.append(model.encode_image(px.to(device)).cpu()); ys.append(y)
    return torch.cat(zs), torch.cat(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--feat_dir", default="feats")
    ap.add_argument("--raw_root", default=None)
    ap.add_argument("--backbone", choices=["clip", "stub"], default="clip")
    ap.add_argument("--objective", choices=["infonce", "supervised", "supcon"], default="infonce")
    ap.add_argument("--unfreeze_last", type=int, default=3)
    ap.add_argument("--img_aug", choices=["none", "noise"], default="none")
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/ft.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 1
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr = make_loader(args, "train", shuffle=True)
    te = make_loader(args, "test", shuffle=False)
    model = FTModel(args.backbone, args.unfreeze_last).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    tag = "{}{}".format(args.objective, "+noise" if args.img_aug == "noise" else "")
    print("=== FT | {} | backbone={} mode={} | 학습파라미터 {} | dev={} ===".format(
        tag, args.backbone, args.mode, sum(p.numel() for p in params), device))

    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for px, txt, y in tr:
            px, txt, y = px.to(device), txt.to(device), y.to(device)
            if args.img_aug == "noise":
                px = px + args.sigma * torch.randn_like(px)
            zi = model.encode_image(px)
            if args.objective == "supervised":
                loss = F.binary_cross_entropy_with_logits(model.cls(zi), y)
            elif args.objective == "supcon":
                loss = supcon_multilabel(zi, txt, args.temp, y)
            else:
                loss = info_nce(zi, txt, args.temp)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())
        print("  epoch {:>2d}  loss={:.4f}".format(ep + 1, tot / max(1, len(tr))))

    # suff(z_image): 학습된 이미지 인코더를 freeze 후 probe.
    # probe는 train에 fit, train/val/test에 각각 eval → train↔val 격차로 과적합 점검(v7 §4).
    va = make_loader(args, "dev", shuffle=False)
    zi_tr, y_tr = extract_z(model, tr, device)
    zi_va, y_va = extract_z(model, va, device)
    zi_te, y_te = extract_z(model, te, device)
    s_tr = multilabel_probe(zi_tr, y_tr, zi_tr, y_tr, device=device, seed=args.seed)["f1_macro"]
    s_va = multilabel_probe(zi_tr, y_tr, zi_va, y_va, device=device, seed=args.seed)["f1_macro"]
    s_te = multilabel_probe(zi_tr, y_tr, zi_te, y_te, device=device, seed=args.seed)["f1_macro"]
    print("  suff(z_image) [{}] train {:.3f} / val {:.3f} / test {:.3f}  (과적합 격차 train-val {:+.3f})".format(
        tag, s_tr, s_va, s_te, s_tr - s_va))

    res = {"tag": tag, "config": vars(args),
           "suff_z_image": {"train": s_tr, "val": s_va, "test": s_te, "overfit_gap": s_tr - s_va}}
    json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    print(">>> saved", args.out)


if __name__ == "__main__":
    main()
