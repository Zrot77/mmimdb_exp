"""v8 블록1 — 대칭 FT (이미지·텍스트 양쪽 타워 함께 학습, 고정 앵커 없음).
대칭 경쟁 복원 후 suff(z_image)·suff(z_text) 둘 다 측정. 의심 #3(텍스트 고정 아티팩트) 해소.
판정: 대칭 학습에서 약한 모달(이미지) 누수가 부분FT(0.420) 때보다 커지면 → 고정이 억눌렀던 것.

실행:
  python train_sym.py --mode synthetic --backbone stub --smoke
  python train_sym.py --mode real --backbone clip --raw_root ~/Yechan/mmimdb_data/mmimdb \
      --feat_dir feats --epochs 8 --early_stop --wd 1e-4 --seed 1
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data_ft import ImageTextTokens, SyntheticSym
from train_ft import StubVision
from utils import info_nce, multilabel_probe, N_GENRE

CLIP_NAME = "openai/clip-vit-base-patch32"


class StubText(nn.Module):
    def __init__(self, vocab=1000, dim=512):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)

    def forward(self, ids, mask):
        e = self.emb(ids)
        m = mask.unsqueeze(-1).float()
        return (e * m).sum(1) / m.sum(1).clamp(min=1)


class SymModel(nn.Module):
    def __init__(self, backbone="clip", unfreeze_last=3, clip_name=CLIP_NAME):
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
            for blk in self.clip.text_model.encoder.layers[-unfreeze_last:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            for mod in (self.clip.visual_projection, self.clip.text_projection,
                        self.clip.vision_model.post_layernorm, self.clip.text_model.final_layer_norm):
                for p in mod.parameters():
                    p.requires_grad_(True)
        else:
            self.vision = StubVision(512)
            self.text = StubText(1000, 512)

    def encode_image(self, px):
        return self.clip.get_image_features(pixel_values=px) if self.backbone == "clip" else self.vision(px)

    def encode_text(self, ids, mask):
        return (self.clip.get_text_features(input_ids=ids, attention_mask=mask)
                if self.backbone == "clip" else self.text(ids, mask))


def make_loader(args, split, shuffle):
    if args.mode == "synthetic":
        ds = SyntheticSym(n={"train": 400, "dev": 200, "test": 200}[split],
                          seed={"train": 0, "dev": 1, "test": 2}[split])
    else:
        from transformers import CLIPProcessor
        ds = ImageTextTokens(args.feat_dir, args.raw_root, split, CLIPProcessor.from_pretrained(CLIP_NAME))
    return DataLoader(ds, batch_size=args.batch, shuffle=shuffle, num_workers=args.workers)


@torch.no_grad()
def extract_both(model, loader, device):
    model.eval()
    zi, zt, ys = [], [], []
    for px, ids, mask, y in loader:
        zi.append(model.encode_image(px.to(device)).cpu())
        zt.append(model.encode_text(ids.to(device), mask.to(device)).cpu())
        ys.append(y)
    return torch.cat(zi), torch.cat(zt), torch.cat(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--feat_dir", default="feats")
    ap.add_argument("--raw_root", default=None)
    ap.add_argument("--backbone", choices=["clip", "stub"], default="clip")
    ap.add_argument("--unfreeze_last", type=int, default=3)
    ap.add_argument("--feat_aug", choices=["none", "noise"], default="none")
    ap.add_argument("--feat_sigma", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--early_stop", action="store_true")
    ap.add_argument("--eval_every", type=int, default=4)
    ap.add_argument("--probe_subset", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/sym.json")
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
    model = SymModel(args.backbone, args.unfreeze_last).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    print("=== SYM(대칭FT) | backbone={} mode={} | 학습파라미터 {} | wd={} es={} | dev={} ===".format(
        args.backbone, args.mode, sum(p.numel() for p in params), args.wd, args.early_stop, device))

    n_sub = min(args.probe_subset, len(tr.dataset))
    sub = DataLoader(Subset(tr.dataset, list(range(n_sub))), batch_size=args.batch, shuffle=False,
                     num_workers=args.workers)

    def val_img_suff():
        zi, _, ys = extract_both(model, sub, device)
        zv, _, yv = extract_both(model, va, device)
        return multilabel_probe(zi, ys, zv, yv, device=device, seed=args.seed)["f1_macro"]

    best = {"val": -1.0, "state": None, "ep": args.epochs}
    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for px, ids, mask, y in tr:
            px, ids, mask = px.to(device), ids.to(device), mask.to(device)
            zi = model.encode_image(px)
            zt = model.encode_text(ids, mask)
            if args.feat_aug == "noise":
                zi = zi + args.feat_sigma * torch.randn_like(zi)
            loss = info_nce(zi, zt, args.temp)          # 대칭: 양쪽 다 학습
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())
        print("  epoch {:>2d}  loss={:.4f}".format(ep + 1, tot / max(1, len(tr))))
        if args.early_stop and ((ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs):
            vs = val_img_suff()
            if vs > best["val"]:
                best.update(val=vs, ep=ep + 1,
                            state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            print("    [val] ep {} img_suff={:.3f} (best {:.3f}@{})".format(ep + 1, vs, best["val"], best["ep"]))
    if args.early_stop and best["state"] is not None:
        model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})
        print("  => 조기중단 선택: epoch {}".format(best["ep"]))

    zi_tr, zt_tr, y_tr = extract_both(model, tr, device)
    zi_te, zt_te, y_te = extract_both(model, te, device)
    si = multilabel_probe(zi_tr, y_tr, zi_te, y_te, device=device, seed=args.seed)["f1_macro"]
    stx = multilabel_probe(zt_tr, y_tr, zt_te, y_te, device=device, seed=args.seed)["f1_macro"]
    print("  suff(z_image)={:.3f}  suff(z_text)={:.3f}  (대칭 FT)".format(si, stx))
    res = {"config": vars(args), "suff": {"z_image": si, "z_text": stx},
           "early_stop_best_epoch": best["ep"] if args.early_stop else None}
    json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    print(">>> saved", args.out)


if __name__ == "__main__":
    main()
