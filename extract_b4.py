"""v8 블록4 — 단독 사전학습 인코더(서로 정렬 안 됨)의 고정 특징 추출.
이미지 = DINOv2 ViT-B/14 (self-sup, 이미지 전용), 텍스트 = BERT-base (텍스트 전용).
CLIP과 달리 cross-modal 정렬이 안 돼 있음 → 이 위에서 정렬(proj)만 학습하면
'CLIP의 사전 정렬이 문제를 지웠는가'를 동일 프로토콜(diagnose_a2)로 판별.

출력: <out>/feats_{train,dev,test}.pt = {"img":(N,768),"txt":(N,768),"y":(N,23),"ids":[...]}
그 뒤:  python diagnose_a1.py --feat_dir <out>   /   python diagnose_a2.py --feat_dir <out>

필요: timm(DINOv2), transformers(BERT). pip install timm
실행:  python extract_b4.py --raw_root ~/Yechan/mmimdb_data/mmimdb --feat_dir feats --out feats_b4 --limit 40
       python extract_b4.py --raw_root ~/Yechan/mmimdb_data/mmimdb --feat_dir feats --out feats_b4
"""
import argparse
import json
import os

import torch

from utils import N_GENRE

DINO = "vit_base_patch14_dinov2.lvd142m"
BERT = "bert-base-uncased"


def parse_plot(obj):
    p = obj.get("plot") or obj.get("plot outline") or obj.get("synopsis") or ""
    if isinstance(p, list):
        p = max(p, key=len) if p else ""
    return str(p).strip() or " "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--feat_dir", default="feats", help="ids/labels/split 재사용용(기존 CLIP feats)")
    ap.add_argument("--out", default="feats_b4")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_pixels", type=int, default=25_000_000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_dir = os.path.join(args.raw_root, "dataset")

    import timm
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    from transformers import AutoTokenizer, AutoModel

    # 이미지: DINOv2 (timm)
    dino = timm.create_model(DINO, pretrained=True, num_classes=0).to(device).eval()
    cfg = timm.data.resolve_data_config({}, model=dino)
    tf = timm.data.create_transform(**cfg)
    # 텍스트: BERT (mean-pool)
    tok = AutoTokenizer.from_pretrained(BERT)
    bert = AutoModel.from_pretrained(BERT).to(device).eval()

    @torch.no_grad()
    def enc_text(texts):
        b = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        out = bert(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
        m = b["attention_mask"].to(device).unsqueeze(-1).float()
        return (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1)   # (B,768)

    for split in ("train", "dev", "test"):
        d = torch.load(os.path.join(args.feat_dir, "feats_{}.pt".format(split)), map_location="cpu")
        ids = d["ids"][:args.limit] if args.limit else d["ids"]
        ylab = d["y"][:len(ids)].float()
        imgs, txts, ys, kept = [], [], [], []
        bi, bt, byi = [], [], []

        @torch.no_grad()
        def flush():
            if not bi:
                return
            px = torch.stack(bi).to(device)
            imgs.append(dino(px).cpu())
            txts.append(enc_text(bt).cpu())
            ys.append(torch.stack(byi))
            bi.clear(); bt.clear(); byi.clear()

        for i, cid in enumerate(ids):
            jp = os.path.join(dataset_dir, cid + ".json")
            ip = os.path.join(dataset_dir, cid + ".jpeg")
            if not (os.path.exists(jp) and os.path.exists(ip)):
                continue
            try:
                img = Image.open(ip)
                if args.max_pixels and img.width * img.height > args.max_pixels:
                    continue
                img = img.convert("RGB")
                plot = parse_plot(json.load(open(jp, encoding="utf-8")))
            except Exception:
                continue
            bi.append(tf(img)); bt.append(plot); byi.append(ylab[i]); kept.append(cid)
            if len(bi) >= args.batch:
                flush()
            if (i + 1) % 1000 == 0:
                print("  {} {}/{}".format(split, i + 1, len(ids)))
        flush()
        out = {"img": torch.cat(imgs) if imgs else torch.empty(0, 768),
               "txt": torch.cat(txts) if txts else torch.empty(0, 768),
               "y": torch.cat(ys) if ys else torch.empty(0, N_GENRE), "ids": kept}
        path = os.path.join(args.out, "feats_{}.pt".format(split))
        torch.save(out, path)
        print("saved {}: {} samples -> {}".format(split, len(kept), path))


if __name__ == "__main__":
    main()
