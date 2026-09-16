"""v14 Stage B — SigLIP(so400m) 고정 특징 추출 (더 강한 의미정렬 백본, 천장 돌파 후보).
CLIP ViT-B/32(가장 약한 CLIP) 대비 SigLIP so400m는 이미지-텍스트 정렬이 훨씬 강해
포스터→장르 이미지 essence 천장을 올릴 후보. 동일 split/라벨(기존 feats/ ids 재사용)로 뽑아
ceiling_probe.py로 CLIP img와 직접 비교.

출력: <out>/feats_{train,dev,test}.pt = {"img":(N,1152),"txt":(N,1152),"y":(N,23),"ids":[...]}
필요: transformers>=4.37 (SiglipModel). 이미지 384px.
주의: SigLIP 텍스트는 max_length=64(짧음) → 긴 플롯은 절단됨(이미지 천장 검증엔 무관, 정렬 파이프라인 땐 고려).

실행:
  python extract_siglip.py --raw_root ~/Yechan/mmimdb_data/mmimdb --feat_dir feats --out feats_siglip --limit 40
  python extract_siglip.py --raw_root ~/Yechan/mmimdb_data/mmimdb --feat_dir feats --out feats_siglip
"""
import argparse
import json
import os

import torch

from utils import N_GENRE

SIGLIP = "google/siglip-so400m-patch14-384"


def parse_plot(obj):
    p = obj.get("plot") or obj.get("plot outline") or obj.get("synopsis") or ""
    if isinstance(p, list):
        p = max(p, key=len) if p else ""
    return str(p).strip() or " "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--feat_dir", default="feats", help="ids/labels/split 재사용용(기존 CLIP feats)")
    ap.add_argument("--out", default="feats_siglip")
    ap.add_argument("--model", default=SIGLIP)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_pixels", type=int, default=25_000_000)
    ap.add_argument("--max_text_len", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_dir = os.path.join(args.raw_root, "dataset")

    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    from transformers import AutoModel, AutoProcessor

    print("SigLIP 로드:", args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    proc = AutoProcessor.from_pretrained(args.model)

    for split in ("train", "dev", "test"):
        d = torch.load(os.path.join(args.feat_dir, "feats_{}.pt".format(split)), map_location="cpu")
        ids = d["ids"][:args.limit] if args.limit else d["ids"]
        ylab = d["y"][:len(ids)].float()
        imgs_z, txts_z, ys, kept = [], [], [], []
        buf_img, buf_txt, buf_y, buf_id = [], [], [], []
        big = 0

        @torch.no_grad()
        def flush():
            if not buf_img:
                return
            inp = proc(text=buf_txt, images=buf_img, return_tensors="pt",
                       padding="max_length", truncation=True, max_length=args.max_text_len)
            zi = model.get_image_features(pixel_values=inp["pixel_values"].to(device))
            zt = model.get_text_features(input_ids=inp["input_ids"].to(device))
            imgs_z.append(zi.cpu()); txts_z.append(zt.cpu())
            ys.extend(buf_y); kept.extend(buf_id)
            buf_img.clear(); buf_txt.clear(); buf_y.clear(); buf_id.clear()

        for i, cid in enumerate(ids):
            jp = os.path.join(dataset_dir, cid + ".json")
            ip = os.path.join(dataset_dir, cid + ".jpeg")
            if not (os.path.exists(jp) and os.path.exists(ip)):
                continue
            try:
                img = Image.open(ip)
                if args.max_pixels and img.width * img.height > args.max_pixels:
                    big += 1; continue
                img = img.convert("RGB")
                plot = parse_plot(json.load(open(jp, encoding="utf-8")))
            except Exception:
                continue
            buf_img.append(img); buf_txt.append(plot); buf_y.append(ylab[i]); buf_id.append(cid)
            if len(buf_img) >= args.batch:
                flush()
            if (i + 1) % 1000 == 0:
                print("  {} {}/{}".format(split, i + 1, len(ids)))
        flush()

        out = {"img": torch.cat(imgs_z) if imgs_z else torch.empty(0, 1152),
               "txt": torch.cat(txts_z) if txts_z else torch.empty(0, 1152),
               "y": torch.stack(ys) if ys else torch.empty(0, N_GENRE), "ids": kept}
        path = os.path.join(args.out, "feats_{}.pt".format(split))
        torch.save(out, path)
        print("saved {}: {} samples (너무 큰 이미지 스킵 {}) -> {}".format(split, len(kept), big, path))


if __name__ == "__main__":
    main()
