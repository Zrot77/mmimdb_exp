"""v14 Stage B — 이미지=SigLIP + 텍스트=CLIP 결합 특징 (id 정렬).
"이미지 백본만 교체(강해짐), 텍스트는 강한 CLIP 유지"라는 통제 비교용.
출력: <out>/feats_{split}.pt = {"img":(N,1152 SigLIP),"txt":(N,512 CLIP),"y","ids"}

실행: python merge_siglip.py --siglip_dir feats_siglip --clip_dir feats --out feats_siglipmix
"""
import argparse
import os

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--siglip_dir", default="feats_siglip")
ap.add_argument("--clip_dir", default="feats")
ap.add_argument("--out", default="feats_siglipmix")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

for split in ("train", "dev", "test"):
    s = torch.load(os.path.join(args.siglip_dir, "feats_{}.pt".format(split)), map_location="cpu")
    c = torch.load(os.path.join(args.clip_dir, "feats_{}.pt".format(split)), map_location="cpu")
    cidx = {cid: i for i, cid in enumerate(c["ids"])}
    img, txt, y, ids = [], [], [], []
    for j, cid in enumerate(s["ids"]):
        if cid in cidx:
            i = cidx[cid]
            img.append(s["img"][j]); txt.append(c["txt"][i]); y.append(c["y"][i]); ids.append(cid)
    out = {"img": torch.stack(img).float(), "txt": torch.stack(txt).float(),
           "y": torch.stack(y).float(), "ids": ids}
    torch.save(out, os.path.join(args.out, "feats_{}.pt".format(split)))
    print("saved {}: {} (img {} txt {})".format(split, len(ids), tuple(out["img"].shape), tuple(out["txt"].shape)))
