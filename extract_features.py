"""raw MM-IMDb → frozen CLIP 특징 캐시 (backbone freeze니 한 번만 뽑음).
출력: <out>/feats_{train,dev,test}.pt = {"img":(N,512),"txt":(N,512),"y":(N,23),"ids":[...]}

raw 구조(archive.org mmimdb.tar.gz 해제): dataset/<id>.json + <id>.jpeg, split.json.
  json 예상 키: "plot"(list[str]), "genres"(list[str]).
  ⚠️ 실제 키/구조는 해제 후 `ls`와 json 하나로 확인해 맞출 것(아래 파서는 방어적으로 작성).

CLIP: openai/clip-vit-base-patch32 (HuggingFace transformers). 텍스트 77토큰 절단(truncation=True).
필요: transformers, pillow. (mmimdb env)

사용:
  python extract_features.py --raw_root <mmimdb> --out feats --limit 40      # 스모크
  python extract_features.py --raw_root <mmimdb> --out feats                 # 전체
"""
import argparse
import glob
import json
import os

import torch

from utils import GENRES, N_GENRE

GENRE_IDX = {g: i for i, g in enumerate(GENRES)}
CLIP_NAME = "openai/clip-vit-base-patch32"


def parse_plot(obj):
    """json에서 플롯 텍스트 하나 추출(방어적). plot이 list면 가장 긴 것."""
    p = obj.get("plot") or obj.get("plot outline") or obj.get("synopsis") or ""
    if isinstance(p, list):
        p = max(p, key=len) if p else ""
    return str(p).strip()


def multihot(genres):
    y = torch.zeros(N_GENRE)
    for g in (genres or []):
        if g in GENRE_IDX:
            y[GENRE_IDX[g]] = 1.0
    return y


def load_split(raw_root, dataset_dir):
    """split.json 있으면 사용({train,dev,test}: id 리스트). 없으면 결정적 8:1:3 근사 split."""
    sp = os.path.join(raw_root, "split.json")
    if os.path.exists(sp):
        s = json.load(open(sp))
        return {k: [str(x) for x in s.get(k, [])] for k in ("train", "dev", "test")}
    ids = sorted(os.path.splitext(os.path.basename(f))[0]
                 for f in glob.glob(os.path.join(dataset_dir, "*.json")) if "split" not in f)
    import random
    random.Random(42).shuffle(ids)
    n = len(ids); n_tr = int(n * 0.60); n_va = int(n * 0.10)
    print("  [split.json 없음 → 결정적 split 생성]")
    return {"train": ids[:n_tr], "dev": ids[n_tr:n_tr + n_va], "test": ids[n_tr + n_va:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True, help="mmimdb 해제 루트 (dataset/, split.json)")
    ap.add_argument("--out", default="feats")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="스모크: split별 앞 N개만")
    ap.add_argument("--max_pixels", type=int, default=25_000_000,
                    help="이 픽셀수(=W*H) 초과 이미지는 스킵(디코딩 느리고 드묾). 0=제한 없음")
    args = ap.parse_args()

    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None      # 초대형 포스터에서 DecompressionBomb 하드에러 방지
    from transformers import CLIPModel, CLIPProcessor

    dataset_dir = os.path.join(args.raw_root, "dataset")
    if not os.path.isdir(dataset_dir):
        dataset_dir = args.raw_root
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("CLIP 로드:", CLIP_NAME)
    model = CLIPModel.from_pretrained(CLIP_NAME).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_NAME)

    split = load_split(args.raw_root, dataset_dir)
    for name in ("train", "dev", "test"):
        ids = split[name]
        if args.limit:
            ids = ids[:args.limit]
        imgs_z, txts_z, ys, kept = [], [], [], []
        buf_img, buf_txt, buf_id, buf_y = [], [], [], []
        big = 0    # 너무 큰 이미지로 스킵된 수

        def flush():
            if not buf_img:
                return
            inp = proc(text=buf_txt, images=buf_img, return_tensors="pt",
                       padding=True, truncation=True, max_length=77)
            with torch.no_grad():
                zi = model.get_image_features(pixel_values=inp["pixel_values"].to(device))
                zt = model.get_text_features(input_ids=inp["input_ids"].to(device),
                                             attention_mask=inp["attention_mask"].to(device))
            imgs_z.append(zi.cpu()); txts_z.append(zt.cpu())
            ys.extend(buf_y); kept.extend(buf_id)
            buf_img.clear(); buf_txt.clear(); buf_id.clear(); buf_y.clear()

        for i, cid in enumerate(ids):
            jp = os.path.join(dataset_dir, cid + ".json")
            ip = os.path.join(dataset_dir, cid + ".jpeg")
            if not (os.path.exists(jp) and os.path.exists(ip)):
                continue
            try:
                obj = json.load(open(jp, encoding="utf-8"))
                img = Image.open(ip)                       # lazy: 헤더만 읽어 크기 확인
                if args.max_pixels and img.width * img.height > args.max_pixels:
                    big += 1
                    continue                               # 지나치게 큰 이미지 제외
                img = img.convert("RGB")
            except Exception:
                continue
            buf_img.append(img); buf_txt.append(parse_plot(obj) or " ")
            buf_id.append(cid); buf_y.append(multihot(obj.get("genres")))
            if len(buf_img) >= args.batch:
                flush()
            if (i + 1) % 1000 == 0:
                print("  {} {}/{}".format(name, i + 1, len(ids)))
        flush()

        out = {"img": torch.cat(imgs_z) if imgs_z else torch.empty(0, 512),
               "txt": torch.cat(txts_z) if txts_z else torch.empty(0, 512),
               "y": torch.stack(ys) if ys else torch.empty(0, N_GENRE),
               "ids": kept}
        path = os.path.join(args.out, "feats_{}.pt".format(name))
        torch.save(out, path)
        print("saved {}: {} samples (너무 큰 이미지 스킵 {}) -> {}".format(name, len(kept), big, path))


if __name__ == "__main__":
    main()
