"""v8 블록3 — 텍스트 77토큰 절단 완화 후 A1 재확인 (의심 #4 해소).
플롯을 문장 단위로 쪼개 각각 CLIP 텍스트 인코딩 → 평균(플롯 전체 정보 반영).
이미지 특징은 기존 feats 재사용. 약한 모달=이미지 결론이 전처리에 견고한지 확인.

실행:
  python block3_text.py --raw_root ~/Yechan/mmimdb_data/mmimdb --feat_dir feats
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import torch

from utils import multilabel_probe

CLIP_NAME = "openai/clip-vit-base-patch32"


def parse_plot(obj):
    p = obj.get("plot") or obj.get("plot outline") or obj.get("synopsis") or ""
    if isinstance(p, list):
        p = max(p, key=len) if p else ""
    return str(p).strip()


def split_sentences(text, max_sents=12):
    """문장 분할(간단). 너무 많으면 앞 max_sents개."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return sents[:max_sents] if sents else [" "]


@torch.no_grad()
def text_full_features(ids, dataset_dir, model, proc, device, batch=256):
    """각 클립: 플롯 문장들을 인코딩→평균 = full-plot 텍스트 특징 (77절단 완화)."""
    out = []
    for cid in ids:
        obj = json.load(open(os.path.join(dataset_dir, cid + ".json"), encoding="utf-8"))
        sents = split_sentences(parse_plot(obj))
        embs = []
        for k in range(0, len(sents), batch):
            enc = proc(text=sents[k:k + batch], return_tensors="pt", padding=True,
                       truncation=True, max_length=77)
            e = model.get_text_features(input_ids=enc["input_ids"].to(device),
                                        attention_mask=enc["attention_mask"].to(device))
            embs.append(e.cpu())
        out.append(torch.cat(embs).mean(0))
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--feat_dir", default="feats")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/b3_textfull.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_dir = os.path.join(args.raw_root, "dataset")

    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    model = CLIPModel.from_pretrained(CLIP_NAME).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_NAME)

    def load(split):
        d = torch.load(os.path.join(args.feat_dir, "feats_{}.pt".format(split)), map_location="cpu")
        ids = d["ids"][:args.limit] if args.limit else d["ids"]
        img = d["img"][:len(ids)].float()
        y = d["y"][:len(ids)].float()
        txt_full = text_full_features(ids, dataset_dir, model, proc, device)
        return img, txt_full, y

    print("=== 블록3: 텍스트 절단 완화(문장평균) 후 A1 재측정 ===")
    img_tr, txtf_tr, y_tr = load("train")
    img_te, txtf_te, y_te = load("test")

    s_img = multilabel_probe(img_tr, y_tr, img_te, y_te, device=device, seed=0)["f1_macro"]
    s_txtf = multilabel_probe(txtf_tr, y_tr, txtf_te, y_te, device=device, seed=0)["f1_macro"]
    print("  raw suff — image {:.3f} / text_full(문장평균) {:.3f}".format(s_img, s_txtf))
    print("  (참고: 절단판 원본 — image 0.407 / text 0.488)")
    weak = "image" if s_img < s_txtf else "text"
    print("  => 약한 모달: {}  (전처리 바꿔도 이미지 약함 유지되면 A1 견고)".format(weak))

    res = {"suff_raw": {"image": s_img, "text_full": s_txtf}, "weak": weak}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    print(">>> saved", args.out)


if __name__ == "__main__":
    main()
