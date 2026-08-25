"""2주차 A1 — 약한 모달 확정.
raw CLIP 특징(proj·정렬 학습 이전, 고정값) 자체를 다중라벨 probe → 각 모달 내재 suff.
= freeze 세팅에서 '그 모달 표현이 담은 라벨 정보량'의 깨끗한 척도(우리 학습과 무관).
낮은 쪽이 약한 모달. (융합 의존 비대칭은 Week1 결손 하락에서 이미 확보.)

실행:
  python diagnose_a1.py --mode real --feat_dir feats --seeds 0,1,2
  python diagnose_a1.py --mode synthetic --seeds 0,1,2      # 로직 스모크
"""
import argparse
import json

import numpy as np
import torch

from data_features import load_feats, synth_feats
from utils import multilabel_probe


def get(mode, feat_dir, split, seed):
    if mode == "synthetic":
        n = {"train": 1500, "test": 600}[split]
        return synth_feats(n, seed={"train": 0, "test": 2}[split])
    return load_feats(feat_dir, split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthetic", "real"], default="real")
    ap.add_argument("--feat_dir", default="feats")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="results/a1_weakmod.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in args.seeds.split(",")]

    img_tr, txt_tr, y_tr = get(args.mode, args.feat_dir, "train", 0)
    img_te, txt_te, y_te = get(args.mode, args.feat_dir, "test", 0)
    print("=== A1 약한 모달 확정 | mode={} | train={} test={} | dev={} ===".format(
        args.mode, len(y_tr), len(y_te), device))

    rows = {"image": [], "text": []}
    for s in seeds:
        ri = multilabel_probe(img_tr, y_tr, img_te, y_te, device=device, seed=s)
        rt = multilabel_probe(txt_tr, y_tr, txt_te, y_te, device=device, seed=s)
        rows["image"].append(ri["f1_macro"]); rows["text"].append(rt["f1_macro"])
        print("  seed {}: suff_raw(image)={:.3f}  suff_raw(text)={:.3f}".format(s, ri["f1_macro"], rt["f1_macro"]))

    mi, si = np.mean(rows["image"]), np.std(rows["image"])
    mt, st = np.mean(rows["text"]), np.std(rows["text"])
    weak = "image" if mi < mt else "text"
    print("\n--- raw CLIP 특징 suff (F1-macro, mean±std, {}seed) ---".format(len(seeds)))
    print("  image: {:.3f}±{:.3f}".format(mi, si))
    print("  text : {:.3f}±{:.3f}".format(mt, st))
    print("=> 약한 모달(내재 suff 낮은 쪽): {}  (격차 {:.3f})".format(weak, abs(mi - mt)))
    print("   (참고: Week1 결손 하락 — 텍스트 결손이 치명적/이미지 결손 무영향 → 융합도 이미지 과소사용)")

    res = {"suff_raw": {"image": {"mean": float(mi), "std": float(si), "seeds": rows["image"]},
                        "text": {"mean": float(mt), "std": float(st), "seeds": rows["text"]}},
           "weak_modality": weak, "gap": float(abs(mi - mt))}
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    print(">>> saved", args.out)


if __name__ == "__main__":
    main()
