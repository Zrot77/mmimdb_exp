"""v14 검증 — 이미지 essence 천장 스캔 (frozen 지도 선형 probe, 학습 불필요).
캐시 특징의 이미지 단독 상한(각 모달을 라벨로 선형 probe한 test F1)을 인코더별로 비교.
목적: γ의 text결손 천장(~0.469, CLIP FT)이 근본인지, 더 강한 인코더(DINOv2)에 헤드룸이 있는지.

읽는 파일:
  feats/feats_{train,test}.pt      = CLIP  img(512)/txt(512)   (extract_features.py 산출)
  feats_b4/feats_{train,test}.pt   = DINOv2 img(768)/BERT txt(768) (extract_b4.py 산출)

실행: python ceiling_probe.py                 (기본 경로)
      python ceiling_probe.py --clip_dir feats --dino_dir feats_b4
"""
import argparse
import os

import torch

from utils import multilabel_probe


def probe(feat_dir, key, device, seed=0):
    tr = torch.load(os.path.join(feat_dir, "feats_train.pt"), map_location="cpu")
    te = torch.load(os.path.join(feat_dir, "feats_test.pt"), map_location="cpu")
    r = multilabel_probe(tr[key].float(), tr["y"].float(), te[key].float(), te["y"].float(),
                         device=device, seed=seed)
    return r, tr[key].shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_dir", default="feats")
    ap.add_argument("--strong_dir", default="feats_siglip", help="더 강한 인코더 특징 캐시")
    ap.add_argument("--strong_name", default="SigLIP img", help="표시 이름")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def multi(feat_dir, key):
        ms = [probe(feat_dir, key, dev, s)[0] for s in range(args.seeds)]
        mac = [r["f1_macro"] for r in ms]; mic = [r["f1_micro"] for r in ms]
        m = sum(mac) / len(mac); s = (sum((x - m) ** 2 for x in mac) / len(mac)) ** 0.5 if len(mac) > 1 else 0
        mm = sum(mic) / len(mic)
        return m, s, mm

    print("== 이미지 essence 천장 (frozen 지도 선형 probe, test F1, {}seed) ==".format(args.seeds))
    rows = [("CLIP img (B/32)", args.clip_dir, "img"), (args.strong_name, args.strong_dir, "img")]
    for name, d, k in rows:
        try:
            mac, sd, mic = multi(d, k)
            print("  {:18s}: macro {:.3f}±{:.3f}  micro {:.3f}".format(name, mac, sd, mic))
        except Exception as e:
            print("  {:18s}: (로드 실패 {})".format(name, e))

    print("\n== (참고) 텍스트 essence ==")
    for name, d, k in [("CLIP txt (B/32)", args.clip_dir, "txt"), (args.strong_name.replace("img", "txt"), args.strong_dir, "txt")]:
        try:
            mac, sd, mic = multi(d, k)
            print("  {:18s}: macro {:.3f}±{:.3f}  micro {:.3f}".format(name, mac, sd, mic))
        except Exception as e:
            print("  {:18s}: (로드 실패 {})".format(name, e))

    print("\n  해석: γ text결손 천장 ≈ 0.469 (CLIP B/32 FT). ")
    print("        강한 인코더 img macro가 CLIP img(≈0.407 frozen)보다 크게 높으면 → 헤드룸 있음 → Stage B(이식) 진행")
    print("        비슷/낮으면 → 천장이 표현 근본 한계 → 이식으론 못 넘음, 다른 각도 필요")


if __name__ == "__main__":
    main()
