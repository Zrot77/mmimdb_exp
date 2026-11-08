"""추가 실험 v2 — A0 미완 항목: raw suff(학습 전 고정 특징의 선형 probe) 재계산 (학습 없음, GPU 있으면 사용)
  python v2_a0_rawsuff.py [SigLIP특징폴더]   # 서버 ~/Yechan/mmimdb_exp (conda mmimdb), 기본 feats_siglip
[고정 규칙 — 실행 전, 2026-11-08]
- 구현: utils.multilabel_probe (Linear+BCE, epochs 300, lr 1e-2, 기준값 0.5) = ceiling_probe.py와 같은 함수.
- 특징: feats/feats_{train,test}.pt (CLIP img·txt), feats_siglip/feats_{train,test}.pt (SigLIP img). 없으면 '없음'으로 표시.
- probe seed 0·1·2(ceiling_probe 기본과 같음), mean ± sd(ddof=1)와 seed별 값을 출력하고 문서값(CLIP img .407, CLIP txt .488,
  SigLIP img .545)과 비교한다. |차| > 0.002면 '★문서값과 다름'으로 표시(A0.3 기준과 같음). 판정은 하지 않음(감사 목적).
- 주의: probe seed는 분류기 초기값만 바꾸며 MM-IMDb 학습 seed와 짝지을 수 없음 → E2의 L은 이 평균을 쓴다.
"""
import json, os, sys
import numpy as np
import torch
from utils import multilabel_probe

SIG_DIR = sys.argv[1] if len(sys.argv) > 1 else "feats_siglip"   # SigLIP so400m 특징 폴더가 다르면 인자로 지정
ITEMS = (("CLIP img", "feats", "img", 0.407), ("CLIP txt", "feats", "txt", 0.488), ("SigLIP img", SIG_DIR, "img", 0.545))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 90); print("A0 raw suff 재계산 | 장치 %s | probe seed 0·1·2 | ddof=1" % dev); print("=" * 90)
    res = {}
    for name, d, key, doc in ITEMS:
        p_tr, p_te = os.path.join(d, "feats_train.pt"), os.path.join(d, "feats_test.pt")
        if not (os.path.exists(p_tr) and os.path.exists(p_te)):
            print("  %-10s: 특징 파일 없음 (%s)" % (name, d)); continue
        tr = torch.load(p_tr, map_location="cpu"); te = torch.load(p_te, map_location="cpu")
        if key not in tr:
            print("  %-10s: '%s' 키 없음 (있는 키 %s)" % (name, key, list(tr.keys()))); continue
        vals = [multilabel_probe(tr[key].float(), tr["y"].float(), te[key].float(), te["y"].float(), device=dev, seed=s)["f1_macro"]
                for s in (0, 1, 2)]
        m, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
        flag = "" if abs(m - doc) <= 0.002 else "  ★문서값과 다름"
        print("  %-10s: macro-F1 %.4f ± %.4f (seed별 %s) | 문서 %.3f | 차이 %+.4f%s | n_train %d, dim %d" % (
            name, m, sd, ["%.4f" % v for v in vals], doc, m - doc, flag, tr[key].shape[0], tr[key].shape[1]))
        res[name] = {"mean": m, "sd": sd, "seeds": vals, "doc": doc}
    os.makedirs("results/thesis_round/V2", exist_ok=True)
    json.dump(res, open("results/thesis_round/V2/a0_rawsuff.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("결과 저장: results/thesis_round/V2/a0_rawsuff.json"); print("=== ALL DONE (A0 raw suff) ===")


if __name__ == "__main__":
    main()
