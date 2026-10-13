"""B2 게이트 집계·판정 — PAD zerofill 5-fold.
G1 규칙(사전등록):
  - 약모달 W = raw suff 낮은 쪽(image_raw vs meta), 전역 평균으로 결정.
  - 강모달 S = 나머지. 대표 결손 = S 결손 케이스(W만 남음):
      W=image → S=meta → meta_missing / W=meta → S=image → image_missing
  - collapse = suff_W − zerofill(S_missing) macro-F1,  fold별 계산 후 평균(짝지음).
  - 평균 collapse ≥ 0.05 → B3/B4 진행. 미만 → 중단·보고.
반복=5 fold(fold당 seed 1 고정), 짝지음 CI: df=4, t.975=2.776, ddof=1.
실행: python agg_b2.py
"""
import glob, json, numpy as np

T975 = {4: 2.776, 3: 4.303}  # df→t.975 (n=5→df4)
GATE = 0.05


def main():
    files = sorted(glob.glob("results/thesis_round/B2/pad_zerofill_f*.json"))
    rows = [json.load(open(f)) for f in files]
    assert len(rows) == 5, "fold 5개 json 필요, 현재 %d개: %s" % (len(rows), files)
    folds = [r["fold"] for r in rows]
    assert sorted(folds) == [0, 1, 2, 3, 4], "fold 0~4 필요, 현재 %s" % folds

    si = np.array([r["suff"]["image_raw"] for r in rows])  # 이미지 raw suff
    sm = np.array([r["suff"]["meta"] for r in rows])        # 메타 suff
    W = "image" if si.mean() < sm.mean() else "meta"
    S = "meta" if W == "image" else "image"
    S_missing = "meta_missing" if W == "image" else "image_missing"
    suffW = si if W == "image" else sm

    fm = np.array([r["eval"][S_missing]["f1_macro"] for r in rows])  # S결손 시 성능(=W만)
    collapse = suffW - fm

    n = 5; df = 4; t = T975[df]
    mean = float(collapse.mean()); sd = float(collapse.std(ddof=1))
    se = sd / np.sqrt(n); ci = t * se
    lo, hi = mean - ci, mean + ci
    passed = mean >= GATE

    print("=" * 68)
    print("B2 게이트 — PAD zerofill 5-fold")
    print("=" * 68)
    print("raw suff 평균:  image %.3f | meta %.3f" % (si.mean(), sm.mean()))
    print("→ 약모달 W = %-5s (raw suff 낮음) | 강모달 S = %s" % (W, S))
    print("→ 대표 결손 케이스 = %s (S 결손 → W만 남음)" % S_missing)
    print("-" * 68)
    print("fold |  suff_W | %s f1 | collapse" % S_missing)
    for i, r in enumerate(rows):
        print("  %d  |  %.3f  |    %.3f     |  %+.3f" % (r["fold"], suffW[i], fm[i], collapse[i]))
    print("-" * 68)
    print("collapse 평균 = %.3f  (SD %.3f, 95%%CI [%+.3f, %+.3f], n=5 df=4)" % (mean, sd, lo, hi))
    print("게이트 기준 = suff_W − zerofill(S_missing) ≥ %.2f" % GATE)
    print("판정: %s" % ("★통과 → B3/B4 진행" if passed else "★미달 → 중단·보고(사전규칙)"))
    print("=" * 68)

    # 참고: 양 결손 케이스 요약(판정엔 미사용)
    full = np.array([r["eval"]["full"]["f1_macro"] for r in rows])
    imm = np.array([r["eval"]["image_missing"]["f1_macro"] for r in rows])
    mmm = np.array([r["eval"]["meta_missing"]["f1_macro"] for r in rows])
    print("[참고] full %.3f | image_missing %.3f | meta_missing %.3f (fold평균 macro-F1)" % (
        full.mean(), imm.mean(), mmm.mean()))
    print("[참고] suff after: image %.3f | meta %.3f" % (
        np.mean([r["suff"]["image_after"] for r in rows]),
        np.mean([r["suff"]["meta_after"] for r in rows])))


if __name__ == "__main__":
    main()
