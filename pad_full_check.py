"""PAD-UFES-20 결손 없는(full) 조건 성능 — 저장된 로짓만 사용(학습·추론 없음), 집계 숫자만 출력
  python pad_full_check.py        # 서버 mmimdb_exp 폴더 (conda mmimdb)
  - results/thesis_round/V2/dump, dumpH 의 <tag>_f<fold>.npz 에서 test_full 로짓이 있는 모델만 계산
  - 지표: argmax macro-F1(labels 0~5, tp=0이면 0) / macro OvR AUROC(softmax, test에 양·음성이 모두 있는 클래스)
  - 집계: 폴드마다 seed 평균 → 폴드 5개 평균과 95% CI(t=2.776)
  - 참고로 결손 조건(image_missing, meta_missing_proj0)의 zf 값도 같은 방식으로 출력(표 4.2 값과 대조용)
"""
import glob, os, re, sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
K, T5 = 6, 2.776
DIRS = ["results/thesis_round/V2/dump", "results/thesis_round/V2/dumpH"]


def macro_f1(y, pred):
    f = []
    for k in range(K):
        tp = np.sum((pred == k) & (y == k)); fp = np.sum((pred == k) & (y != k)); fn = np.sum((pred != k) & (y == k))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def macro_auc(y, L):
    from sklearn.metrics import roc_auc_score
    P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    a = [roc_auc_score(y == k, P[:, k]) for k in range(K) if 0 < np.sum(y == k) < len(y)]
    return float(np.mean(a))


def ci(v):
    v = np.asarray(v, float); m = v.mean(); h = T5 * v.std(ddof=1) / np.sqrt(len(v))
    return "%.3f [%.3f, %.3f]" % (m, m - h, m + h)


for D in DIRS:
    fs = sorted(glob.glob(os.path.join(D, "*_f0.npz")))
    if not fs:
        print("없음:", D); continue
    tags = sorted({re.sub(r"_f0\.npz$", "", os.path.basename(f)) for f in fs})
    print("=" * 90); print(D, "| 모델 태그:", tags); print("=" * 90)
    fam = {}
    for t in tags:
        fam.setdefault(re.sub(r"_s\d+$", "", t), []).append(t)
    for name, ts in fam.items():
        for key in ("full", "image_missing", "meta_missing_proj0"):
            F, A, ok = [], [], True
            for f in range(5):
                fv, av = [], []
                for t in ts:
                    p = os.path.join(D, "%s_f%d.npz" % (t, f))
                    if not os.path.exists(p):
                        ok = False; break
                    d = np.load(p)
                    if "test_" + key not in d.files:
                        ok = False; break
                    L, y = d["test_" + key], d["test_y"]
                    fv.append(macro_f1(y, L.argmax(1))); av.append(macro_auc(y, L))
                if not ok:
                    break
                F.append(np.mean(fv)); A.append(np.mean(av))
            if ok:
                print("  %-12s %-20s seed %d개 | macro-F1 %s | macro AUROC %s" % (name, key, len(ts), ci(F), ci(A)))
print("\n=== ALL DONE (pad_full_check) ===")
