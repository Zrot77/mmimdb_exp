"""PAD 재보정 상한 진단 (진단 전용, 학습 없음, CPU). 입력: results/thesis_round/V2/dump/*.npz
[사전 고정 규칙 — 2026-11-04, 결과 열람 전]
- 대상: zf·γ8의 결손 케이스 로짓(image_missing / meta_missing_input0), seed 0·1·2 × fold 0~4. 천장 R: only_meta / only_image.
- (a) 상한-CE: 클래스별 편향 6개를 test로 교차엔트로피 최소화
  (b) 상한-F1: 클래스별 편향 6개를 test로 macro-F1 직접 최대화(좌표 상승, 격자 −6~+6, 0.1 간격, 3회 반복) — 편향 재보정의 최선 상한
  (c) 참고: 편향 6개를 validation으로 macro-F1 최대화 → test 적용 (현실적으로 쓸 수 있는 버전; 목적함수 문제 대 데이터 크기 문제 구분)
- 판정: (b) 회복 비율 = (b − zf)/(R − zf) ≥ 0.5 → "편향 재보정으로 메울 수 있는 붕괴", 미만 → "편향 재보정으로는 메울 수 없는 붕괴".
  함께 보고: γ8 − (b), (c) − zf. paired 95%CI (fold별 seed 평균, n=5, df=4).
- (a)(b)는 test에 맞춘 낙관적 값이므로 진단 전용이며 선택·판정(E9)에 쓰지 않는다.
  python pad_recal_ub.py
"""
import os
import numpy as np
from scipy.optimize import minimize

DUMP = "results/thesis_round/V2/dump"; K = 6; T4 = 2.776
GRID = np.round(np.linspace(-6, 6, 121), 2)


def macro_f1(y, pred):
    f = []
    for k in range(K):
        tp = np.sum((pred == k) & (y == k)); fp = np.sum((pred == k) & (y != k)); fn = np.sum((pred != k) & (y == k))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def F1(y, L, b=None): return macro_f1(y, (L if b is None else L + b).argmax(1))


def fit_ce(L, y):
    def f(b):
        z = L + b; z = z - z.max(1, keepdims=True)
        return float(-(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean())
    return minimize(f, np.zeros(K), method="L-BFGS-B").x


def fit_f1(L, y):
    b = np.zeros(K); best = F1(y, L, b)
    for _ in range(3):
        for k in range(K):
            cur = b[k]
            for v in GRID:
                b[k] = v; s = F1(y, L, b)
                if s > best + 1e-12 or (abs(s - best) <= 1e-12 and abs(v) < abs(cur)):
                    best, cur = s, v
            b[k] = cur
    return b


def ci(d):
    d = np.asarray(d, float); m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
    return float(m), float(m - T4 * se), float(m + T4 * se)


def verdict(lo, hi): return "개선" if lo > 0 else ("악화" if hi < 0 else "구분 불가")


def L_(tag, f):
    p = os.path.join(DUMP, "%s_f%d.npz" % (tag, f)); return dict(np.load(p))


print("=" * 100); print("PAD 재보정 상한 진단 (진단 전용) — fold별 seed {0,1,2} 평균, n=5 df=4"); print("=" * 100)
for case, key, rtag, rkey in (("image_missing", "image_missing", "onlymeta_s0", "image_missing"),
                              ("meta_missing", "meta_missing_input0", "onlyimage_s0", "meta_missing_proj0")):
    cols = ["R", "zf", "zf_ubCE", "zf_ubF1", "zf_valF1", "g8", "g8_ubF1"]
    rows = {c: [] for c in cols}
    for f in range(5):
        Rd = L_(rtag, f); rows["R"].append(F1(Rd["test_y"], Rd["test_" + rkey]))
        acc = {c: [] for c in cols if c != "R"}
        for s in (0, 1, 2):
            for m in ("zf", "g8"):
                d = L_("%s_s%d" % (m, s), f); Lt, yt, Lv, yv = d["test_" + key], d["test_y"], d["val_" + key], d["val_y"]
                acc[m].append(F1(yt, Lt)); acc[m + "_ubF1"].append(F1(yt, Lt, fit_f1(Lt, yt)))
                if m == "zf":
                    acc["zf_ubCE"].append(F1(yt, Lt, fit_ce(Lt, yt)))
                    acc["zf_valF1"].append(F1(yt, Lt, fit_f1(Lv, yv)))
        for c, v in acc.items(): rows[c].append(np.mean(v))
    A = {c: np.array(v) for c, v in rows.items()}
    print("\n[%s] macro-F1: " % case + " | ".join("%s %.3f" % (c, A[c].mean()) for c in cols))
    col = (A["R"] - A["zf"]).mean()
    for name in ("zf_ubCE", "zf_ubF1", "zf_valF1"):
        m, lo, hi = ci(A[name] - A["zf"]); ratio = (A[name].mean() - A["zf"].mean()) / col
        tag = ""
        if name == "zf_ubF1":
            tag = " → 판정: %s" % ("편향 재보정으로 메울 수 있는 붕괴" if ratio >= 0.5 else "편향 재보정으로는 메울 수 없는 붕괴")
        print("   %-9s − zf %+.3f [%+.3f,%+.3f] %s | 회복 비율 %.2f (붕괴 R−zf %+.3f)%s" % (name, m, lo, hi, verdict(lo, hi), ratio, col, tag))
    m, lo, hi = ci(A["g8"] - A["zf_ubF1"]); print("   γ8 − zf_ubF1 %+.3f [%+.3f,%+.3f] %s" % (m, lo, hi, verdict(lo, hi)))
    m, lo, hi = ci(A["g8_ubF1"] - A["zf_ubF1"]); print("   참고 γ8_ubF1 − zf_ubF1 %+.3f [%+.3f,%+.3f] %s" % (m, lo, hi, verdict(lo, hi)))
print("\n=== ALL DONE (PAD recal UB) ===")
