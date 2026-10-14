"""B3 집계 — PAD SigLIP so400m 5-fold (fold 짝지음, seed 0 고정).
단계 1 (select): python agg_b3.py select
  - 재현점검: B3 zerofill vs B2 zerofill, fold별 |Δ| (full·image_missing macro-F1). >0.010이면 중단·보고.
  - γ 선택: 대표 결손 케이스의 **val** macro-F1(val_eval) fold 평균이 높은 γ (γ2 vs γ8). 테스트 미사용.
    대표 결손 = B2 사전규칙(전역 raw suff 평균)으로 정한 W만 남는 케이스.
단계 2 (final): python agg_b3.py final --gsel <2.0|8.0>
  - 조건별 fold평균, paired 차이(95%CI, ddof=1, n=5, df=4, t=2.776), 2x2 interaction I_f = gm−γ−md+zf.
판정(P1~P4)은 원본 지시서 규칙으로만 — 이 스크립트는 수치만 산출.
"""
import glob, json, sys
import numpy as np

T = 2.776; N = 5
D = "results/thesis_round"
CASES = ["full", "image_missing", "meta_missing", "meta_missing_input0"]


def load(pattern):
    fs = sorted(glob.glob(pattern)); rows = {json.load(open(f))["fold"]: json.load(open(f)) for f in fs}
    assert sorted(rows) == [0, 1, 2, 3, 4], "fold 0~4 필요: %s → %s" % (pattern, sorted(rows))
    return [rows[k] for k in range(5)]


def rep_case():
    b2 = load(D + "/B2/pad_zerofill_f*.json")
    si = np.mean([r["suff"]["image_raw"] for r in b2]); sm = np.mean([r["suff"]["meta"] for r in b2])
    return ("image_missing" if si > sm else "meta_missing"), b2   # S(강) 결손 → W만 남음


def f1(rows, case, split="eval"):
    return np.array([r[split][case]["f1_macro"] for r in rows])


def ci(d):
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(N); return m, m - T * se, m + T * se


def select():
    rc, b2 = rep_case()
    zf = load(D + "/B3/pad_zerofill_f*.json")
    print("=" * 70); print("B3 단계1 — 재현점검 + γ 선택 | 대표 결손 = %s" % rc); print("=" * 70)
    worst = 0.0
    for c in ("full", "image_missing"):
        d = f1(zf, c) - f1(b2, c); worst = max(worst, np.abs(d).max())
        print("재현 %-13s B3−B2 fold별 Δ: %s" % (c, " ".join("%+.4f" % x for x in d)))
    print("재현점검 max|Δ| = %.4f → %s" % (worst, "통과(≤0.010)" if worst <= 0.010 else "★초과 → 중단·보고"))
    print("[확인] val_eval full == val_best_f1: %s" % all(
        abs(r["val_eval"]["full"]["f1_macro"] - r["val_best_f1"]) < 1e-6 for r in zf))
    print("-" * 70)
    vals = {}
    for g in ("2.0", "8.0"):
        rows = load(D + "/B3/pad_gamma_g%s_f*.json" % g)
        v = f1(rows, rc, "val_eval"); vals[g] = v.mean()
        print("γ%s val %s macro-F1 fold별 %s | 평균 %.4f" % (g, rc, " ".join("%.3f" % x for x in v), v.mean()))
    if vals["2.0"] == vals["8.0"]:
        print("★동률 → 사용자 질문"); return
    gsel = max(vals, key=vals.get)
    print("→ 선택 γ = %s (val 기준). 다음: gamma_moddrop --gamma %s 5 fold" % (gsel, gsel))


def final(gsel):
    rc, _ = rep_case()
    C = {"zf": load(D + "/B3/pad_zerofill_f*.json"),
         "g2": load(D + "/B3/pad_gamma_g2.0_f*.json"),
         "g8": load(D + "/B3/pad_gamma_g8.0_f*.json"),
         "md": load(D + "/B3/pad_moddrop_f*.json"),
         "gm": load(D + "/B3/pad_gamma_moddrop_g%s_f*.json" % gsel)}
    gk = "g2" if gsel.startswith("2") else "g8"
    print("=" * 70); print("B3 단계2 — 대표 결손 = %s | 선택 γ = %s | n=5 df=4" % (rc, gsel)); print("=" * 70)
    print("[fold평균 test macro-F1]")
    print("%-5s " % "" + " ".join("%-20s" % c for c in CASES))
    for k, rows in C.items():
        print("%-5s " % k + " ".join("%-20.3f" % f1(rows, c).mean() for c in CASES))
    print("[학습후 suff] " + " | ".join("%s img %.3f meta %.3f" % (k, np.mean([r["suff"]["image_after"] for r in v]),
                                                              np.mean([r["suff"]["meta_after"] for r in v])) for k, v in C.items()))
    print("[raw suff] img %.3f meta %.3f" % (np.mean([r["suff"]["image_raw"] for r in C["zf"]]),
                                            np.mean([r["suff"]["meta"] for r in C["zf"]])))
    print("-" * 70); print("[paired 차이, 95%%CI]  (case: %s 및 full)" % rc)
    pairs = [("γsel − zf", gk, "zf"), ("md − zf", "md", "zf"), ("M = gm − γsel", "gm", gk),
             ("gm − md", "gm", "md"), ("γ8 − γ2", "g8", "g2")]
    for c in (rc, "full"):
        for nm, a, b in pairs:
            m, lo, hi = ci(f1(C[a], c) - f1(C[b], c))
            print("  %-13s %-15s %+.3f [%+.3f, %+.3f]" % (c, nm, m, lo, hi))
    I = f1(C["gm"], rc) - f1(C[gk], rc) - f1(C["md"], rc) + f1(C["zf"], rc)
    m, lo, hi = ci(I)
    print("-" * 70)
    print("2x2 interaction I (%s) fold별 %s | 평균 %+.3f 95%%CI [%+.3f, %+.3f] df=4" % (
        rc, " ".join("%+.3f" % x for x in I), m, lo, hi))
    tot = sum(r["run"]["elapsed_sec"] for v in C.values() for r in v) / 3600
    print("총 GPU시간 %.2f h" % tot)


if __name__ == "__main__":
    if sys.argv[1] == "select": select()
    elif sys.argv[1] == "final": final(sys.argv[sys.argv.index("--gsel") + 1])
