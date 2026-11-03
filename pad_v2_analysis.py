"""추가 실험 v2/v3 — PAD 분석 (학습 없음, CPU): E2-PAD(붕괴 정의) · E3(seed 확장 P1~P4) · E1-PAD(부분결손) · E9-PAD(재보정 대 γ)
  python pad_v2_analysis.py
입력: B3 json(seed 0), V2/E3 json(seed 1·2), V2/E2 json(단일모달 천장), V2/dump/*.npz(로짓)
있는 seed만 사용하고, 사용한 seed를 출력한다. 최종 판정은 seed {0,1,2}가 모두 있을 때의 결과로 한다.
[규칙] 사전등록 v2(E1·E2·E3) / E9는 아래(실행 전 고정, 2026-11-03):
  E9-PAD: zf 모델의 결손 케이스 로짓에 validation으로만 학습한 재보정 적용 → test macro-F1.
    주 = 클래스별 편향(6개) 재보정, 보조 = 벡터 스케일링(클래스별 배율+편향 12개). 둘 다 val 교차엔트로피 최소화.
    케이스: image_missing(주) / meta_missing(지시서 B0 정의 = 메타 입력 0). 붕괴 임계 0.05(macro-F1).
    E9-H1 재보정−zf / E9-H2 (재보정−zf)/(R−zf) ≥ 0.5 → 재보정으로 수렴 / E9-H3 γ8 − (zf+재보정) → 처방이 재보정 이상·구분 불가·재보정 우위.
    천장 R: image_missing → only_meta, meta_missing → only_image.
"""
import glob, json, os
import numpy as np
from sklearn.metrics import f1_score
from scipy.optimize import minimize

B3 = "results/thesis_round/B3"; V2 = "results/thesis_round/V2"; DUMP = V2 + "/dump"
FOLDS = range(5); T = {2: 4.303, 3: 3.182, 4: 2.776}; THR = 0.05


def F1(y, logits): return float(f1_score(y, logits.argmax(1), average="macro", labels=list(range(6))))


def ci(d):
    d = np.asarray(d, float); n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    return float(m), float(m - T[n - 1] * se), float(m + T[n - 1] * se)


def verdict(lo, hi): return "개선" if lo > 0 else ("악화" if hi < 0 else "구분 불가")


def js(p):
    try: return json.load(open(p))
    except Exception: return None


def seeds_for(kind):
    pats = {0: B3 + "/pad_%s_f{f}.json", 1: V2 + "/E3/pad_%s_s1_f{f}.json", 2: V2 + "/E3/pad_%s_s2_f{f}.json"}
    return [s for s in (0, 1, 2) if all(os.path.exists((pats[s] % kind).format(f=f)) for f in FOLDS)], pats


def fold_mean(kind, getter):
    S, pats = seeds_for(kind)
    return np.array([np.mean([getter(js((pats[s] % kind).format(f=f))) for s in S]) for f in FOLDS]), S


print("=" * 100); print("PAD 추가 실험 분석"); print("=" * 100)
# ---------------- E3 ----------------
print("\n[E3] seed 확장 — fold별 seed 평균, n=5 df=4")
im = lambda r: r["eval"]["image_missing"]["f1_macro"]; sm = lambda r: r["suff"]["meta_after"]
zf_i, Sz = fold_mean("zerofill", im); g8_i, Sg = fold_mean("gamma_g8.0", im); md_i, Sm = fold_mean("moddrop", im)
zf_s, _ = fold_mean("zerofill", sm); g8_s, _ = fold_mean("gamma_g8.0", sm); md_s, _ = fold_mean("moddrop", sm)
print("  사용 seed: zf %s | γ8 %s | ModDrop %s %s" % (Sz, Sg, Sm, "" if Sz == Sg == Sm == [0, 1, 2] else "(★ seed {0,1,2} 미완 — 예비 결과)"))
m, lo, hi = ci(g8_i - zf_i); print("  P1 γ8−zf (image_missing) %+.3f [%+.3f,%+.3f] → %s" % (m, lo, hi, "재현" if lo > 0 else ("부분 재현" if m > 0 else "재현 안 됨")))
a = ci(g8_s - zf_s); b = ci(md_s - zf_s)
c1 = a[1] > 0; c2 = b[1] <= 0 <= b[2]
print("  P2 γ8 suff_W변화 %+.3f [%+.3f,%+.3f] / ModDrop %+.3f [%+.3f,%+.3f] → %s" % (a + b + ("재현" if c1 and c2 else ("부분 재현" if c1 or c2 else "재현 안 됨"),)))
m, lo, hi = ci(g8_i - md_i); print("  P3 γ8−ModDrop %+.3f [%+.3f,%+.3f] → %s" % (m, lo, hi, "γ 우위" if lo > 0 else ("ModDrop 우위" if hi < 0 else "구분 불가")))
gap = g8_s - g8_i; print("  P4 gap(suff_W after − γ8 image_missing) %.3f ± %.3f" % (gap.mean(), gap.std(ddof=1)))

# ---------------- E2 ----------------
print("\n[E2] 붕괴 정의 통일 (전원결손, seed 0, 임계 0.05)")
for case, onlyk, zk, lk, ok in (("image_missing", "only_meta", "image_missing", "meta", "image_missing"),
                                ("meta_missing", "only_image", "meta_missing_input0", "image_raw", "meta_missing")):
    R = [js(V2 + "/E2/pad_%s_f%d.json" % (onlyk, f)) for f in FOLDS]
    Z = [js(B3 + "/pad_zerofill_f%d.json" % f) for f in FOLDS]
    if any(r is None for r in R):
        print("  %-13s: 단일모달 천장 결과 미완 → 건너뜀" % case); continue
    Rv = np.array([r["eval"][ok]["f1_macro"] for r in R]); Zv = np.array([z["eval"][zk]["f1_macro"] for z in Z])
    Lv = np.array([z["suff"][lk] for z in Z])
    cR, cL = (Rv - Zv).mean(), (Lv - Zv).mean()
    print("  %-13s: R %.3f | L %.3f | zf %.3f | 붕괴_R %+.3f 붕괴_L %+.3f → %s" % (
        case, Rv.mean(), Lv.mean(), Zv.mean(), cR, cL, "정의 간 일치" if (cR >= THR) == (cL >= THR) else "불일치"))

# ---------------- dump 기반: E1-PAD, E9-PAD ----------------
def L(tag, f):
    p = DUMP + "/%s_f%d.npz" % (tag, f)
    return dict(np.load(p)) if os.path.exists(p) else None


avail = [s for s in (0, 1, 2) if all(L("%s_s%d" % (k, s), f) is not None for k in ("zf", "g8", "md") for f in FOLDS)]
have_R = all(L(t, f) is not None for t in ("onlymeta_s0", "onlyimage_s0") for f in FOLDS)
print("\n[dump] 사용 가능 seed: %s | 단일모달 천장 로짓: %s" % (avail, "있음" if have_R else "없음"))
CASES = (("image_missing", "image_missing", "onlymeta_s0"), ("meta_missing", "meta_missing_input0", "onlyimage_s0"))
R_KEY = {"onlymeta_s0": "image_missing", "onlyimage_s0": "meta_missing_proj0"}


def recal_fit(lv, yv, vector):
    K = lv.shape[1]
    def f(th):
        w = th[:K] if vector else np.ones(K); b = th[K:] if vector else th
        z = lv * w + b; z = z - z.max(1, keepdims=True)
        return float(-(z[np.arange(len(yv)), yv] - np.log(np.exp(z).sum(1))).mean())
    x0 = np.concatenate([np.ones(K), np.zeros(K)]) if vector else np.zeros(K)
    th = minimize(f, x0, method="L-BFGS-B").x
    return (lambda l: l * th[:K] + th[K:]) if vector else (lambda l: l + th)


if avail and have_R:
    for case, key, rtag in CASES:
        print("\n[E9-PAD] %s — 재보정 대 γ (fold별 seed %s 평균, n=5 df=4)" % (case, avail))
        rows = {k: [] for k in ("R", "zf", "zf_bias", "zf_vec", "g8", "g8_bias", "md")}
        for f in FOLDS:
            Rd = L(rtag, f); rows["R"].append(F1(Rd["test_y"], Rd["test_" + R_KEY[rtag]]))
            acc = {k: [] for k in rows if k != "R"}
            for s in avail:
                for mdl in ("zf", "g8", "md"):
                    d = L("%s_s%d" % (mdl, s), f); yt, yv = d["test_y"], d["val_y"]
                    acc[mdl].append(F1(yt, d["test_" + key]))
                    if mdl in ("zf", "g8"):
                        acc[mdl + "_bias"].append(F1(yt, recal_fit(d["val_" + key], yv, False)(d["test_" + key])))
                    if mdl == "zf":
                        acc["zf_vec"].append(F1(yt, recal_fit(d["val_" + key], yv, True)(d["test_" + key])))
            for k, v in acc.items(): rows[k].append(np.mean(v))
        A = {k: np.array(v) for k, v in rows.items()}
        print("  macro-F1: " + " | ".join("%s %.3f" % (k, v.mean()) for k, v in A.items()))
        col = (A["R"] - A["zf"]).mean(); h = col >= THR
        m1 = ci(A["zf_bias"] - A["zf"]); m3 = ci(A["g8"] - A["zf_bias"])
        lab = {"개선": "처방(γ8)이 재보정 이상", "구분 불가": "재보정과 구분 불가", "악화": "재보정 우위"}[verdict(m3[1], m3[2])]
        print("  붕괴(R−zf) %+.3f (%s)" % (col, "있음" if h else "없음"))
        print("  E9-H1 재보정(편향)−zf %+.3f [%+.3f,%+.3f] %s" % (m1 + (verdict(m1[1], m1[2]),)))
        print("  E9-H2 %s" % ("해당 없음" if not h else "%.2f %s" % ((A["zf_bias"].mean() - A["zf"].mean()) / col,
              "재보정으로 수렴" if (A["zf_bias"].mean() - A["zf"].mean()) / col >= 0.5 else "미수렴")))
        print("  E9-H3 γ8 − (zf+재보정) %+.3f [%+.3f,%+.3f] → %s" % (m3 + (lab,)))
        for a_, b_ in (("zf_vec", "zf"), ("g8_bias", "zf_bias"), ("md", "zf_bias")):
            mm = ci(A[a_] - A[b_]); print("     참고 %s − %s %+.3f [%+.3f,%+.3f] %s" % ((a_, b_) + mm + (verdict(mm[1], mm[2]),)))

    for ci_, (case, key, rtag) in enumerate(CASES):
        print("\n[E1-PAD] %s 부분결손 (fold별 seed %s 평균, n=5) — oracle = 결손 환자는 단일모달 천장, 나머지는 zf full" % (case, avail))
        for q in (0.1, 0.3, 0.5):
            res = {k: [] for k in ("oracle", "zf", "g8", "md")}
            for f in FOLDS:
                Rd = L(rtag, f); acc = {k: [] for k in res}
                for s in avail:
                    dz = L("zf_s%d" % s, f); yt = dz["test_y"]
                    mk = np.random.RandomState(1000 * f + 10 * ci_ + int(round(10 * q))).rand(len(yt)) < q
                    orc = dz["test_full"].copy(); orc[mk] = Rd["test_" + R_KEY[rtag]][mk]; acc["oracle"].append(F1(yt, orc))
                    for mdl in ("zf", "g8", "md"):
                        d = L("%s_s%d" % (mdl, s), f); lg = d["test_full"].copy(); lg[mk] = d["test_" + key][mk]
                        acc[mdl].append(F1(yt, lg))
                for k, v in acc.items(): res[k].append(np.mean(v))
            A = {k: np.array(v) for k, v in res.items()}; col = (A["oracle"] - A["zf"]).mean()
            tg = ci(A["g8"] - A["zf"]); tm = ci(A["md"] - A["zf"])
            print("  q=%.1f%s | oracle %.3f zf %.3f → 붕괴 %+.3f %-9s | γ8−zf %+.3f [%+.3f,%+.3f] %s | MD−zf %+.3f [%+.3f,%+.3f] %s" % (
                q, "(주)" if q == 0.3 else "    ", A["oracle"].mean(), A["zf"].mean(), col, "붕괴 있음" if col >= THR else "붕괴 없음",
                *tg, verdict(tg[1], tg[2]), *tm, verdict(tm[1], tm[2])))
else:
    print("  → dump 또는 단일모달 천장 로짓이 아직 없어 E1-PAD·E9-PAD 건너뜀")
print("\n=== ALL DONE (PAD v2 analysis) ===")
