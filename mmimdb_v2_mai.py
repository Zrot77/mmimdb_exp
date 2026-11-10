"""추가 실험 v4 — 제안 방법 MAI(결손 인지 추론) MM-IMDb 개발 분석 (학습 없음, CPU, 저장된 로짓)
  V2_SEEDS=1,2,3,4,5 python mmimdb_v2_mai.py clip | siglip     # 서버 ~/Yechan/mmimdb_exp (conda mmimdb)
  python mmimdb_v2_mai.py --selftest
규칙: thesis_round/ADDENDUM_v4_MAI_prereg.md (2026-11-11). MM-IMDb는 개발용 — 판정 없음, 개발 점검 규칙만 적용.
입력: E9H_mmimdb(γ8·GM, 보조 분류기 출력 포함; seed 4·5는 E9_mmimdb에 있음), E9_mmimdb(zf·MD), E2_mmimdb(R).
MAI: 결손 조건 c에서 z(α) = α·head_r + (1−α)·cls(zero-fill), α ∈ {0,0.1,…,1}을 val macro AUPRC로 고름(동률은 작은 α),
     판단은 라벨별 기준값 재보정(정확 탐색, val). full 조건은 정의상 γ8 융합과 같으므로 계산하지 않음.
"""
import os, sys, json, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmimdb_v2_analysis import f1_at05, f1_recal_exact, auprc, ci, verdict, load_run, SEEDS, T_N3, NDF, SUF

ALPHAS = np.round(np.arange(0, 11) / 10.0, 1)
CELLS = (("text", "textmissing", "headi", "R_image_only"), ("image", "imagemissing", "headt", "R_text_only"))
PREREG = "thesis_round/ADDENDUM_v4_MAI_prereg.md"


def pick_alpha(yv, fv, hv):
    """val macro AUPRC 최대 α (동률은 작은 α). 반환 (α, α별 val AUPRC)"""
    sc = [auprc(yv, a * hv + (1 - a) * fv) for a in ALPHAS]
    best = max(sc); i = min(k for k, s in enumerate(sc) if s >= best - 1e-12)
    return float(ALPHAS[i]), sc


def mai(yv, yt, fv, hv, ft, ht):
    a, _ = pick_alpha(yv, fv, hv)
    zv, zt = a * hv + (1 - a) * fv, a * ht + (1 - a) * ft
    return a, {"x": f1_recal_exact(yv, zv, yt, zt), "a": auprc(yt, zt), "f": f1_at05(yt, zt)}


def load(root, bb):
    v2 = os.path.join(root, "results/thesis_round/V2")
    names = {"R_image_only": ("E2_mmimdb", "image_only"), "R_text_only": ("E2_mmimdb", "text_only"), "zf": ("E9_mmimdb", "zerofill"),
             "md": ("E9_mmimdb", "moddrop"), "g8": ("E9H_mmimdb", "gamma_g8.0"), "gm": ("E9H_mmimdb", "gamma_moddrop_g8.0")}
    runs = {}
    for k, (d, m) in names.items():
        for s in SEEDS:
            r = load_run(os.path.join(v2, d, "%s_%s_s%d" % (bb, m, s)))
            if r is None and d == "E9H_mmimdb":
                r2 = load_run(os.path.join(v2, "E9_mmimdb", "%s_%s_s%d" % (bb, m, s)))
                r = r2 if (r2 is not None and "test_headi" in r2[1]) else None
            if r is None:
                sys.exit("★ 없는 run: %s/%s_%s_s%d → 중단" % (d, bb, m, s))
            runs[(k, s)] = r[1]
    for k in ("g8", "gm"):
        for s in SEEDS:
            if "test_headi" not in runs[(k, s)]:
                sys.exit("★ %s s%d 보조 분류기 출력 없음 → 중단" % (k, s))
    yv, yt = runs[("zf", SEEDS[0])]["val_y"], runs[("zf", SEEDS[0])]["test_y"]
    for (k, s), z in runs.items():
        if not (np.array_equal(z["val_y"], yv) and np.array_equal(z["test_y"], yt)):
            sys.exit("★ K3 라벨 불일치 %s s%d → 중단" % (k, s))
    return runs, yv, yt


def analyze(root, bb):
    runs, yv, yt = load(root, bb)
    pr = os.path.join(root, PREREG)
    md5 = hashlib.md5(open(pr, "rb").read()).hexdigest() if os.path.exists(pr) else "없음"
    print("=" * 104); print("MM-IMDb [%s] MAI 개발 분석 | seed %s, %s, t=%.3f | 사전 고정 문서 md5 %s" % (bb, SEEDS, NDF, T_N3, md5)); print("=" * 104)
    print("[K3] 라벨 동일성 통과(%d개 run)" % len(runs))
    res = {"backbone": bb, "prereg_md5": md5, "cells": {}}; dev_flags = []
    for reg, nk, hk, rk in CELLS:
        M, alphas = {}, {"g8": [], "gm": []}
        for s in SEEDS:
            def std(tag, key, z):
                M[(tag, s)] = {"x": f1_recal_exact(yv, z["val_" + key], yt, z["test_" + key]), "a": auprc(yt, z["test_" + key]),
                               "f": f1_at05(yt, z["test_" + key])}
            std("R", nk, runs[(rk, s)]); std("zf", nk, runs[("zf", s)]); std("md", nk, runs[("md", s)])
            for mdl in ("g8", "gm"):
                z = runs[(mdl, s)]
                std(mdl + "_fused", nk, z); std(mdl + "_head", hk, z)
                a, m = mai(yv, yt, z["val_" + nk], z["val_" + hk], z["test_" + nk], z["test_" + hk])
                M[(mdl + "_mai", s)] = m; alphas[mdl].append(a)
        tags = ["R", "zf", "md", "g8_fused", "g8_head", "g8_mai", "gm_fused", "gm_head", "gm_mai"]
        print("\n" + "-" * 104); print("[%s 결손] 보조 = %s | 고른 α: γ8 %s, GM %s" % (reg, hk, alphas["g8"], alphas["gm"])); print("-" * 104)
        for t in tags:
            A = lambda m: np.mean([M[(t, s)][m] for s in SEEDS])
            print("  %-9s F1@0.5 %.4f | F1@재보정 %.4f | AUPRC %.4f" % (t, A("f"), A("x"), A("a")))
        D = lambda a, b, m: ci(np.array([M[(a, s)][m] - M[(b, s)][m] for s in SEEDS]))
        cell = {"alpha": alphas, "cmp": {}}
        comps = [("M1 MAI − γ8 융합(α=0)", "g8_mai", "g8_fused"), ("M2 MAI − ModDrop", "g8_mai", "md"),
                 ("M3 MAI − GM 융합", "g8_mai", "gm_fused"), ("M4 MAI − 보조만(α=1)", "g8_mai", "g8_head"),
                 ("M5 MAI − 천장 R", "g8_mai", "R"), ("M6 MAI − zf", "g8_mai", "zf"), ("참고 MAI(GM) − GM 융합", "gm_mai", "gm_fused")]
        for name, a, b in comps:
            x, p = D(a, b, "x"), D(a, b, "a")
            cell["cmp"][name] = {"x": x, "a": p}
            print("  %-24s 재보정 F1 %+.4f [%+.4f,%+.4f] %-5s | AUPRC %+.4f [%+.4f,%+.4f] %s" % (
                name, *x, verdict(x[1], x[2]), *p, verdict(p[1], p[2])))
            if name.startswith(("M1", "M4")) and x[2] < 0:
                dev_flags.append("%s %s" % (reg, name))
        nr = D("g8_mai", "g8_fused", "f"); cell["cmp"]["절제 재보정 제거: MAI@0.5 − γ8 융합@0.5"] = {"f": nr}
        print("  %-24s F1@0.5 %+.4f [%+.4f,%+.4f] %s" % ("절제 재보정 제거(@0.5)", *nr, verdict(nr[1], nr[2])))
        res["cells"][reg] = cell
    print("\n▶ 개발 점검(사전 고정): " + ("★설계 결함 의심 — PAD 추출 전에 문서 개정 필요: %s" % dev_flags if dev_flags
                                       else "MAI가 α=0·α=1보다 재보정 F1로 나쁜 셀 없음 → 설계 그대로 PAD 확인으로 진행"))
    res["dev_flags"] = dev_flags
    return res


def selftest():
    import tempfile, shutil
    ok = []
    # 1) α 선택: 보조가 좋으면 1 쪽, 나쁘면 0 쪽, 서로 보완이면 중간
    rng = np.random.RandomState(0); L, N = 10, 3000; prev = np.linspace(0.05, 0.4, L)
    y = (rng.rand(N, L) < prev).astype(np.float32)
    sig_ = lambda mu: mu * (2 * y - 1) + rng.randn(N, L)
    good, bad = sig_(2.0), sig_(0.3)
    a1, _ = pick_alpha(y, bad, good); a0, _ = pick_alpha(y, good, bad)
    n1, n2 = rng.randn(N, L), rng.randn(N, L); comp_f, comp_h = (2 * y - 1) + n1, (2 * y - 1) + n2
    am, _ = pick_alpha(y, comp_f, comp_h)
    ok += [a1 >= 0.9, a0 <= 0.1, 0.3 <= am <= 0.7]
    print("[α 선택] 보조 좋음 → %.1f, 보조 나쁨 → %.1f, 독립 잡음 보완 → %.1f" % (a1, a0, am))
    # 2) 전체 경로: 가짜 run 트리에서 양성(텍스트 보조 좋음)·음성(보조 = 융합 수준) 대조
    for case in ("pos", "neg"):
        r0 = np.random.RandomState(1); L, NV, NT = 23, 2600, 4000; pv = np.linspace(0.02, 0.45, L)
        root = tempfile.mkdtemp(prefix="mai_")
        for d in ("E2_mmimdb", "E9_mmimdb", "E9H_mmimdb"):
            os.makedirs(os.path.join(root, "results/thesis_round/V2", d))
        yv = (r0.rand(NV, L) < pv).astype(np.float32); yt = (r0.rand(NT, L) < pv).astype(np.float32)
        cal = lambda yy, mu, rng: mu * (yy * mu + rng.randn(*yy.shape)) - mu ** 2 / 2 + np.log(pv / (1 - pv))
        def write(d, name, s, V, T):
            arr = {"val_y": yv, "test_y": yt}
            for k in V:
                arr["val_" + k], arr["test_" + k] = V[k].astype(np.float32), T[k].astype(np.float32)
            stem = os.path.join(root, "results/thesis_round/V2", d, "siglip_%s_s%d" % (name, s))
            np.savez_compressed(stem + "_logits.npz", **arr); json.dump({"config": {}, "eval": {}}, open(stem + ".json", "w"))
        for s in SEEDS:
            rng = np.random.RandomState(700 + s)
            G = {nm: (cal(yv, mu, rng), cal(yt, mu, rng)) for nm, mu in (("good", 2.0), ("mid", 1.0))}
            P = lambda tm, im, i: {"full": G["good"][i], "textmissing": tm[i], "imagemissing": im[i]}
            for d, nm, tm, im in (("E2_mmimdb", "image_only", "good", "mid"), ("E2_mmimdb", "text_only", "mid", "good"),
                                  ("E9_mmimdb", "zerofill", "good", "mid"), ("E9_mmimdb", "moddrop", "mid", "good")):
                write(d, nm, s, P(G[tm], G[im], 0), P(G[tm], G[im], 1))
            ht = G["good"] if case == "pos" else G["mid"]
            for nm in ("gamma_g8.0", "gamma_moddrop_g8.0"):
                write("E9H_mmimdb", nm, s, dict(P(G["good"], G["mid"], 0), headi=G["good"][0], headt=ht[0] - 1.0),
                      dict(P(G["good"], G["mid"], 1), headi=G["good"][1], headt=ht[1] - 1.0))
        r = analyze(root, "siglip"); shutil.rmtree(root, ignore_errors=True)
        m1 = r["cells"]["image"]["cmp"]["M1 MAI − γ8 융합(α=0)"]["x"]
        ok.append(m1[1] > 0 if case == "pos" else (m1[1] <= 0 <= m1[2] or abs(m1[0]) < 0.01))
        print("[자체 시험 %s] image 결손 M1 재보정 F1 %+.4f [%+.4f,%+.4f]" % (case, *m1))
    print("\n자체 시험:", ok); print("자체 시험 판정:", "모두 통과" if all(ok) else "★실패")
    return all(ok)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    BB = sys.argv[1]; r = analyze(".", BB)
    out = "results/thesis_round/V2/mmimdb_v4_mai_%s%s.json" % (BB, SUF)
    json.dump(r, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float); print("\n결과 저장:", out)
    print("=== ALL DONE (MM-IMDb MAI 개발 %s) ===" % BB)
