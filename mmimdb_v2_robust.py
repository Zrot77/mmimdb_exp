"""추가 실험 v2 — MM-IMDb 견고성 점검 R1·R2·R3 + E4 입력 (학습 없음, CPU, 저장된 로짓만 사용)
  python mmimdb_v2_robust.py clip | siglip     # 서버 ~/Yechan/mmimdb_exp
  python mmimdb_v2_robust.py --selftest
입력: results/thesis_round/V2/E2_mmimdb/{BB}_{image_only,text_only}_s*, E9_mmimdb/{BB}_{zerofill,gamma_g8.0,moddrop,gamma_moddrop_g8.0}_s*
계산 함수는 mmimdb_v2_analysis.py 것을 그대로 씀(재보정 = 정확 탐색 규칙).

[고정 규칙 — 실행 전, 2026-11-07]
비교 목록(셀 = text 결손 / image 결손, 각 백본):
  C1 zf+재보정 − zf(F1@0.5)      C2 R+재보정 − zf+재보정(남는 붕괴)   C3 AUPRC R − zf
  C4 γ8+재보정 − zf+재보정        C5 AUPRC γ8 − zf
  C6 MD+재보정 − zf+재보정        C7 AUPRC MD − zf
  C8 GM+재보정 − (text: γ8 / image: MD)+재보정   C9 같은 비교의 AUPRC
R1 test 표본 부트스트랩: B=1000, 난수 20261107, test 행을 복원추출(모든 모델·seed에 같은 행), 재보정 기준값은 val에서 한 번 맞춘 것 고정.
   통계량 = seed 평균 차이, 95% 백분위 구간. seed CI(n=3, t=4.303)와 부트스트랩 구간의 판정(개선/구분 불가/악화)이 같으면 "일치".
R2 희귀 라벨: val 양성 30개 미만 라벨을 빼고 같은 비교(seed CI)를 다시 계산. 판정이 원래와 같으면 "일치".
R3 val 크기: val을 무작위로 n ∈ {300, 600, 1200, 전체}개로 줄여(반복 20회, 난수 1000+반복) 기준값을 다시 맞춤.
   보고 = zf 재보정 이득(C1)과 남는 붕괴(C2)의 반복 평균. 해석 규칙: text 결손에서 n=300의 C1 평균이 전체 val C1의 50% 미만이면
   "val 크기 설명과 일치", 50% 이상이면 "val 크기만으로는 PAD 실패를 설명하기 어려움"(PAD fold당 val은 약 300장 수준).
E4 입력(사전등록 v2 §E4, 전원결손 프로토콜): 붕괴 = R − zf(F1@0.5) ≥ 0.05, γ 이득 = γ8 − zf(F1@0.5) seed CI.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmimdb_v2_analysis import f1_at05, fit_thr_exact, auprc, ci, verdict, load_run, SEEDS, macro_f1, SUF

B, BOOT_SEED, MINPOS, VAL_NS, REPS = 1000, 20261107, 30, (300, 600, 1200, None), 20
CELLS = (("text", "textmissing", "R_image_only", "g8"), ("image", "imagemissing", "R_text_only", "md"))


def load_all(root, bb):
    e2d, e9d = (os.path.join(root, "results/thesis_round/V2", d) for d in ("E2_mmimdb", "E9_mmimdb"))
    names = {"R_image_only": (e2d, "image_only"), "R_text_only": (e2d, "text_only"), "zf": (e9d, "zerofill"),
             "g8": (e9d, "gamma_g8.0"), "md": (e9d, "moddrop"), "gm": (e9d, "gamma_moddrop_g8.0")}
    runs = {}
    for k, (d, m) in names.items():
        for s in SEEDS:
            r = load_run(os.path.join(d, "%s_%s_s%d" % (bb, m, s)))
            if r is None:
                sys.exit("★ 없는 run: %s_%s_s%d → 중단" % (bb, m, s))
            runs[(k, s)] = r[1]
    return runs


def wf1(pred, y, w, labs):
    """가중(부트스트랩 횟수) macro-F1. pred,y: (N,L) bool, w: (B,N) → (B,)"""
    tp = w @ (pred & y)[:, labs].astype(np.float32); fp = w @ (pred & ~y)[:, labs].astype(np.float32)
    fn = w @ (~pred & y)[:, labs].astype(np.float32); den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0).mean(1)


def wap(score, y, w, labs):
    """가중 average precision(라벨 평균, 해당 resample에서 양성 0인 라벨은 제외). → (B,)"""
    out = np.zeros(w.shape[0]); cnt = np.zeros(w.shape[0])
    for j in labs:
        o = np.argsort(-score[:, j], kind="mergesort"); yy = y[o, j].astype(np.float32); ww = w[:, o]
        cw = np.cumsum(ww, 1); ctp = np.cumsum(ww * yy, 1); P = ctp[:, -1]
        prec = np.where(cw > 0, ctp / np.maximum(cw, 1e-12), 0.0)
        ap = (prec * ww * yy).sum(1) / np.maximum(P, 1e-12)
        ok = P > 0; out[ok] += ap[ok]; cnt[ok] += 1
    return out / np.maximum(cnt, 1)


def comparisons(runs, nk, rk, best):
    keys = [rk, "zf", "g8", "md", "gm"]
    return keys, [("C1 zf+재보정 − zf", ("zf", "x"), ("zf", "f")), ("C2 R+재보정 − zf+재보정", (rk, "x"), ("zf", "x")),
                  ("C3 AUPRC R − zf", (rk, "a"), ("zf", "a")), ("C4 γ8+재보정 − zf+재보정", ("g8", "x"), ("zf", "x")),
                  ("C5 AUPRC γ8 − zf", ("g8", "a"), ("zf", "a")), ("C6 MD+재보정 − zf+재보정", ("md", "x"), ("zf", "x")),
                  ("C7 AUPRC MD − zf", ("md", "a"), ("zf", "a")),
                  ("C8 GM+재보정 − %s+재보정" % ("γ8" if best == "g8" else "MD"), ("gm", "x"), (best, "x")),
                  ("C9 AUPRC GM − %s" % ("γ8" if best == "g8" else "MD"), ("gm", "a"), (best, "a"))]


def analyze(root, bb, B_=B):
    runs = load_all(root, bb)
    z0 = runs[("zf", 1)]; yv, yt = z0["val_y"].astype(bool), z0["test_y"].astype(bool)
    for (k, s), z in runs.items():
        if not (np.array_equal(z["val_y"].astype(bool), yv) and np.array_equal(z["test_y"].astype(bool), yt)):
            sys.exit("★ 라벨 불일치 %s s%d → 중단" % (k, s))
    L = yt.shape[1]; vpos = yv.sum(0); tpos = yt.sum(0)
    print("=" * 100); print("MM-IMDb [%s] 견고성 점검 R1·R2·R3 + E4 입력 | seed %s" % (bb, SEEDS)); print("=" * 100)
    print("라벨별 양성 수 (val/test): " + ", ".join("%d/%d" % (a, b) for a, b in zip(vpos, tpos)))
    keep = [j for j in range(L) if vpos[j] >= MINPOS]
    print("R2: val 양성 %d개 이상 라벨 %d/%d개 유지 (제외: %s)" % (MINPOS, len(keep), L, [j for j in range(L) if j not in keep]))
    rng = np.random.RandomState(BOOT_SEED); N = yt.shape[0]
    W = np.stack([np.bincount(rng.randint(0, N, N), minlength=N) for _ in range(B_)]).astype(np.float32)
    res = {"backbone": bb, "cells": {}}
    for reg, nk, rk, best in CELLS:
        keys, comps = comparisons(runs, nk, rk, best)
        P, S = {}, {}
        for k in keys:
            for s in SEEDS:
                z = runs[(k, s)]; lt = np.asarray(z["test_" + nk], np.float64)
                thr = fit_thr_exact(yv, z["val_" + nk])
                P[(k, s, "x")] = lt >= thr[None, :]
                P[(k, s, "f")] = (1.0 / (1.0 + np.exp(-np.asarray(z["test_" + nk], np.float32)))) > np.float32(0.5)
                S[(k, s)] = lt
        def stat(k, s, m, labs, w):
            return wap(S[(k, s)], yt, w, labs) if m == "a" else wf1(P[(k, s, m)], yt, w, labs)
        one = np.ones((1, N), np.float32)
        print("\n" + "-" * 100); print("[%s 결손]  (seed CI | 부트스트랩 95%% | R2 희귀 라벨 제외)" % reg); print("-" * 100)
        cell = {}
        for name, (a, ma), (b, mb) in comps:
            dseed = np.array([stat(a, s, ma, range(L), one)[0] - stat(b, s, mb, range(L), one)[0] for s in SEEDS])
            c = ci(dseed)
            bt = np.mean([stat(a, s, ma, range(L), W) - stat(b, s, mb, range(L), W) for s in SEEDS], 0)
            lo, hi = np.percentile(bt, [2.5, 97.5])
            d2 = np.array([stat(a, s, ma, keep, one)[0] - stat(b, s, mb, keep, one)[0] for s in SEEDS]); c2 = ci(d2)
            v0, v1, v2 = verdict(c[1], c[2]), verdict(lo, hi), verdict(c2[1], c2[2])
            cell[name] = {"seed": c, "boot": (float(bt.mean()), float(lo), float(hi)), "rare": c2,
                          "R1": v0 == v1, "R2": v0 == v2}
            print("  %-26s %+.4f [%+.4f,%+.4f] %-5s | %+.4f [%+.4f,%+.4f] %-5s %s | %+.4f [%+.4f,%+.4f] %-5s %s" % (
                name, *c, v0, bt.mean(), lo, hi, v1, "일치" if v0 == v1 else "★불일치", *c2, v2, "일치" if v0 == v2 else "★불일치"))
        # E4 입력
        col = np.mean([f1_at05(yt, S[(rk, s)]) - f1_at05(yt, S[("zf", s)]) for s in SEEDS])
        g = ci(np.array([f1_at05(yt, S[("g8", s)]) - f1_at05(yt, S[("zf", s)]) for s in SEEDS]))
        cell["E4"] = {"collapse": float(col), "has": bool(col >= 0.05), "g8_gain": g, "gain_v": verdict(g[1], g[2])}
        print("  [E4 입력] 붕괴(R−zf, F1@0.5) %+.4f → %s | γ8−zf %+.4f [%+.4f,%+.4f] %s → 셀 %s" % (
            col, "있음" if col >= 0.05 else "없음", *g, verdict(g[1], g[2]),
            "일치" if (col >= 0.05) == (g[1] > 0) else "★불일치"))
        # R3
        print("  [R3 val 크기]  n: zf 재보정 이득(C1) 평균 / 남는 붕괴(C2) 평균 / γ8·MD·GM 재보정 F1 (반복 %d회 평균)" % REPS)
        full_c1 = cell["C1 zf+재보정 − zf"]["seed"][0]; r3 = {}
        for n in VAL_NS:
            reps = [None] if n is None else range(REPS)
            acc = {"c1": [], "c2": [], "g8": [], "md": [], "gm": []}
            for rp in reps:
                idx = np.arange(yv.shape[0]) if n is None else np.random.RandomState(1000 + rp).choice(yv.shape[0], n, replace=False)
                f = {}
                for k in (rk, "zf", "g8", "md", "gm"):
                    f[k] = np.mean([macro_f1(yt, S[(k, s)] >= fit_thr_exact(yv[idx], runs[(k, s)]["val_" + nk][idx])[None, :]) for s in SEEDS])
                zf05 = np.mean([f1_at05(yt, S[("zf", s)]) for s in SEEDS])
                acc["c1"].append(f["zf"] - zf05); acc["c2"].append(f[rk] - f["zf"])
                for k in ("g8", "md", "gm"):
                    acc[k].append(f[k])
            m = {k: float(np.mean(v)) for k, v in acc.items()}; sd = float(np.std(acc["c1"], ddof=1)) if len(acc["c1"]) > 1 else 0.0
            r3[str(n or "full")] = m
            print("     %-5s C1 %+.4f (sd %.4f, 전체 대비 %.2f) | C2 %+.4f | γ8 %.4f MD %.4f GM %.4f" % (
                n or "전체", m["c1"], sd, m["c1"] / full_c1 if full_c1 else float("nan"), m["c2"], m["g8"], m["md"], m["gm"]))
        cell["R3"] = r3
        if reg == "text":
            ratio = r3["300"]["c1"] / full_c1
            cell["R3_verdict"] = "val 크기 설명과 일치" if ratio < 0.5 else "val 크기만으로는 PAD 실패를 설명하기 어려움"
            print("  ▶ R3 해석(text 결손, n=300 이득 비율 %.2f): %s" % (ratio, cell["R3_verdict"]))
        res["cells"][reg] = cell
    return res


def selftest():
    import tempfile, shutil, json
    from mmimdb_v2_analysis import f1_at05 as F
    rng0 = np.random.RandomState(0); L, NV, NT = 23, 2600, 3000; prev = np.linspace(0.005, 0.45, L)
    root = tempfile.mkdtemp(prefix="mmrb_")
    for d in ("E2_mmimdb", "E9_mmimdb"):
        os.makedirs(os.path.join(root, "results/thesis_round/V2", d))
    yv = (rng0.rand(NV, L) < prev).astype(np.float32); yt = (rng0.rand(NT, L) < prev).astype(np.float32)
    cal = lambda y, mu, rng: mu * (y * mu + rng.randn(*y.shape)) - mu ** 2 / 2 + np.log(prev / (1 - prev))
    def write(name, s, V, T):
        arr = {"val_y": yv, "test_y": yt}; ev = {}
        for nk, ek in (("full", "full"), ("textmissing", "text"), ("imagemissing", "image")):
            arr["val_" + nk], arr["test_" + nk] = V[nk].astype(np.float32), T[nk].astype(np.float32); ev[ek] = {"macro": F(yt, arr["test_" + nk])}
        stem = os.path.join(root, "results/thesis_round/V2", "E2_mmimdb" if "only" in name else "E9_mmimdb", "clip_%s_s%d" % (name, s))
        np.savez_compressed(stem + "_logits.npz", **arr); json.dump({"config": {}, "eval": ev}, open(stem + ".json", "w"))
    for s in SEEDS:
        rng = np.random.RandomState(400 + s)
        G = {nm: (cal(yv, mu, rng), cal(yt, mu, rng)) for nm, mu in (("good", 2.0), ("good2", 2.0), ("mid", 1.0))}
        P = lambda tm, im, i, st=0.0, si=0.0: {"full": G["good"][i], "textmissing": tm[i] + st, "imagemissing": im[i] + si}
        write("image_only", s, P(G["good"], G["mid"], 0), P(G["good"], G["mid"], 1))
        write("text_only", s, P(G["mid"], G["good"], 0), P(G["mid"], G["good"], 1))
        write("zerofill", s, P(G["good"], G["mid"], 0, -3), P(G["good"], G["mid"], 1, -3))      # text: 순수 기준값 붕괴, image: 순위 붕괴
        write("gamma_g8.0", s, P(G["good"], G["mid"], 0), P(G["good"], G["mid"], 1))
        write("moddrop", s, P(G["good"], G["good"], 0, -3, -1), P(G["good"], G["good"], 1, -3, -1))
        write("gamma_moddrop_g8.0", s, P(G["good"], G["good"], 0), P(G["good"], G["good"], 1))
    r = analyze(root, "clip", B_=200); shutil.rmtree(root, ignore_errors=True)
    t, i = r["cells"]["text"], r["cells"]["image"]
    ok = [t["C1 zf+재보정 − zf"]["seed"][1] > 0, abs(t["C2 R+재보정 − zf+재보정"]["seed"][0]) < 1e-9,
          abs(t["C3 AUPRC R − zf"]["boot"][0]) < 1e-6, i["C7 AUPRC MD − zf"]["boot"][1] > 0,
          i["C3 AUPRC R − zf"]["seed"][1] > 0, t["E4"]["has"] and t["E4"]["gain_v"] == "개선",
          abs(t["C3 AUPRC R − zf"]["rare"][0]) < 1e-6, "300" in t["R3"]]
    print("\n자체 시험 판정:", "모두 통과" if all(ok) else "★실패 %s" % ok)
    return all(ok)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    import json
    BB = sys.argv[1]; r = analyze(".", BB)
    out = "results/thesis_round/V2/mmimdb_v2_robust_%s%s.json" % (BB, SUF)
    json.dump(r, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
    print("\n결과 저장:", out); print("=== ALL DONE (MM-IMDb 견고성 %s) ===" % BB)
