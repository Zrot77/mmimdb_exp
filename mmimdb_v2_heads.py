"""추가 실험 v2 — MM-IMDb E9-H: γ가 학습한 '보조 분류기'는 좋은데 융합 분류기가 그것을 못 쓰는가 (원인 설명 검정)
  python mmimdb_v2_heads.py siglip | clip
  python mmimdb_v2_heads.py --selftest
입력: results/thesis_round/V2/E9H_mmimdb/{BB}_{gamma_g8.0,gamma_moddrop_g8.0}_s{1,2,3}.json + _logits.npz
      (train_v12 --save_logits가 보조 분류기 출력 val/test_headi, _headt도 저장하는 버전으로 재학습. 실행기 --outdir E9H_mmimdb)
      + 기존 E9(zf·γ8·MD·GM 로짓), E2(R). 계산 함수는 mmimdb_v2_analysis.py 것을 그대로 씀(재보정 = 정확 탐색).

[고정 규칙 — 결과 열람 전, 2026-11-07]
배경(구조적 사실, 코드 확인): 텍스트 경로 = 고정 CLIP 텍스트 특징 → 선형 투영 pt → 융합 분류기(선형) cls(0, pt).
  γ는 별도 보조 분류기 head_t(pt)·head_i(pi)에 손실을 주고, 융합 분류기 cls는 결손 입력을 학습 중에 보지 않는다.
설명(검정 대상): image 결손에서 γ8이 순위를 못 고친 것은, γ가 좋은 텍스트 단독 분류기(head_t)를 학습했지만
  융합 분류기의 텍스트 가중치는 그 방향으로 바뀌지 않았기 때문이다.
조건(셀): text 결손 → 보조 head_i(pi) vs 융합 cls(pi,0) / image 결손 → 보조 head_t(pt) vs 융합 cls(0,pt). 모델 γ8, GM.
비교(seed 1·2·3 paired, n=3, t=4.303):
  H6a AUPRC(보조) − AUPRC(융합)       H6b 보조+재보정 − 융합+재보정
  H6c 보조 vs 천장 R: AUPRC, 재보정 F1 (보고)
  H6d 실용: γ8 보조+재보정 − MD+재보정, γ8 보조+재보정 − GM 융합+재보정 (보고)
주 판정(SigLIP, γ8, image 결손): H6a·H6b 모두 개선 → "설명 지지: γ의 텍스트 보조 분류기는 좋지만 융합 분류기가 쓰지 못함"
  하나만 개선 → "부분 지지" / 둘 다 개선 아님 → "설명 불지지". CLIP·GM은 같은 계산을 보고만 함.
점검: K0 재현 — 새 run의 융합 로짓(full/text/image 결손, val·test)이 기존 E9 로짓과 정확히 같아야 함(최대 |차| 출력, 0이 아니면 경고)
      K1 로짓 재계산 F1 = json, K3 라벨 동일성. 보조 분류기 출력 키가 없으면 중단.
한계: n=3, 보조 분류기는 γ 손실로만 학습됨(재보정 필요할 수 있음), MM-IMDb 한 데이터셋.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmimdb_v2_analysis import f1_at05, f1_recal_exact, auprc, ci, verdict, load_run, SEEDS, T_N3

CELLS = (("text", "textmissing", "headi", "R_image_only"), ("image", "imagemissing", "headt", "R_text_only"))


def analyze(root, bb):
    v2 = os.path.join(root, "results/thesis_round/V2")
    names = {"R_image_only": ("E2_mmimdb", "image_only"), "R_text_only": ("E2_mmimdb", "text_only"), "zf": ("E9_mmimdb", "zerofill"),
             "md": ("E9_mmimdb", "moddrop"), "g8_old": ("E9_mmimdb", "gamma_g8.0"), "gm_old": ("E9_mmimdb", "gamma_moddrop_g8.0"),
             "g8": ("E9H_mmimdb", "gamma_g8.0"), "gm": ("E9H_mmimdb", "gamma_moddrop_g8.0")}
    runs, missing = {}, []
    for k, (d, m) in names.items():
        for s in SEEDS:
            r = load_run(os.path.join(v2, d, "%s_%s_s%d" % (bb, m, s)))
            (missing.append("%s/%s_%s_s%d" % (d, bb, m, s)) if r is None else runs.__setitem__((k, s), r))
    print("=" * 100); print("MM-IMDb [%s] E9-H 보조 분류기 대 융합 분류기 | seed %s, n=3, t=%.3f" % (bb, SEEDS, T_N3)); print("=" * 100)
    if missing:
        print("★ 없는 run: %s" % ", ".join(missing))
    need = ("R_image_only", "R_text_only", "zf", "md", "g8")
    if any((k, s) not in runs for k in need for s in SEEDS):
        print("★ 필요한 run 부족 → 중단"); return None
    has_gm = all(("gm", s) in runs for s in SEEDS)
    yv, yt = runs[("zf", 1)][1]["val_y"], runs[("zf", 1)][1]["test_y"]
    for (k, s), (j, z) in runs.items():
        if not (np.array_equal(z["val_y"], yv) and np.array_equal(z["test_y"], yt)):
            sys.exit("★ K3 라벨 불일치 %s s%d → 중단" % (k, s))
    print("[K3] 라벨 동일성 통과(%d개 run)" % len(runs))
    for new, old in (("g8", "g8_old"), ("gm", "gm_old")):
        if new == "gm" and not has_gm:
            continue
        for s in SEEDS:
            zn = runs[(new, s)][1]
            if "test_headi" not in zn or "test_headt" not in zn:
                sys.exit("★ %s s%d에 보조 분류기 출력이 없음(구버전 train_v12?) → 중단" % (new, s))
            if (old, s) in runs:
                zo = runs[(old, s)][1]
                dmax = max(float(np.abs(zn[k].astype(np.float64) - zo[k].astype(np.float64)).max())
                           for k in zo if k.endswith(("full", "textmissing", "imagemissing")))
                print("[K0] %s s%d 융합 로짓 vs 기존 E9: 최대 |차| %.2e %s" % (new, s, dmax, "" if dmax == 0 else "★경고(재현 안 됨)"))
            j = runs[(new, s)][0]
            for ek, nk in (("full", "full"), ("text", "textmissing"), ("image", "imagemissing")):
                d = abs(f1_at05(yt, zn["test_" + nk]) - j["eval"][ek]["macro"])
                if d > 1e-6:
                    print("[K1] ★경고 %s s%d %s |차| %.1e" % (new, s, ek, d))
    print("[K1] 점검 완료(경고가 없으면 통과)")
    res = {"backbone": bb, "cells": {}}
    for reg, nk, hk, rk in CELLS:
        M = {}
        def put(tag, k, key):
            for s in SEEDS:
                z = runs[(k, s)][1]
                M[(tag, s)] = {"f1": f1_at05(yt, z["test_" + key]), "x": f1_recal_exact(yv, z["val_" + key], yt, z["test_" + key]),
                               "a": auprc(yt, z["test_" + key])}
        put("R", rk, nk); put("zf", "zf", nk); put("md", "md", nk)
        put("g8_fused", "g8", nk); put("g8_head", "g8", hk)
        tags = ["R", "zf", "md", "g8_fused", "g8_head"]
        if has_gm:
            put("gm_fused", "gm", nk); put("gm_head", "gm", hk); tags += ["gm_fused", "gm_head"]
        D = lambda a, b, m: ci(np.array([M[(a, s)][m] - M[(b, s)][m] for s in SEEDS]))
        print("\n" + "-" * 100); print("[%s 결손] 보조 = %s%s" % (reg, hk, "  ← 주 판정 셀(γ8)" if (bb == "siglip" and reg == "image") else "")); print("-" * 100)
        for t in tags:
            A = lambda m: np.mean([M[(t, s)][m] for s in SEEDS])
            print("  %-9s F1@0.5 %.4f | F1@재보정 %.4f | AUPRC %.4f" % (t, A("f1"), A("x"), A("a")))
        cell = {}
        for mdl in (("g8", "gm") if has_gm else ("g8",)):
            h, f = mdl + "_head", mdl + "_fused"
            a, b = D(h, f, "a"), D(h, f, "x"); cr_a, cr_x = D(h, "R", "a"), D(h, "R", "x")
            cell[mdl] = {"H6a": a, "H6b": b, "H6c_a": cr_a, "H6c_x": cr_x}
            print("  [%s] H6a AUPRC 보조 − 융합 %+.4f [%+.4f,%+.4f] %s | H6b 재보정 F1 보조 − 융합 %+.4f [%+.4f,%+.4f] %s" % (
                mdl, *a, verdict(a[1], a[2]), *b, verdict(b[1], b[2])))
            print("        H6c 보조 − R: AUPRC %+.4f [%+.4f,%+.4f] %s | 재보정 F1 %+.4f [%+.4f,%+.4f] %s" % (
                *cr_a, verdict(cr_a[1], cr_a[2]), *cr_x, verdict(cr_x[1], cr_x[2])))
        p1 = D("g8_head", "md", "x"); cell["H6d_vs_md"] = p1
        line = "  H6d γ8 보조+재보정 − MD+재보정 %+.4f [%+.4f,%+.4f] %s" % (*p1, verdict(p1[1], p1[2]))
        if has_gm:
            p2 = D("g8_head", "gm_fused", "x"); cell["H6d_vs_gm"] = p2
            line += " | − GM 융합+재보정 %+.4f [%+.4f,%+.4f] %s" % (*p2, verdict(p2[1], p2[2]))
        print(line)
        if bb == "siglip" and reg == "image":
            ia, ib = cell["g8"]["H6a"][1] > 0, cell["g8"]["H6b"][1] > 0
            v = ("설명 지지: γ의 텍스트 보조 분류기는 좋지만 융합 분류기가 쓰지 못함" if ia and ib else ("부분 지지" if ia or ib else "설명 불지지"))
            cell["verdict"] = v; print("  ▶ 주 판정: %s" % v)
        res["cells"][reg] = cell
    return res


def selftest():
    import tempfile, shutil
    out = []
    for case in ("pos", "neg"):
        rng0 = np.random.RandomState(0); L, NV, NT = 23, 2600, 4000; prev = np.linspace(0.02, 0.45, L)
        root = tempfile.mkdtemp(prefix="mmhd_")
        for d in ("E2_mmimdb", "E9_mmimdb", "E9H_mmimdb"):
            os.makedirs(os.path.join(root, "results/thesis_round/V2", d))
        yv = (rng0.rand(NV, L) < prev).astype(np.float32); yt = (rng0.rand(NT, L) < prev).astype(np.float32)
        cal = lambda y, mu, rng: mu * (y * mu + rng.randn(*y.shape)) - mu ** 2 / 2 + np.log(prev / (1 - prev))
        def write(d, name, s, V, T):
            arr = {"val_y": yv, "test_y": yt}; ev = {}
            for nk in V:
                arr["val_" + nk], arr["test_" + nk] = V[nk].astype(np.float32), T[nk].astype(np.float32)
            for nk, ek in (("full", "full"), ("textmissing", "text"), ("imagemissing", "image")):
                ev[ek] = {"macro": f1_at05(yt, arr["test_" + nk])}
            stem = os.path.join(root, "results/thesis_round/V2", d, "siglip_%s_s%d" % (name, s))
            np.savez_compressed(stem + "_logits.npz", **arr); json.dump({"config": {}, "eval": ev}, open(stem + ".json", "w"))
        for s in SEEDS:
            rng = np.random.RandomState(500 + s)
            G = {nm: (cal(yv, mu, rng), cal(yt, mu, rng)) for nm, mu in (("good", 2.0), ("mid", 1.0))}
            P = lambda tm, im, i: {"full": G["good"][i], "textmissing": tm[i], "imagemissing": im[i]}
            write("E2_mmimdb", "image_only", s, P(G["good"], G["mid"], 0), P(G["good"], G["mid"], 1))
            write("E2_mmimdb", "text_only", s, P(G["mid"], G["good"], 0), P(G["mid"], G["good"], 1))
            write("E9_mmimdb", "zerofill", s, P(G["good"], G["mid"], 0), P(G["good"], G["mid"], 1))
            write("E9_mmimdb", "moddrop", s, P(G["mid"], G["good"], 0), P(G["mid"], G["good"], 1))
            fusedV, fusedT = P(G["good"], G["mid"], 0), P(G["good"], G["mid"], 1)
            write("E9_mmimdb", "gamma_g8.0", s, fusedV, fusedT)
            ht = G["good"] if case == "pos" else G["mid"]          # 양성: 텍스트 보조 분류기가 좋음 / 음성: 융합과 같은 수준
            write("E9H_mmimdb", "gamma_g8.0", s, dict(fusedV, headi=G["good"][0], headt=ht[0] - 1.0),
                  dict(fusedT, headi=G["good"][1], headt=ht[1] - 1.0))
        r = analyze(root, "siglip"); shutil.rmtree(root, ignore_errors=True)
        exp = "설명 지지: γ의 텍스트 보조 분류기는 좋지만 융합 분류기가 쓰지 못함" if case == "pos" else "설명 불지지"
        out.append(r["cells"]["image"]["verdict"] == exp)
        print("\n[자체 시험 %s] 기대 '%s' → 얻음 '%s'" % (case, exp, r["cells"]["image"]["verdict"]))
    print("\n자체 시험 판정:", "모두 통과" if all(out) else "★실패 %s" % out)
    return all(out)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    BB = sys.argv[1]; r = analyze(".", BB)
    if r is not None:
        out = "results/thesis_round/V2/mmimdb_v2_heads_%s.json" % BB
        json.dump(r, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float); print("\n결과 저장:", out)
    print("=== ALL DONE (MM-IMDb E9-H %s) ===" % BB)
