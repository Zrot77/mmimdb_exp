"""논문용 표 일괄 생성 — MM-IMDb(n=5)·PAD 수치의 단일 출처 (학습 없음, CPU, 저장된 로짓)
  V2_SEEDS=1,2,3,4,5 python thesis_tables.py            # 서버 ~/Yechan/mmimdb_exp (conda mmimdb)
  V2_SEEDS=1,2,3,4,5 python thesis_tables.py --selftest
출력: results/thesis_round/FINAL/tables.md (논문 붙여넣기용), tables.json (그림 스크립트 입력)
[고정 규칙 — 2026-11-11] 새 판정을 만들지 않는다. 기존 분석과 같은 함수로 n=5 수치를 다시 계산해 한 곳에 모은다.
 - 판정 용어(개선/구분 불가/악화)는 seed(MM-IMDb) 또는 fold(PAD) 짝지은 95% CI의 기술적 표시이며, 사전 주 판정은 기존 결과를 따른다.
 - 사후로 새로 계산하는 항목은 표 제목에 '(사후)'를 붙인다: MAI 절제(재보정 제거), 결합 모델에 MAI 적용, PAD 결합(seed 0) 비교, full 조건 비교.
 - 분해 항등식: 붕괴(R − zf, F1@0.5) = 순위 층(R+재보정 − zf+재보정) + 기준값 층((zf+재보정 − zf) − (R+재보정 − R)).
   seed마다 정확히 성립해야 하며(최대 |오차| 출력, 1e-9 초과 시 중단), 기준값 층은 음수일 수 있다(천장 R이 더 어긋난 경우).
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmimdb_v2_analysis import f1_at05, f1_recal_exact, auprc, ci, verdict, SEEDS, T_N3, NDF
import mmimdb_v2_mai as MAI

OUT = "results/thesis_round/FINAL"
CELLS = (("text", "textmissing", "headi", "R_image_only"), ("image", "imagemissing", "headt", "R_text_only"))
CELLNAME = {"text": "text 결손(이미지 남음)", "image": "image 결손(텍스트 남음)"}
L_RECOMP = {"clip": {"text": 0.4066, "image": 0.4876}, "siglip": {"text": 0.5449, "image": 0.4876}}   # A0 재계산(2026-11-08)


def fmt(c):
    return "%+.3f [%+.3f, %+.3f] %s" % (c[0], c[1], c[2], verdict(c[1], c[2]))


def fmt0(c):
    """크기(붕괴·층) 표시: 0보다 큼 / 0과 구분 불가 / 0보다 작음"""
    tag = "> 0" if c[1] > 0 else ("< 0" if c[2] < 0 else "구분 불가")
    return "%+.3f [%+.3f, %+.3f] %s" % (c[0], c[1], c[2], tag)


def metrics(yv, yt, vl, tl):
    return {"f": f1_at05(yt, tl), "x": f1_recal_exact(yv, vl, yt, tl), "a": auprc(yt, tl)}


def mmimdb(root, bb, out):
    runs, yv, yt = MAI.load(root, bb)
    md = ["## MM-IMDb %s (seed %s, %s, t=%.3f)" % (bb.upper(), SEEDS, NDF, T_N3)]
    J = {}
    for reg, nk, hk, rk in CELLS:
        M, alph = {}, {"g8": [], "gm": []}
        for s in SEEDS:
            M[("R", s)] = metrics(yv, yt, runs[(rk, s)]["val_" + nk], runs[(rk, s)]["test_" + nk])
            for k in ("zf", "md", "g8", "gm"):
                z = runs[(k, s)]; M[(k, s)] = metrics(yv, yt, z["val_" + nk], z["test_" + nk])
            for k in ("g8", "gm"):
                z = runs[(k, s)]; M[(k + "_head", s)] = metrics(yv, yt, z["val_" + hk], z["test_" + hk])
                a, _ = MAI.pick_alpha(yv, z["val_" + nk], z["val_" + hk]); alph[k].append(a)
                M[(k + "_mai", s)] = metrics(yv, yt, a * z["val_" + hk] + (1 - a) * z["val_" + nk], a * z["test_" + hk] + (1 - a) * z["test_" + nk])
        arr = lambda k, m: np.array([M[(k, s)][m] for s in SEEDS])
        D = lambda a, b, m: ci(arr(a, m) - arr(b, m))
        # 분해 항등식
        raw = arr("R", "f") - arr("zf", "f"); rank = arr("R", "x") - arr("zf", "x")
        thr = (arr("zf", "x") - arr("zf", "f")) - (arr("R", "x") - arr("R", "f"))
        err = float(np.abs(raw - (rank + thr)).max())
        if err > 1e-9:
            sys.exit("★ 분해 항등식 오차 %.2e → 중단" % err)
        apg = arr("R", "a") - arr("zf", "a")
        c = {"means": {k: {m: float(arr(k, m).mean()) for m in ("f", "x", "a")} for k in
                       ("R", "zf", "md", "g8", "gm", "g8_head", "gm_head", "g8_mai", "gm_mai")},
             "L": L_RECOMP[bb][reg], "alpha": alph,
             "decomp": {"collapse": ci(raw), "rank": ci(rank), "thr": ci(thr), "zf_gain": ci(arr("zf", "x") - arr("zf", "f")),
                        "R_gain": ci(arr("R", "x") - arr("R", "f")), "ap_gap": ci(apg), "identity_err": err},
             "presc": {}, "aux": {}, "mai": {}}
        for k in ("g8", "md", "gm"):
            gx, ga = D(k, "zf", "x"), D(k, "zf", "a")
            c["presc"][k] = {"x": gx, "a": ga, "fill": float(ga[0] / apg.mean()) if abs(apg.mean()) > 1e-12 else None}
        for k in ("g8", "gm"):
            c["aux"][k] = {"a_head_fused": D(k + "_head", k, "a"), "x_head_fused": D(k + "_head", k, "x"), "a_head_R": D(k + "_head", "R", "a")}
        for name, a, b, m in (("MAI − γ8 융합", "g8_mai", "g8", "x"), ("MAI − ModDrop", "g8_mai", "md", "x"), ("MAI − 결합 융합", "g8_mai", "gm", "x"),
                              ("MAI − 보조만(α=1)", "g8_mai", "g8_head", "x"), ("MAI − 천장 R", "g8_mai", "R", "x"), ("MAI − zf", "g8_mai", "zf", "x"),
                              ("(사후) 재보정 제거: MAI − MAI@0.5", None, None, None), ("(사후) 결합 모델에 MAI: MAI(결합) − 결합 융합", "gm_mai", "gm", "x")):
            if a is None:
                c["mai"][name] = ci(arr("g8_mai", "x") - arr("g8_mai", "f"))
            else:
                c["mai"][name] = D(a, b, m)
        J[reg] = c
    # full 조건 (사후)
    F = {}
    for s in SEEDS:
        for k in ("zf", "md", "g8", "gm"):
            z = runs[(k, s)]; F[(k, s)] = metrics(yv, yt, z["val_full"], z["test_full"])
    fa = lambda k, m: np.array([F[(k, s)][m] for s in SEEDS])
    J["full"] = {"means": {k: {m: float(fa(k, m).mean()) for m in ("f", "x", "a")} for k in ("zf", "md", "g8", "gm")},
                 "cmp": {"%s − zf" % n: {"x": ci(fa(k, "x") - fa("zf", "x")), "a": ci(fa(k, "a") - fa("zf", "a"))}
                         for k, n in (("g8", "γ8"), ("md", "ModDrop"), ("gm", "결합"))},
                 "gm_vs_g8": {"x": ci(fa("gm", "x") - fa("g8", "x")), "a": ci(fa("gm", "a") - fa("g8", "a"))}}
    # 마크다운
    md.append("\n### 표 A. 붕괴와 층 분해 (F1 차이, 분해 항등식 오차 최대 %.1e)" % max(J[r]["decomp"]["identity_err"] for r in ("text", "image")))
    md.append("| 셀 | R (F1@0.5) | L (선형) | zf (F1@0.5) | 붕괴 R − zf | 기준값 층 | 순위 층 | AUPRC R − zf |\n|---|---|---|---|---|---|---|---|")
    for reg in ("text", "image"):
        c = J[reg]; d = c["decomp"]
        md.append("| %s | %.3f | %.3f | %.3f | %s | %s | %s | %s |" % (CELLNAME[reg], c["means"]["R"]["f"], c["L"], c["means"]["zf"]["f"],
                                                                    fmt0(d["collapse"]), fmt0(d["thr"]), fmt0(d["rank"]), fmt0(d["ap_gap"])))
    md.append("\n### 표 B. 처방 효과 (재보정 후 zf 대비 F1 차 / AUPRC 차 / 순위 층 메움 비율)\n| 셀 | 처방 | 재보정 F1 차 | AUPRC 차 | 메움 |\n|---|---|---|---|---|")
    for reg in ("text", "image"):
        for k, n in (("g8", "γ8"), ("md", "ModDrop"), ("gm", "결합")):
            p = J[reg]["presc"][k]
            md.append("| %s | %s | %s | %s | %s |" % (CELLNAME[reg], n, fmt(p["x"]), fmt(p["a"]), "%.2f" % p["fill"] if p["fill"] is not None else "—"))
    md.append("\n### 표 C. 보조 분류기 대 융합 분류기 (AUPRC)\n| 셀 | 모델 | 융합 | 보조 | 천장 R | 보조 − 융합 | 보조 − R |\n|---|---|---|---|---|---|---|")
    for reg in ("text", "image"):
        for k, n in (("g8", "γ8"), ("gm", "결합")):
            m = J[reg]["means"]; a = J[reg]["aux"][k]
            md.append("| %s | %s | %.3f | %.3f | %.3f | %s | %s |" % (CELLNAME[reg], n, m[k]["a"], m[k + "_head"]["a"], m["R"]["a"], fmt(a["a_head_fused"]), fmt(a["a_head_R"])))
    md.append("\n### 표 D. 제안 방법 MAI (재보정 F1 차, 개발 데이터) — 고른 α: " + "; ".join(
        "%s γ8 %s" % (r, J[r]["alpha"]["g8"]) for r in ("text", "image")))
    md.append("| 비교 | %s | %s |\n|---|---|---|" % (CELLNAME["text"], CELLNAME["image"]))
    for name in J["text"]["mai"]:
        md.append("| %s | %s | %s |" % (name, fmt(J["text"]["mai"][name]), fmt(J["image"]["mai"][name])))
    md.append("\n### 표 E. (사후) full 조건 (재보정 F1 차 / AUPRC 차)\n| 비교 | 재보정 F1 | AUPRC |\n|---|---|---|")
    for n, v in J["full"]["cmp"].items():
        md.append("| %s | %s | %s |" % (n, fmt(v["x"]), fmt(v["a"])))
    md.append("| 결합 − γ8 | %s | %s |" % (fmt(J["full"]["gm_vs_g8"]["x"]), fmt(J["full"]["gm_vs_g8"]["a"])))
    out.extend(md)
    return J


def pad(root, out):
    import pad_v4_mai as P
    Lf = lambda d, tag, f: dict(np.load(os.path.join(root, d, "%s_f%d.npz" % (tag, f))))
    if not all(os.path.exists(os.path.join(root, P.DUMPH, "gm_s0_f%d.npz" % f)) for f in P.FOLDS):
        out.append("\n## PAD: 결합(seed 0) 보조 분류기 출력 없음 → PAD 사후 표 생략"); return None
    md = ["\n## PAD (사후 보충: seed 0만 쓴 공정 비교, fold 5개, t=%.3f)" % P.T5,
          "seed 0끼리만 비교 — γ8 융합, MAI(γ8), 결합 융합, MAI(결합), ModDrop. 사전 확인 판정은 pad_v4_mai 결과(seed 0~2)를 따른다.",
          "| 셀 | 비교 | macro-F1 차 | AUROC 차 |\n|---|---|---|---|"]
    J = {}
    for cell, key, hk, rtag, rkey in P.CASES:
        V = {k: {"f": [], "a": []} for k in ("g8", "g8_mai", "gm", "gm_mai", "md")}
        for f in P.FOLDS:
            for tag, base in (("g8_s0", "g8"), ("gm_s0", "gm")):
                d = Lf(P.DUMPH, tag, f); yt = d["test_y"]
                a = P.pick_alpha(d["val_y"], d["val_" + key], d["val_" + hk]); z = a * d["test_" + hk] + (1 - a) * d["test_" + key]
                V[base]["f"].append(P.f1(yt, d["test_" + key])); V[base]["a"].append(P.auroc(yt, d["test_" + key]))
                V[base + "_mai"]["f"].append(P.f1(yt, z)); V[base + "_mai"]["a"].append(P.auroc(yt, z))
            d = Lf(P.DUMP, "md_s0", f); V["md"]["f"].append(P.f1(d["test_y"], d["test_" + key])); V["md"]["a"].append(P.auroc(d["test_y"], d["test_" + key]))
        V = {k: {m: np.array(v) for m, v in d.items()} for k, d in V.items()}
        J[cell] = {}
        for name, a, b in (("MAI(γ8) − γ8 융합", "g8_mai", "g8"), ("MAI(결합) − 결합 융합", "gm_mai", "gm"), ("MAI(γ8) − 결합 융합", "g8_mai", "gm"),
                           ("결합 융합 − ModDrop", "gm", "md"), ("MAI(결합) − ModDrop", "gm_mai", "md")):
            x, y = P.ci(V[a]["f"] - V[b]["f"]), P.ci(V[a]["a"] - V[b]["a"]); J[cell][name] = {"f1": x, "auroc": y}
            md.append("| %s 결손 | %s | %s | %s |" % (cell, name, fmt(x), fmt(y)))
    out.extend(md)
    return J


def main(root="."):
    out = ["# 논문용 표 (자동 생성, thesis_tables.py)", "판정 표시는 짝지은 95% CI의 기술 표시. '(사후)' 항목은 사전 판정이 아님."]
    res = {"seeds": SEEDS}
    for bb in ("clip", "siglip"):
        res[bb] = mmimdb(root, bb, out)
    res["pad_seed0"] = pad(root, out)
    os.makedirs(os.path.join(root, OUT), exist_ok=True)
    open(os.path.join(root, OUT, "tables.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    json.dump(res, open(os.path.join(root, OUT, "tables.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print("\n".join(out)); print("\n결과 저장: %s/tables.md, tables.json" % OUT); print("=== ALL DONE (논문용 표) ===")


def selftest():
    ok = []
    # 1) 분해 항등식은 대수적으로 항상 성립
    rng = np.random.RandomState(0)
    for _ in range(100):
        Rf, Rx, Zf, Zx = rng.rand(4)
        raw, rank, thr = Rf - Zf, Rx - Zx, (Zx - Zf) - (Rx - Rf)
        ok.append(abs(raw - (rank + thr)) < 1e-12)
    ok = [all(ok)]
    # 2) 가짜 run 트리에서 전 과정(마크다운 생성까지) 오류 없이 돌고, 순위 층 + 기준값 층 = 붕괴
    import tempfile, shutil
    r0 = np.random.RandomState(1); L, NV, NT = 23, 1200, 1500; pv = np.linspace(0.02, 0.45, L)
    root = tempfile.mkdtemp(prefix="ttab_")
    for d in ("E2_mmimdb", "E9_mmimdb", "E9H_mmimdb"):
        os.makedirs(os.path.join(root, "results/thesis_round/V2", d))
    yv = (r0.rand(NV, L) < pv).astype(np.float32); yt = (r0.rand(NT, L) < pv).astype(np.float32)
    cal = lambda yy, mu, rng, sh: mu * (yy * mu + rng.randn(*yy.shape)) - mu ** 2 / 2 + np.log(pv / (1 - pv)) + sh
    for bb in ("clip", "siglip"):
        for s in SEEDS:
            rng = np.random.RandomState(50 + s)
            G = {nm: (cal(yv, mu, rng, sh), cal(yt, mu, rng, sh)) for nm, mu, sh in (("good", 2.0, 0), ("mid", 1.0, 0), ("off", 1.5, -2.0))}
            def write(d, name, V, T):
                arr = {"val_y": yv, "test_y": yt}
                for k in V:
                    arr["val_" + k], arr["test_" + k] = V[k].astype(np.float32), T[k].astype(np.float32)
                stem = os.path.join(root, "results/thesis_round/V2", d, "%s_%s_s%d" % (bb, name, s))
                np.savez_compressed(stem + "_logits.npz", **arr); json.dump({"config": {}, "eval": {}}, open(stem + ".json", "w"))
            P = lambda tm, im, i: {"full": G["good"][i], "textmissing": tm[i], "imagemissing": im[i]}
            write("E2_mmimdb", "image_only", P(G["good"], G["mid"], 0), P(G["good"], G["mid"], 1))
            write("E2_mmimdb", "text_only", P(G["mid"], G["good"], 0), P(G["mid"], G["good"], 1))
            write("E9_mmimdb", "zerofill", P(G["off"], G["mid"], 0), P(G["off"], G["mid"], 1))
            write("E9_mmimdb", "moddrop", P(G["mid"], G["good"], 0), P(G["mid"], G["good"], 1))
            for nm in ("gamma_g8.0", "gamma_moddrop_g8.0"):
                write("E9H_mmimdb", nm, dict(P(G["good"], G["mid"], 0), headi=G["good"][0], headt=G["good"][0]),
                      dict(P(G["good"], G["mid"], 1), headi=G["good"][1], headt=G["good"][1]))
    cwd = os.getcwd(); os.chdir(root)
    try:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(root)
        J = json.load(open(os.path.join(root, OUT, "tables.json")))
        d = J["clip"]["text"]["decomp"]
        ok.append(abs(d["collapse"][0] - (d["rank"][0] + d["thr"][0])) < 1e-9)
        ok.append(d["thr"][0] > 0.05)                   # zf를 기준값만 어긋나게 만든 셀 → 기준값 층이 큼
        ok.append(os.path.getsize(os.path.join(root, OUT, "tables.md")) > 2000)
        ok.append("PAD" in open(os.path.join(root, OUT, "tables.md"), encoding="utf-8").read())
    finally:
        os.chdir(cwd); shutil.rmtree(root, ignore_errors=True)
    print("자체 시험:", ok); print("자체 시험 판정:", "모두 통과" if all(ok) else "★실패")
    return all(ok)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    main(".")
