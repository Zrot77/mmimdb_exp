"""추가 실험 v2 — MM-IMDb E9-MD: ModDrop이 '순위 층' 붕괴를 메우는가 (zf·γ8·R은 기존 E2·E9 run 재사용)
  python mmimdb_v2_moddrop.py siglip        # 주 대상
  python mmimdb_v2_moddrop.py clip          # 완결성(같은 규칙)
  python mmimdb_v2_moddrop.py --selftest    # 가짜 로짓 대조 시험
입력: results/thesis_round/V2/E9_mmimdb/{BB}_moddrop_s{1,2,3}.json + _logits.npz (실행기 v2_mmimdb_e2.py --methods moddrop)
      + 기존 E9 zf·γ8, E2 image_only·text_only. 계산 함수는 mmimdb_v2_analysis.py의 것을 그대로 가져다 씀(같은 구현).

[고정 규칙 — ModDrop 결과 열람 전, 2026-11-05]
배경(사실): SigLIP image 결손(텍스트만 남음)에서 붕괴 +0.104, 기준값 층 0.34~0.37, AUPRC(R)−AUPRC(zf) +0.085 [+0.074,+0.095]
  → 붕괴의 다수가 순위 층이고, γ8은 이를 메우지 못함(AUPRC γ8−zf −0.035 악화). ModDrop(p=0.3, 학습 중 모달 무작위 제거)은
  결손 상황을 학습에 직접 노출하므로 순위 층을 메울 수 있는지 시험한다.
공통: seed 1·2·3 paired, n=3, df=2, t=4.303, ddof=1. 용어 개선/구분 불가/악화.
  재보정은 '정확 탐색' 규칙(fit_thr_exact: 라벨별 val F1 최대 기준값, 격자 없음)만 쓴다(격자 규칙의 절단 결함은 이미 확인됨).
  ModDrop 설정은 기존 zf config 그대로 + method만 moddrop(moddrop_p 0.3, 조기종료 val full micro-F1 — 원 프로토콜과 동일).
조건별(text 결손 / image 결손 / full) 계산:
  H4a  MD − zf                       (F1@0.5)
  H4b  MD+재보정 − zf+재보정          (기준값을 똑같이 맞춘 뒤 남는 차 = 순위 수준 기여)
  H4c  AUPRC MD − zf                 (순위 지표)
  H4d  MD+재보정 − γ8+재보정, AUPRC MD − γ8   (두 처방 비교)
  메움 비율 = (AUPRC(MD) − AUPRC(zf)) / (AUPRC(R) − AUPRC(zf)) — AUPRC(R)−AUPRC(zf)의 CI 하한 > 0인 조건에서만
주 판정(SigLIP image 결손 1셀):
  H4b 개선 AND H4c 개선 AND 메움 비율 ≥ 0.5  → "ModDrop이 순위 층 붕괴를 메움"
  H4b·H4c 중 하나 이상 개선이나 위 조건 미충족 → "부분적으로 메움"
  둘 다 개선 아님                                → "메우지 못함"
  나머지 셀(SigLIP text 결손·full, CLIP 전부)은 같은 수치를 보고만 한다(판정 문구 없음).
점검: K1 MD 로짓 재계산 F1@0.5 = json(|차| ≤ 1e-6), K3 라벨 동일성(E2·E9 전 run),
  K2 기존 ModDrop 결과와 재현 대조 — 설정이 같은 기존 run만 대조(무관 키 제외 비교), 설정이 다른 run은 '대조 제외'로 표시.
한계: n=3, val은 조기종료·재보정 겸용(모든 모델 동일), ModDrop p는 0.3 하나만(탐색 없음).
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmimdb_v2_analysis import f1_at05, f1_recal_exact, auprc, ci, verdict, load_run, SEEDS, T_N3, NDF, SUF

IGN = {"out", "save_preds", "save_model", "save_logits", "workers", "causal", "smoke", "lam_b"}
REGS = (("text", "textmissing", "R_image_only"), ("image", "imagemissing", "R_text_only"), ("full", "full", None))


def cfg_diff(a, b):
    """두 config 모두에 있는 키만 비교(새 run에는 이후 추가된 인자 키가 더 있어 합집합 비교 시 기존 run이 전부 '설정 다름'이 됨)."""
    return {k: (a.get(k), b.get(k)) for k in set(a) & set(b) if k not in IGN and a.get(k) != b.get(k)}


def analyze(root, bb):
    e2d, e9d = (os.path.join(root, "results/thesis_round/V2", d) for d in ("E2_mmimdb", "E9_mmimdb"))
    names = {"R_image_only": (e2d, "image_only"), "R_text_only": (e2d, "text_only"),
             "zf": (e9d, "zerofill"), "g8": (e9d, "gamma_g8.0"), "md": (e9d, "moddrop")}
    runs, missing = {}, []
    for key, (d, m) in names.items():
        for s in SEEDS:
            r = load_run(os.path.join(d, "%s_%s_s%d" % (bb, m, s)))
            (missing.append("%s_%s_s%d" % (bb, m, s)) if r is None else runs.__setitem__((key, s), r))
    print("=" * 100); print("추가 실험 v2 — MM-IMDb [%s] E9-MD (ModDrop) | seed %s, %s, t=%.3f" % (bb, SEEDS, NDF, T_N3)); print("=" * 100)
    if missing:
        print("★ 없는 run: %s" % ", ".join(missing))
    if any(("md", s) not in runs or ("zf", s) not in runs for s in SEEDS):
        print("★ md 또는 zf run이 seed 수(%d)만큼 없음 → 분석 중단" % len(SEEDS)); return None
    ref = None
    for (k, s), (j, z) in runs.items():
        cur = (z["val_y"], z["test_y"])
        if ref is None:
            ref = cur
        elif not (np.array_equal(ref[0], cur[0]) and np.array_equal(ref[1], cur[1])):
            sys.exit("★ K3 실패: %s s%d 라벨 불일치 → 중단" % (k, s))
    print("[K3] 라벨 동일성: 통과(%d개 run)" % len(runs))
    k1 = 0.0; bad = []
    for s in SEEDS:
        j, z = runs[("md", s)]
        for ek, nk in (("full", "full"), ("text", "textmissing"), ("image", "imagemissing")):
            d = abs(f1_at05(z["test_y"], z["test_" + nk]) - j["eval"][ek]["macro"]); k1 = max(k1, d)
            if d > 1e-6:
                bad.append("s%d %s %.1e" % (s, ek, d))
    print("[K1] MD 로짓 재계산 F1 vs json: 최대 |차| %.1e → %s" % (k1, "통과" if not bad else "★경고 " + "; ".join(bad)))
    k2 = []
    for s in SEEDS:
        jn = runs[("md", s)][0]; cn, ev = jn.get("config", {}), jn["eval"]
        for f in sorted(glob.glob(os.path.join(root, "results", "*.json"))):
            try:
                r = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            c = r.get("config")
            if (not isinstance(c, dict) or c.get("mode") != "real" or c.get("backbone") != bb or c.get("method") != "moddrop"
                    or c.get("seed") != s or "text" not in r.get("eval", {})):
                continue
            dif = cfg_diff(cn, c); rel = os.path.relpath(f, root)
            if dif:
                print("[K2] md s%d vs %s: 설정 다름 → 대조 제외 %s" % (s, rel, dif)); continue
            dt, di = ev["text"]["macro"] - r["eval"]["text"]["macro"], ev["image"]["macro"] - r["eval"]["image"]["macro"]
            k2.append(dict(seed=s, orig=rel, d_text=dt, d_image=di))
            print("[K2] md s%d vs %s: text결손 차 %+.4f, image결손 차 %+.4f%s" % (
                s, rel, dt, di, "" if max(abs(dt), abs(di)) <= 1e-3 else "  ★재현 차 > 0.001"))
    if not k2:
        print("[K2] 설정이 같은 기존 ModDrop 결과 없음(대조 불가)")

    res = {"backbone": bb, "missing": missing, "K1_max": k1, "K1_bad": bad, "K2": k2, "cells": {}}
    for reg, nk, rkey in REGS:
        keys = [k for k in (rkey, "zf", "g8", "md") if k and all((k, s) in runs for s in SEEDS)]
        M = {}
        for k in keys:
            for s in SEEDS:
                z = runs[(k, s)][1]
                M[(k, s)] = {"f1": f1_at05(z["test_y"], z["test_" + nk]),
                             "f1x": f1_recal_exact(z["val_y"], z["val_" + nk], z["test_y"], z["test_" + nk]),
                             "ap": auprc(z["test_y"], z["test_" + nk])}
        D = lambda a, ma, b, mb: np.array([M[(a, s)][ma] - M[(b, s)][mb] for s in SEEDS])
        print("\n" + "-" * 100)
        print("[%s] %s" % ({"text": "text 결손", "image": "image 결손", "full": "full"}[reg],
                           "← 주 판정 셀" if (bb == "siglip" and reg == "image") else "(보고만)"))
        print("-" * 100)
        for k in keys:
            lab = {"zf": "zf", "g8": "γ8", "md": "MD"}.get(k, "R")
            A = lambda m: np.mean([M[(k, s)][m] for s in SEEDS])
            print("  %-3s F1@0.5 %.4f | F1@재보정(정확) %.4f | AUPRC %.4f" % (lab, A("f1"), A("f1x"), A("ap")))
        a, b, c = ci(D("md", "f1", "zf", "f1")), ci(D("md", "f1x", "zf", "f1x")), ci(D("md", "ap", "zf", "ap"))
        cell = {"H4a": a, "H4b": b, "H4c": c}
        print("  H4a MD − zf (F1@0.5)          %+.4f [%+.4f,%+.4f] %s" % (*a, verdict(a[1], a[2])))
        print("  H4b MD+재보정 − zf+재보정      %+.4f [%+.4f,%+.4f] %s" % (*b, verdict(b[1], b[2])))
        print("  H4c AUPRC MD − zf             %+.4f [%+.4f,%+.4f] %s" % (*c, verdict(c[1], c[2])))
        if "g8" in keys:
            d1, d2 = ci(D("md", "f1x", "g8", "f1x")), ci(D("md", "ap", "g8", "ap"))
            cell["H4d"] = {"f1x": d1, "ap": d2}
            print("  H4d MD+재보정 − γ8+재보정      %+.4f [%+.4f,%+.4f] %s | AUPRC MD − γ8 %+.4f [%+.4f,%+.4f] %s" % (
                *d1, verdict(d1[1], d1[2]), *d2, verdict(d2[1], d2[2])))
        ratio = None
        if rkey in keys:
            g = ci(D(rkey, "ap", "zf", "ap"))
            if g[1] > 0:
                ratio = float(D("md", "ap", "zf", "ap").mean() / D(rkey, "ap", "zf", "ap").mean())
                print("  순위 층(AUPRC R − zf) %+.4f [%+.4f,%+.4f] → MD 메움 비율 %.2f" % (*g, ratio))
            else:
                print("  순위 층(AUPRC R − zf) %+.4f [%+.4f,%+.4f] → 유의하지 않아 메움 비율 해당 없음" % g)
        cell["ratio"] = ratio
        if bb == "siglip" and reg == "image":
            ib, ic = b[1] > 0, c[1] > 0
            v = ("ModDrop이 순위 층 붕괴를 메움" if (ib and ic and ratio is not None and ratio >= 0.5)
                 else ("부분적으로 메움" if (ib or ic) else "메우지 못함"))
            cell["verdict"] = v
            print("  ▶ 주 판정: %s" % v)
        res["cells"][reg] = cell
    return res


def selftest():
    import tempfile, shutil
    rng0 = np.random.RandomState(0); L, NV, NT = 23, 2600, 7800; prev = np.linspace(0.02, 0.45, L)
    root = tempfile.mkdtemp(prefix="mmmd_")
    for d in ("E2_mmimdb", "E9_mmimdb"):
        os.makedirs(os.path.join(root, "results/thesis_round/V2", d))
    yv = (rng0.rand(NV, L) < prev).astype(np.float32); yt = (rng0.rand(NT, L) < prev).astype(np.float32)
    cal = lambda y, mu, rng: mu * (y * mu + rng.randn(*y.shape)) - mu ** 2 / 2 + np.log(prev / (1 - prev))
    base_cfg = {"mode": "real", "backbone": "siglip", "seed": None, "moddrop_p": 0.3, "epochs": 8, "limit": 0}

    def write(name, s, V, T, method=None, mods=None):
        arr, ev = {"val_y": yv, "test_y": yt}, {}
        for nk, ek in (("full", "full"), ("textmissing", "text"), ("imagemissing", "image")):
            arr["val_" + nk], arr["test_" + nk] = V[nk].astype(np.float32), T[nk].astype(np.float32)
            ev[ek] = {"macro": f1_at05(yt, arr["test_" + nk])}
        for k, v in (mods or {}).items():
            ev[k]["macro"] += v
        stem = os.path.join(root, "results/thesis_round/V2", "E2_mmimdb" if "only" in name else "E9_mmimdb", "siglip_%s_s%d" % (name, s))
        np.savez_compressed(stem + "_logits.npz", **arr)
        cfg = dict(base_cfg, seed=s, method=method or name, out=stem + ".json", save_logits=stem + "_logits.npz")
        json.dump({"config": cfg, "eval": ev}, open(stem + ".json", "w"))
        return cfg, ev

    for s in SEEDS:
        rng = np.random.RandomState(200 + s)
        G = {nm: (cal(yv, mu, rng), cal(yt, mu, rng)) for nm, mu in (("good", 2.0), ("mid", 1.2), ("weak", 0.8))}
        full = (G["good"][0] + G["mid"][0], G["good"][1] + G["mid"][1])
        pick = lambda tm, im, i: {"full": full[i], "textmissing": tm[i], "imagemissing": im[i]}
        write("image_only", s, pick(G["good"], G["weak"], 0), pick(G["good"], G["weak"], 1))
        write("text_only", s, pick(G["weak"], G["good"], 0), pick(G["weak"], G["good"], 1))
        # image 결손: R = good, zf = mid(순위 붕괴), γ8 = weak, MD = good − 1(순위 회복 + 기준값 어긋남) → 양성 대조
        # text 결손: MD = zf 와 완전히 같은 로짓 → 음성 대조(차 정확히 0)
        zfV = {"full": full[0], "textmissing": G["mid"][0], "imagemissing": G["mid"][0]}
        zfT = {"full": full[1], "textmissing": G["mid"][1], "imagemissing": G["mid"][1]}
        write("zerofill", s, zfV, zfT)
        write("gamma_g8.0", s, dict(zfV, imagemissing=G["weak"][0]), dict(zfT, imagemissing=G["weak"][1]), method="gamma")
        cfg, ev = write("moddrop", s, dict(zfV, imagemissing=G["good"][0] - 1), dict(zfT, imagemissing=G["good"][1] - 1),
                        mods={"full": 0.01} if s == 2 else None)
        # 기존 결과: 같은 설정(s3만 일부러 text +0.02) + p가 다른 run(대조 제외되어야 함)
        o = json.loads(json.dumps(ev)); o["text"]["macro"] += (0.02 if s == 3 else 0.0)
        oc = dict(cfg, out="old.json", save_logits=None, workers=2); oc.pop("limit")   # 이전 버전엔 limit 키 없음
        json.dump({"config": oc, "eval": o},
                  open(os.path.join(root, "results", "old_md_s%d.json" % s), "w"))
        json.dump({"config": dict(cfg, moddrop_p=0.5), "eval": ev}, open(os.path.join(root, "results", "old_md_p05_s%d.json" % s), "w"))
    print("기대: K1 경고 1건(s2 full), K2 재현 경고 1건(s3), p0.5 run 3개 '대조 제외'")
    print("      image: H4b·H4c 개선, 메움 비율 ≈ 1 → 'ModDrop이 순위 층 붕괴를 메움' | text: H4a·b·c 정확히 0")
    r = analyze(root, "siglip"); shutil.rmtree(root, ignore_errors=True)
    im, tx = r["cells"]["image"], r["cells"]["text"]
    ok = [len(r["K1_bad"]) == 1, sum(abs(x["d_text"]) > 1e-3 for x in r["K2"]) == 1, len(r["K2"]) == len(SEEDS),
          im["H4b"][1] > 0, im["H4c"][1] > 0, im["ratio"] is not None and 0.9 < im["ratio"] < 1.1,
          im["verdict"] == "ModDrop이 순위 층 붕괴를 메움",
          all(abs(tx[h][0]) < 1e-12 for h in ("H4a", "H4b", "H4c")), im["H4d"]["ap"][1] > 0]
    print("\n자체 시험 판정:", "모두 통과" if all(ok) else "★실패 %s" % ok)
    return all(ok)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    BB = sys.argv[1]
    r = analyze(".", BB)
    if r is not None:
        out = "results/thesis_round/V2/mmimdb_v2_moddrop_%s%s.json" % (BB, SUF)
        json.dump(r, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("\n결과 저장:", out)
    print("=== ALL DONE (MM-IMDb E9-MD %s) ===" % BB)
