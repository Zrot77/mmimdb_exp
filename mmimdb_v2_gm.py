"""추가 실험 v2 — MM-IMDb E9-GM: γ8+ModDrop 결합이 두 방향의 순위 층을 함께 다루는가
  python mmimdb_v2_gm.py siglip        # 주 대상
  python mmimdb_v2_gm.py clip          # 같은 규칙
  python mmimdb_v2_gm.py --selftest    # 가짜 로짓 대조 시험(양성·음성)
입력: results/thesis_round/V2/E9_mmimdb/{BB}_gamma_moddrop_g8.0_s{1,2,3}.json + _logits.npz
      (실행기: v2_mmimdb_e2.py BB --methods gamma_moddrop --gamma 8.0 --outdir results/thesis_round/V2/E9_mmimdb)
      + 기존 E9 zf·γ8·MD, E2 image_only·text_only. 계산 함수는 mmimdb_v2_analysis.py 것을 그대로 씀.

[고정 규칙 — GM 결과 열람 전, 2026-11-06]
배경(사실, E9·E9-MD, 두 백본 공통):
  text 결손(이미지 경로가 남음): γ8은 기준값+순위 층을 메우고(SigLIP AUPRC +0.043), ModDrop은 기준값만(대칭 재보정 후 차 0과 구분 불가)
  image 결손(텍스트 경로가 남음): ModDrop은 순위 층의 57~63%를 메우고, γ8은 못 메움(SigLIP AUPRC −0.035)
  → 두 처방이 서로 다른 경로를 고침. 결합(GM)이 두 방향을 함께 다루는지 시험
주의: "γ+ModDrop 시너지"는 text 결손 F1@0.5 기준으로 이미 기각됨(A1·A4). 이 실험은 그 판정을 뒤집지 않으며,
  다른 질문(방향별 순위 층 포괄)을 새로 묻는 것이다. 결과가 좋아도 '시너지'라는 표현은 쓰지 않는다.
공통: seed 1·2·3 paired, n=3, df=2, t=4.303, ddof=1, 용어 개선/구분 불가/악화. 재보정은 정확 탐색 규칙만.
  GM 설정 = 기존 zf config + method gamma_moddrop + γ=8 (moddrop_p 0.3, 조기종료 val full micro-F1).
판정(백본별, 주 판정은 SigLIP, CLIP은 같은 규칙으로 보고):
  H5-t  text 결손: GM+재보정 − γ8+재보정   (text 결손에서 잘 듣는 단일 처방 = γ8, 위 배경에서 정함)
  H5-i  image 결손: GM+재보정 − MD+재보정  (image 결손에서 잘 듣는 단일 처방 = MD)
  각각 '악화'가 아니면 "유지", '악화'면 "손실"
  종합: 둘 다 유지 → "GM이 두 방향을 함께 다룸" / 하나만 손실 → "한 방향만 유지" / 둘 다 손실 → "결합 시 두 방향 모두 손실"
보고(판정 없음): 각 조건의 F1@0.5·F1@재보정·AUPRC(zf·γ8·MD·GM·R), GM − zf (F1@0.5·대칭 재보정·AUPRC),
  같은 비교의 AUPRC판(H5-t·H5-i), full 조건 GM+재보정 − zf+재보정·γ8+재보정(결합 비용).
점검: K1(GM 로짓 재계산 = json), K3(라벨 동일성), K2(설정이 같은 기존 gamma_moddrop γ8 결과와 재현 대조,
  results/ 아래 전체 검색하되 이번 V2 산출물은 제외, 공통 키만 비교).
한계: n=3, ModDrop p·γ 탐색 없음, 처방별 '잘 듣는 방향'은 앞선 결과로 정한 것(사후 지식).
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmimdb_v2_analysis import f1_at05, f1_recal_exact, auprc, ci, verdict, load_run, SEEDS, T_N3

IGN = {"out", "save_preds", "save_model", "save_logits", "workers", "causal", "smoke", "lam_b"}
REGS = (("text", "textmissing", "R_image_only"), ("image", "imagemissing", "R_text_only"), ("full", "full", None))
LAB = {"zf": "zf", "g8": "γ8", "md": "MD", "gm": "GM", "R_image_only": "R", "R_text_only": "R"}


def cfg_diff(a, b):
    return {k: (a.get(k), b.get(k)) for k in set(a) & set(b) if k not in IGN and a.get(k) != b.get(k)}


def analyze(root, bb):
    e2d, e9d = (os.path.join(root, "results/thesis_round/V2", d) for d in ("E2_mmimdb", "E9_mmimdb"))
    names = {"R_image_only": (e2d, "image_only"), "R_text_only": (e2d, "text_only"), "zf": (e9d, "zerofill"),
             "g8": (e9d, "gamma_g8.0"), "md": (e9d, "moddrop"), "gm": (e9d, "gamma_moddrop_g8.0")}
    runs, missing = {}, []
    for key, (d, m) in names.items():
        for s in SEEDS:
            r = load_run(os.path.join(d, "%s_%s_s%d" % (bb, m, s)))
            (missing.append("%s_%s_s%d" % (bb, m, s)) if r is None else runs.__setitem__((key, s), r))
    print("=" * 100); print("추가 실험 v2 — MM-IMDb [%s] E9-GM (γ8+ModDrop) | seed %s, n=3, df=2, t=%.3f" % (bb, SEEDS, T_N3)); print("=" * 100)
    if missing:
        print("★ 없는 run: %s" % ", ".join(missing))
    need = ("zf", "g8", "md", "gm", "R_image_only", "R_text_only")
    if any((k, s) not in runs for k in need for s in SEEDS):
        print("★ 필요한 run이 다 있지 않음 → 분석 중단"); return None
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
        j, z = runs[("gm", s)]
        for ek, nk in (("full", "full"), ("text", "textmissing"), ("image", "imagemissing")):
            d = abs(f1_at05(z["test_y"], z["test_" + nk]) - j["eval"][ek]["macro"]); k1 = max(k1, d)
            if d > 1e-6:
                bad.append("s%d %s %.1e" % (s, ek, d))
    print("[K1] GM 로짓 재계산 F1 vs json: 최대 |차| %.1e → %s" % (k1, "통과" if not bad else "★경고 " + "; ".join(bad)))
    v2dir = os.path.normpath(os.path.join(root, "results/thesis_round/V2"))
    k2 = []
    for s in SEEDS:
        jn = runs[("gm", s)][0]; cn, ev = jn.get("config", {}), jn["eval"]
        for f in sorted(glob.glob(os.path.join(root, "results", "**", "*.json"), recursive=True)):
            if os.path.normpath(f).startswith(v2dir):
                continue
            try:
                r = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            c = r.get("config") if isinstance(r, dict) else None
            if (not isinstance(c, dict) or c.get("mode") != "real" or c.get("backbone") != bb or c.get("method") != "gamma_moddrop"
                    or c.get("seed") != s or "text" not in r.get("eval", {}) or abs(float(c.get("gamma", -1)) - 8.0) > 1e-9):
                continue
            dif = cfg_diff(cn, c); rel = os.path.relpath(f, root)
            if dif:
                print("[K2] gm s%d vs %s: 설정 다름 → 대조 제외 %s" % (s, rel, dif)); continue
            dt, di = ev["text"]["macro"] - r["eval"]["text"]["macro"], ev["image"]["macro"] - r["eval"]["image"]["macro"]
            k2.append(dict(seed=s, orig=rel, d_text=dt, d_image=di))
            print("[K2] gm s%d vs %s: text결손 차 %+.4f, image결손 차 %+.4f%s" % (
                s, rel, dt, di, "" if max(abs(dt), abs(di)) <= 1e-3 else "  ★재현 차 > 0.001"))
    if not k2:
        print("[K2] 설정이 같은 기존 γ8+ModDrop 결과 없음(대조 불가)")

    res = {"backbone": bb, "K1_max": k1, "K1_bad": bad, "K2": k2, "cells": {}}
    keep = {}
    for reg, nk, rkey in REGS:
        keys = [k for k in (rkey, "zf", "g8", "md", "gm") if k]
        M = {}
        for k in keys:
            for s in SEEDS:
                z = runs[(k, s)][1]
                M[(k, s)] = {"f1": f1_at05(z["test_y"], z["test_" + nk]),
                             "f1x": f1_recal_exact(z["val_y"], z["val_" + nk], z["test_y"], z["test_" + nk]),
                             "ap": auprc(z["test_y"], z["test_" + nk])}
        D = lambda a, ma, b, mb: ci(np.array([M[(a, s)][ma] - M[(b, s)][mb] for s in SEEDS]))
        print("\n" + "-" * 100); print("[%s]" % {"text": "text 결손", "image": "image 결손", "full": "full"}[reg]); print("-" * 100)
        for k in keys:
            A = lambda m: np.mean([M[(k, s)][m] for s in SEEDS])
            print("  %-3s F1@0.5 %.4f | F1@재보정(정확) %.4f | AUPRC %.4f" % (LAB[k], A("f1"), A("f1x"), A("ap")))
        cell = {}
        for nm, a, ma, b, mb in (("GM − zf (F1@0.5)", "gm", "f1", "zf", "f1"), ("GM+재보정 − zf+재보정", "gm", "f1x", "zf", "f1x"),
                                 ("AUPRC GM − zf", "gm", "ap", "zf", "ap"), ("GM+재보정 − γ8+재보정", "gm", "f1x", "g8", "f1x"),
                                 ("AUPRC GM − γ8", "gm", "ap", "g8", "ap"), ("GM+재보정 − MD+재보정", "gm", "f1x", "md", "f1x"),
                                 ("AUPRC GM − MD", "gm", "ap", "md", "ap")):
            c_ = D(a, ma, b, mb); cell[nm] = c_
            print("  %-24s %+.4f [%+.4f,%+.4f] %s" % (nm, *c_, verdict(c_[1], c_[2])))
        if rkey:
            g = D(rkey, "ap", "zf", "ap")
            if g[1] > 0:
                rr = float(np.mean([M[("gm", s)]["ap"] - M[("zf", s)]["ap"] for s in SEEDS]) / g[0])
                print("  순위 층(AUPRC R − zf) %+.4f → GM 메움 비율 %.2f" % (g[0], rr)); cell["ratio"] = rr
        res["cells"][reg] = cell; keep[reg] = cell
    ht, hi = keep["text"]["GM+재보정 − γ8+재보정"], keep["image"]["GM+재보정 − MD+재보정"]
    vt = "손실" if ht[2] < 0 else "유지"; vi = "손실" if hi[2] < 0 else "유지"
    tot = {("유지", "유지"): "GM이 두 방향을 함께 다룸", ("손실", "손실"): "결합 시 두 방향 모두 손실"}.get((vt, vi), "한 방향만 유지")
    res.update(H5t=vt, H5i=vi, verdict=tot)
    print("\n" + "=" * 100)
    print("H5-t text 결손 GM+재보정 − γ8+재보정 %+.4f [%+.4f,%+.4f] → %s" % (*ht, vt))
    print("H5-i image 결손 GM+재보정 − MD+재보정 %+.4f [%+.4f,%+.4f] → %s" % (*hi, vi))
    print("▶ %s 판정: %s" % ("주" if bb == "siglip" else "보고(같은 규칙)", tot))
    return res


def selftest():
    import tempfile, shutil
    out = []
    for case in ("pos", "neg"):
        rng0 = np.random.RandomState(0); L, NV, NT = 23, 2600, 7800; prev = np.linspace(0.02, 0.45, L)
        root = tempfile.mkdtemp(prefix="mmgm_")
        for d in ("E2_mmimdb", "E9_mmimdb"):
            os.makedirs(os.path.join(root, "results/thesis_round/V2", d))
        os.makedirs(os.path.join(root, "results/thesis_round/A1"))
        yv = (rng0.rand(NV, L) < prev).astype(np.float32); yt = (rng0.rand(NT, L) < prev).astype(np.float32)
        cal = lambda y, mu, rng: mu * (y * mu + rng.randn(*y.shape)) - mu ** 2 / 2 + np.log(prev / (1 - prev))

        def write(name, s, V, T, method):
            arr, ev = {"val_y": yv, "test_y": yt}, {}
            for nk, ek in (("full", "full"), ("textmissing", "text"), ("imagemissing", "image")):
                arr["val_" + nk], arr["test_" + nk] = V[nk].astype(np.float32), T[nk].astype(np.float32)
                ev[ek] = {"macro": f1_at05(yt, arr["test_" + nk])}
            stem = os.path.join(root, "results/thesis_round/V2", "E2_mmimdb" if "only" in name else "E9_mmimdb", "siglip_%s_s%d" % (name, s))
            np.savez_compressed(stem + "_logits.npz", **arr)
            cfg = {"mode": "real", "backbone": "siglip", "seed": s, "method": method, "gamma": 8.0, "moddrop_p": 0.3, "out": stem, "limit": 0}
            json.dump({"config": cfg, "eval": ev}, open(stem + ".json", "w"))
            return cfg, ev

        for s in SEEDS:
            rng = np.random.RandomState(300 + s)
            G = {nm: (cal(yv, mu, rng), cal(yt, mu, rng)) for nm, mu in (("good", 2.0), ("mid", 1.2), ("weak", 0.8))}
            full = (G["good"][0] + G["mid"][0], G["good"][1] + G["mid"][1])
            P = lambda tm, im, i, sh_t=0.0, sh_i=0.0: {"full": full[i], "textmissing": tm[i] + sh_t, "imagemissing": im[i] + sh_i}
            write("image_only", s, P(G["good"], G["weak"], 0), P(G["good"], G["weak"], 1), "image_only")
            write("text_only", s, P(G["weak"], G["good"], 0), P(G["weak"], G["good"], 1), "text_only")
            write("zerofill", s, P(G["mid"], G["mid"], 0, -2), P(G["mid"], G["mid"], 1, -2), "zerofill")
            write("gamma_g8.0", s, P(G["good"], G["weak"], 0), P(G["good"], G["weak"], 1), "gamma")
            write("moddrop", s, P(G["mid"], G["good"], 0, 0, -1), P(G["mid"], G["good"], 1, 0, -1), "moddrop")
            im = G["good"] if case == "pos" else G["mid"]          # 음성: image 방향 순위가 zf 수준으로 떨어진 GM
            cfg, ev = write("gamma_moddrop_g8.0", s, P(G["good"], im, 0, 0, -1), P(G["good"], im, 1, 0, -1), "gamma_moddrop")
            json.dump({"config": dict(cfg, out="a1.json", workers=2), "eval": ev},
                      open(os.path.join(root, "results/thesis_round/A1", "a1_gm_s%d.json" % s), "w"))
        r = analyze(root, "siglip"); shutil.rmtree(root, ignore_errors=True)
        exp = "GM이 두 방향을 함께 다룸" if case == "pos" else "한 방향만 유지"
        out += [r["verdict"] == exp, len(r["K2"]) == 3 and all(abs(x["d_text"]) < 1e-12 for x in r["K2"])]
        print("\n[자체 시험 %s] 기대 '%s' → 얻음 '%s'" % (case, exp, r["verdict"]))
    print("\n자체 시험 판정:", "모두 통과" if all(out) else "★실패 %s" % out)
    return all(out)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    BB = sys.argv[1]
    r = analyze(".", BB)
    if r is not None:
        out = "results/thesis_round/V2/mmimdb_v2_gm_%s.json" % BB
        json.dump(r, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("\n결과 저장:", out)
    print("=== ALL DONE (MM-IMDb E9-GM %s) ===" % BB)
