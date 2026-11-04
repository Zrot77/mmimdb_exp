"""추가 실험 v2 — MM-IMDb 분석 (E2 붕괴 정의 통일 / E9 기준값 재보정)
  python mmimdb_v2_analysis.py clip              # 서버 ~/Yechan/mmimdb_exp 에서 (conda mmimdb)
  python mmimdb_v2_analysis.py --selftest        # 가짜 로짓 대조 시험(양성·음성·순위붕괴·점검 탐지)
입력(실행기 v2_mmimdb_e2.py 산출물):
  E2: results/thesis_round/V2/E2_mmimdb/{BB}_{image_only,text_only}_s{1,2,3}.json + _logits.npz
  E9: results/thesis_round/V2/E9_mmimdb/{BB}_{zerofill,gamma_g8.0}_s{1,2,3}.json + _logits.npz
  재현 대조: results/*.json 중 같은 backbone·seed의 기존 zerofill / gamma(γ=8.0) 결과
없는 파일의 셀은 대체·추정하지 않고 제외하며 목록을 출력한다(v2 §0).

[고정 규칙 — 결과 열람 전, 2026-11-04]
공통: seed 1·2·3, paired 95%CI n=3, df=2, t=4.303, ddof=1. 판정 용어 개선/구분 불가/악화. 붕괴 임계값 macro-F1 0.05.
  주 결손 = text 결손(이미지만 남음), image 결손은 함께 보고. 지표 계산은 test.
지표:
  F1@0.5   = macro-F1(기준값 0.5) — 기존 프로토콜(utils.f1_all과 동일: sklearn macro, zero_division=0)
  F1@재보정 = 각 모델·결손 조건마다 val의 같은 결손 조건 출력으로 라벨별 기준값 t∈{0.01,…,0.99}를 val F1 최대로 고르고
              (동률이면 0.5에 가장 가까운 값, val 양성 0개 라벨은 0.5) test에 적용한 macro-F1.
              zf·γ8·천장 R에 똑같이 적용(대칭 비교).
  AUPRC    = 라벨별 average precision의 평균(test 양성 ≥1 라벨). 순위만 보는 지표 — 기준값과 무관.
E2 (ADDENDUM_v2 §E2):
  R 붕괴 = 천장 R(F1@0.5) − zf(F1@0.5).  R: text 결손 → image_only, image 결손 → text_only.
  L 붕괴 = 기존 raw suff(문서값, A0 canonical_numbers.csv; seed별 원자료 없음 → 평균만) − zf 평균.
  셀별 '붕괴 있음/없음'(≥0.05)이 두 정의에서 같으면 "정의 간 일치", 다르면 "불일치".
E9 (MM-IMDb 재보정, zf·γ8은 로짓 저장을 위해 같은 설정으로 재학습한 run):
  H1  zf+재보정 − zf (F1)                                   → 개선/구분 불가/악화
  H2  R − zf ≥ 0.05(붕괴 있음)일 때: 잔여 붕괴 = R+재보정 − zf+재보정 (CI).
      기준값 층 비율 = 1 − 평균 잔여/평균 원붕괴. 순위 층 근거 = AUPRC(R) − AUPRC(zf) (CI).
  H3  γ8 − (zf+재보정), γ8+재보정 − (zf+재보정)              → 처방이 재보정 이상 / 구분 불가 / 재보정 우위
  참고 γ8 − zf (F1@0.5), 각 모델 full F1.
점검(판정 전):
  K1 로짓으로 다시 계산한 F1@0.5(full/text/image) = 결과 json 값 (|차| ≤ 1e-6; 어긋나면 경고)
  K2 같은 seed의 기존 결과 json과의 재현 차(zf·γ8, text/image 결손). |차| > 0.001이면 경고(판정은 진행, 보고에 명시)
  K3 모든 로짓 파일의 val·test 라벨이 서로 같음(다르면 중단)
[사후 추가 — CLIP 결과 열람 후 추가, 2026-11-04. 위 고정 규칙의 판정은 바꾸지 않음]
F  full(결손 없음) 조건 기술 통계: zf·γ8의 F1@0.5, F1@재보정(val full 출력으로 같은 방식 기준값), AUPRC.
   γ8 − zf (F1@0.5), γ8+재보정 − zf+재보정, AUPRC γ8 − zf 의 CI와 용어(개선/구분 불가/악화)를 보고하되 '사후 기술'로 표시.
   목적: 기존 'γ8이 full에서도 높다(.509 vs .462)'가 기준값 효과인지 순위 효과인지 구분.
[수정 규칙 — 사후 발견 결함, 2026-11-04] 고정 규칙의 확률 격자 t∈{0.01,…,0.99}는 (1) 최적 기준값이 0.01 밖이면 잘리고
   (2) 낮은 확률 구간 해상도가 거칠어(0.02→0.03 = 로짓 0.4) 로짓이 크게 이동한 모델의 재보정을 과소평가한다.
   음성 대조(순위 동일, 로짓만 −2 이동한 두 모델)에서 격자 규칙은 대칭 재보정 차 +0.009 '개선'을 거짓으로 냈다.
   → 라벨별 val 로짓 값 자체에서 F1 최대 기준값을 정확 탐색(fit_thr_exact; 동률은 로짓 0에 가장 가까운 값)하는 규칙으로
   H1~H3를 반복해 나란히 보고한다. 고정 규칙 결과는 삭제하지 않는다. 각 모델의 '격자 끝 라벨 수'(절단 진단)도 출력.
알려진 한계: val은 조기종료에도 쓰였으므로 재보정 이득은 약간 낙관적일 수 있음(모든 모델 동일 조건). n=3이라 검정력 낮음.
"""
import glob, json, os, sys
import numpy as np

T_N3, THR = 4.303, 0.05
SEEDS = [1, 2, 3]
GRID = np.round(np.arange(1, 100) / 100.0, 2)
# L(선형 천장) 문서값: A0 canonical_numbers.csv 'raw_suff_CLIPimg_SigLIPimg_etc' (CLIP img .407 / CLIP txt .488 / SigLIP img .545)
# SigLIP 파이프라인의 텍스트는 CLIP 텍스트 앵커 → 텍스트 raw suff는 CLIP txt 값을 공유
L_RAW = {"clip": {"text": 0.407, "image": 0.488}, "siglip": {"text": 0.545, "image": 0.488}}
# 결손 조건 → (npz 키 접미사, json eval 키, 천장 R 방법)
REG = {"text": ("textmissing", "text", "image_only"), "image": ("imagemissing", "image", "text_only")}


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def macro_f1(y, pred):
    from sklearn.metrics import f1_score
    return float(f1_score(y.astype(int), pred.astype(int), average="macro", zero_division=0))


def f1_at05(y, logit):  # utils.f1_all과 같은 방식: float32 sigmoid > 0.5
    lg = np.asarray(logit, np.float32)
    return macro_f1(y, (1.0 / (1.0 + np.exp(-lg)) > np.float32(0.5)))


def fit_thr(yv, pv):
    """라벨별 val F1 최대 기준값. 동률은 0.5에 가장 가까운 값, val 양성 0개는 0.5."""
    yv = yv.astype(bool); L = yv.shape[1]; thr = np.full(L, 0.5)
    for j in range(L):
        pos = yv[:, j]
        if pos.sum() == 0:
            continue
        P = pv[:, j][:, None] >= GRID[None, :]
        tp = (P & pos[:, None]).sum(0); fp = (P & ~pos[:, None]).sum(0); fn = pos.sum() - tp
        f = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
        best = f.max(); cand = np.where(f >= best - 1e-12)[0]
        thr[j] = GRID[cand[np.argmin(np.abs(GRID[cand] - 0.5))]]
    return thr


def f1_recal(yv, lv, yt, lt):
    thr = fit_thr(yv, sig(lv))
    return macro_f1(yt, sig(lt) >= thr[None, :])


def n_edge(yv, lv):
    """격자 기준값이 양 끝(0.01 또는 0.99)에 걸린 라벨 수 — 격자 절단 진단."""
    t = fit_thr(yv, sig(lv)); return int(((t <= 0.01 + 1e-9) | (t >= 0.99 - 1e-9)).sum())


def fit_thr_exact(yv, lv):
    """[수정 규칙] 라벨별 val F1 최대 기준값을 로짓 값 자체에서 정확 탐색(격자 없음 → 절단·해상도 문제 없음).
    동률 블록 끝에서만 자르고, F1 동률이면 로짓 0(=확률 0.5)에 가장 가까운 값. val 양성 0개 라벨은 로짓 0."""
    yv = yv.astype(bool); L = yv.shape[1]; thr = np.zeros(L)
    for j in range(L):
        y, sc = yv[:, j], np.asarray(lv[:, j], np.float64); P = y.sum()
        if P == 0:
            continue
        o = np.argsort(-sc, kind="mergesort"); ss, yy = sc[o], y[o]
        tp = np.cumsum(yy); k = np.arange(1, len(ss) + 1); f = 2 * tp / (k + P)
        cut = np.r_[ss[:-1] > ss[1:], True]
        idx = np.where(cut)[0]; fb = f[idx]; cand = idx[fb >= fb.max() - 1e-12]
        thr[j] = ss[cand[np.argmin(np.abs(ss[cand]))]]
    return thr


def f1_recal_exact(yv, lv, yt, lt):
    thr = fit_thr_exact(yv, lv)
    return macro_f1(yt, np.asarray(lt, np.float64) >= thr[None, :])


def auprc(y, logit):
    from sklearn.metrics import average_precision_score
    p = sig(logit); keep = [j for j in range(y.shape[1]) if y[:, j].sum() > 0]
    return float(np.mean([average_precision_score(y[:, j], p[:, j]) for j in keep]))


def ci(d):
    d = np.asarray(d, float); m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
    return float(m), float(m - T_N3 * se), float(m + T_N3 * se)


def verdict(lo, hi):
    return "개선" if lo > 0 else ("악화" if hi < 0 else "구분 불가")


def load_run(path_stem):
    j, z = path_stem + ".json", path_stem + "_logits.npz"
    if not (os.path.exists(j) and os.path.exists(z)):
        return None
    return json.load(open(j, encoding="utf-8")), dict(np.load(z))


def originals(root, bb, method, seed, gamma=None):
    out = []
    for f in sorted(glob.glob(os.path.join(root, "results", "*.json"))):
        try:
            r = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        c = r.get("config")
        if (not isinstance(c, dict) or c.get("mode") != "real" or c.get("backbone") != bb or c.get("method") != method
                or c.get("seed") != seed or c.get("limit") or "text" not in r.get("eval", {})):
            continue
        if bb == "siglip" and os.path.basename(str(c.get("siglip_name"))) != "siglip-so400m-patch14-384":
            continue
        if gamma is not None and abs(float(c.get("gamma", -1)) - gamma) > 1e-9:
            continue
        out.append((os.path.relpath(f, root), r["eval"]))
    return out


def analyze(root, bb):
    e2d = os.path.join(root, "results/thesis_round/V2/E2_mmimdb")
    e9d = os.path.join(root, "results/thesis_round/V2/E9_mmimdb")
    names = {"R_image_only": (e2d, "image_only"), "R_text_only": (e2d, "text_only"),
             "zf": (e9d, "zerofill"), "g8": (e9d, "gamma_g8.0")}
    runs, missing = {}, []
    for key, (d, m) in names.items():
        for s in SEEDS:
            r = load_run(os.path.join(d, "%s_%s_s%d" % (bb, m, s)))
            if r is None:
                missing.append("%s_%s_s%d" % (bb, m, s))
            else:
                runs[(key, s)] = r
    print("=" * 100); print("추가 실험 v2 — MM-IMDb [%s] E2·E9 분석 | seed %s, n=3, df=2, t=%.3f" % (bb, SEEDS, T_N3)); print("=" * 100)
    if missing:
        print("★ 없는 run(해당 셀 제외): %s" % ", ".join(missing))

    # K3 라벨 동일성
    ref = None
    for (key, s), (j, z) in runs.items():
        cur = (z["val_y"], z["test_y"])
        if ref is None:
            ref = cur
        elif not (np.array_equal(ref[0], cur[0]) and np.array_equal(ref[1], cur[1])):
            sys.exit("★ K3 실패: %s s%d의 라벨이 다른 run과 다름 → 중단" % (key, s))
    print("[K3] 라벨 동일성: %s" % ("통과(%d개 run)" % len(runs) if runs else "run 없음"))
    if not runs:
        return None

    # K1 json 대조
    k1 = 0.0; k1_bad = []
    for (key, s), (j, z) in runs.items():
        for reg_json, nk in (("full", "full"), ("text", "textmissing"), ("image", "imagemissing")):
            d = abs(f1_at05(z["test_y"], z["test_" + nk]) - j["eval"][reg_json]["macro"])
            k1 = max(k1, d)
            if d > 1e-6:
                k1_bad.append("%s s%d %s %.2e" % (key, s, reg_json, d))
    print("[K1] 로짓 재계산 F1 vs json: 최대 |차| %.1e → %s" % (k1, "통과" if not k1_bad else "★경고 " + "; ".join(k1_bad)))

    # K2 재현 대조
    k2_rows = []
    for key, meth, g in (("zf", "zerofill", None), ("g8", "gamma", 8.0)):
        for s in SEEDS:
            if (key, s) not in runs:
                continue
            ev = runs[(key, s)][0]["eval"]
            for f, oe in originals(root, bb, meth, s, g):
                dt, di = ev["text"]["macro"] - oe["text"]["macro"], ev["image"]["macro"] - oe["image"]["macro"]
                k2_rows.append((key, s, f, dt, di))
    for key, s, f, dt, di in k2_rows:
        print("[K2] %-3s s%d vs %s: text결손 차 %+.4f, image결손 차 %+.4f%s" % (
            key, s, f, dt, di, "" if max(abs(dt), abs(di)) <= 1e-3 else "  ★재현 차 > 0.001"))
    if not k2_rows:
        print("[K2] 대조할 기존 결과 없음")

    res = {"backbone": bb, "missing": missing, "K1_max": k1, "K1_bad": k1_bad,
           "K2": [dict(model=a, seed=b, orig=c, d_text=d, d_image=e) for a, b, c, d, e in k2_rows], "cells": {}}
    for reg, (nk, ek, rmeth) in REG.items():
        rkey = "R_" + rmeth
        avail = {k: [s for s in SEEDS if (k, s) in runs] for k in (rkey, "zf", "g8")}
        print("\n" + "-" * 100); print("[%s 결손] (%s) 천장 R = %s" % (reg, "주" if reg == "text" else "함께 보고", rmeth)); print("-" * 100)
        M = {}
        for k in (rkey, "zf", "g8"):
            for s in avail[k]:
                j, z = runs[(k, s)]
                yv, yt = z["val_y"], z["test_y"]
                M[(k, s)] = {"f1": f1_at05(yt, z["test_" + nk]),
                             "f1r": f1_recal(yv, z["val_" + nk], yt, z["test_" + nk]),
                             "f1x": f1_recal_exact(yv, z["val_" + nk], yt, z["test_" + nk]),
                             "edge": n_edge(yv, z["val_" + nk]),
                             "ap": auprc(yt, z["test_" + nk]), "full": f1_at05(yt, z["test_full"])}
        for k, lab in ((rkey, "R"), ("zf", "zf"), ("g8", "γ8")):
            if avail[k]:
                A = lambda m: np.mean([M[(k, s)][m] for s in avail[k]])
                print("  %-3s (n=%d)  F1@0.5 %.4f | F1@재보정 %.4f | F1@재보정(정확) %.4f | AUPRC %.4f | full F1 %s | 격자 끝 라벨 %s" % (
                    lab, len(avail[k]), A("f1"), A("f1r"), A("f1x"), A("ap"), "-" if k == rkey else "%.4f" % A("full"),
                    "/".join(str(M[(k, s)]["edge"]) for s in avail[k])))
        cell = {"means": {k: {m: float(np.mean([M[(k, s)][m] for s in avail[k]])) for m in ("f1", "f1r", "f1x", "ap", "full")}
                          for k in (rkey, "zf", "g8") if avail[k]}}
        both = lambda a, b: [s for s in SEEDS if (a, s) in M and (b, s) in M]
        D = lambda a, ma, b, mb, ss: np.array([M[(a, s)][ma] - M[(b, s)][mb] for s in ss])

        # E2
        ss = both(rkey, "zf")
        if len(ss) == 3:
            rc = D(rkey, "f1", "zf", "f1", ss); rcm = float(rc.mean())
            lc = L_RAW[bb][reg] - float(np.mean([M[("zf", s)]["f1"] for s in ss]))
            hr, hl = rcm >= THR, lc >= THR
            print("  E2  R 붕괴 %+.4f [%+.4f,%+.4f] → %s | L 붕괴 %+.4f (raw suff 문서값 %.3f) → %s | %s" % (
                *ci(rc), "있음" if hr else "없음", lc, L_RAW[bb][reg], "있음" if hl else "없음",
                "정의 간 일치" if hr == hl else "불일치"))
            cell["E2"] = {"R_collapse": ci(rc), "L_collapse": lc, "R_has": hr, "L_has": hl, "agree": hr == hl}
        else:
            print("  E2  해당 없음(R 또는 zf run 부족)")
        # E9
        zs = [s for s in SEEDS if ("zf", s) in M]
        if len(zs) == 3:
            h1 = ci(D("zf", "f1r", "zf", "f1", zs))
            print("  H1  zf+재보정 − zf %+.4f [%+.4f,%+.4f] → %s" % (*h1, verdict(h1[1], h1[2])))
            cell["H1"] = h1
            if len(ss) == 3:
                raw = D(rkey, "f1", "zf", "f1", ss); resid = D(rkey, "f1r", "zf", "f1r", ss); apg = D(rkey, "ap", "zf", "ap", ss)
                if raw.mean() >= THR:
                    share = 1 - resid.mean() / raw.mean()
                    print("  H2  원붕괴 %+.4f | 잔여 붕괴(R+재보정 − zf+재보정) %+.4f [%+.4f,%+.4f] | 기준값 층 비율 %.2f | 순위 층 AUPRC(R)−AUPRC(zf) %+.4f [%+.4f,%+.4f]" % (
                        raw.mean(), *ci(resid), share, *ci(apg)))
                    cell["H2"] = {"raw": float(raw.mean()), "resid": ci(resid), "thr_share": float(share), "ap_gap": ci(apg)}
                else:
                    print("  H2  해당 없음(원붕괴 %+.4f < %.2f) | 참고 AUPRC(R)−AUPRC(zf) %+.4f [%+.4f,%+.4f]" % (raw.mean(), THR, *ci(apg)))
                    cell["H2"] = None
            gs = both("g8", "zf")
            lab = {"개선": "처방이 재보정 이상", "구분 불가": "재보정과 구분 불가", "악화": "재보정 우위"}
            if len(gs) == 3:
                lab = {"개선": "처방이 재보정 이상", "구분 불가": "재보정과 구분 불가", "악화": "재보정 우위"}
                h3a = ci(D("g8", "f1", "zf", "f1r", gs)); h3b = ci(D("g8", "f1r", "zf", "f1r", gs)); ref_ = ci(D("g8", "f1", "zf", "f1", gs))
                apd = ci(D("g8", "ap", "zf", "ap", gs))
                print("  H3  γ8 − (zf+재보정) %+.4f [%+.4f,%+.4f] → %s" % (*h3a, lab[verdict(h3a[1], h3a[2])]))
                print("      γ8+재보정 − (zf+재보정) %+.4f [%+.4f,%+.4f] → %s" % (*h3b, lab[verdict(h3b[1], h3b[2])]))
                print("  참고 γ8 − zf (F1@0.5) %+.4f [%+.4f,%+.4f] %s | AUPRC γ8 − zf %+.4f [%+.4f,%+.4f] %s" % (
                    *ref_, verdict(ref_[1], ref_[2]), *apd, verdict(apd[1], apd[2])))
                cell["H3"] = {"g8_vs_zfr": h3a, "g8r_vs_zfr": h3b, "g8_vs_zf": ref_, "ap_g8_vs_zf": apd}
            # 수정 규칙(정확 탐색) — 같은 H1~H3를 F1@재보정(정확)으로 반복
            x1 = ci(D("zf", "f1x", "zf", "f1", zs)); X = {"H1": x1}
            line = "  [수정 규칙: 정확 탐색] H1 %+.4f [%+.4f,%+.4f] %s" % (*x1, verdict(x1[1], x1[2]))
            if len(ss) == 3 and D(rkey, "f1", "zf", "f1", ss).mean() >= THR:
                xr = D(rkey, "f1x", "zf", "f1x", ss); raw_ = D(rkey, "f1", "zf", "f1", ss)
                X["resid"] = ci(xr); X["thr_share"] = float(1 - xr.mean() / raw_.mean())
                line += " | 잔여 붕괴 %+.4f [%+.4f,%+.4f], 기준값 층 비율 %.2f" % (*X["resid"], X["thr_share"])
            if len(gs) == 3:
                xa = ci(D("g8", "f1", "zf", "f1x", gs)); xb = ci(D("g8", "f1x", "zf", "f1x", gs)); X.update(g8_vs_zfr=xa, g8r_vs_zfr=xb)
                line += "\n      γ8 − (zf+재보정) %+.4f [%+.4f,%+.4f] → %s | γ8+재보정 − (zf+재보정) %+.4f [%+.4f,%+.4f] → %s" % (
                    *xa, lab[verdict(xa[1], xa[2])], *xb, lab[verdict(xb[1], xb[2])])
            print(line); cell["exact"] = X
        else:
            print("  E9  해당 없음(zf run 부족)")
        res["cells"][reg] = cell

    # F (사후 추가 · 기술 통계): full 조건
    print("\n" + "-" * 100); print("[full 조건] (사후 추가 · 기술 통계 — 사전 고정 판정 아님)"); print("-" * 100)
    fs = [s for s in SEEDS if ("zf", s) in runs and ("g8", s) in runs]
    if len(fs) == 3:
        Fm = {}
        for k in ("zf", "g8"):
            for s in fs:
                z = runs[(k, s)][1]
                Fm[(k, s)] = {"f1": f1_at05(z["test_y"], z["test_full"]),
                              "f1r": f1_recal(z["val_y"], z["val_full"], z["test_y"], z["test_full"]),
                              "f1x": f1_recal_exact(z["val_y"], z["val_full"], z["test_y"], z["test_full"]),
                              "ap": auprc(z["test_y"], z["test_full"])}
        for k, lab in (("zf", "zf"), ("g8", "γ8")):
            A = lambda m: np.mean([Fm[(k, s)][m] for s in fs])
            print("  %-3s (n=3)  F1@0.5 %.4f | F1@재보정 %.4f | F1@재보정(정확) %.4f | AUPRC %.4f" % (lab, A("f1"), A("f1r"), A("f1x"), A("ap")))
        Df = lambda m: np.array([Fm[("g8", s)][m] - Fm[("zf", s)][m] for s in fs])
        a, b, c, x = ci(Df("f1")), ci(Df("f1r")), ci(Df("ap")), ci(Df("f1x"))
        print("  γ8 − zf (F1@0.5) %+.4f [%+.4f,%+.4f] %s | γ8+재보정 − zf+재보정 %+.4f [%+.4f,%+.4f] %s | AUPRC γ8 − zf %+.4f [%+.4f,%+.4f] %s" % (
            *a, verdict(a[1], a[2]), *b, verdict(b[1], b[2]), *c, verdict(c[1], c[2])))
        print("  [정확 탐색] γ8+재보정 − zf+재보정 %+.4f [%+.4f,%+.4f] %s" % (*x, verdict(x[1], x[2])))
        res["full_posthoc"] = {"means": {k: {m: float(np.mean([Fm[(k, s)][m] for s in fs])) for m in ("f1", "f1r", "f1x", "ap")} for k in ("zf", "g8")},
                               "g8_vs_zf": a, "g8r_vs_zfr": b, "ap_g8_vs_zf": c, "g8x_vs_zfx": x}
    else:
        print("  해당 없음(zf 또는 γ8 run 부족)")
        res["full_posthoc"] = None
    return res


# ============================ 가짜 로짓 대조 시험 ============================
def selftest():
    import tempfile
    rng0 = np.random.RandomState(0)
    L, NV, NT = 23, 2600, 7800
    prev = np.linspace(0.02, 0.45, L)
    root = tempfile.mkdtemp(prefix="mmv2_")
    for d in ("results/thesis_round/V2/E2_mmimdb", "results/thesis_round/V2/E9_mmimdb"):
        os.makedirs(os.path.join(root, d))
    yv = (rng0.rand(NV, L) < prev).astype(np.float32); yt = (rng0.rand(NT, L) < prev).astype(np.float32)

    def calib_logit(y, mu, rng):  # 등분산 정규 → 사후 로짓(잘 보정됨)
        s = y * mu + rng.randn(*y.shape)
        return mu * s - mu ** 2 / 2 + np.log(prev / (1 - prev))

    def write(name, s, lv, lt, mods=None, orig=None):
        arr, ev = {"val_y": yv, "test_y": yt}, {}
        for nk, ek in (("full", "full"), ("textmissing", "text"), ("imagemissing", "image")):
            arr["val_" + nk], arr["test_" + nk] = lv[nk].astype(np.float32), lt[nk].astype(np.float32)
            ev[ek] = {"macro": f1_at05(yt, arr["test_" + nk])}
        if mods:
            for k, v in mods.items():
                ev[k]["macro"] += v
        d = "E2_mmimdb" if ("only" in name) else "E9_mmimdb"
        stem = os.path.join(root, "results/thesis_round/V2", d, "clip_%s_s%d" % (name, s))
        np.savez_compressed(stem + "_logits.npz", **arr)
        json.dump({"config": {}, "eval": ev}, open(stem + ".json", "w"))
        if orig is not None:
            meth, g, dt = orig
            oe = json.loads(json.dumps(ev)); oe["text"]["macro"] += dt
            json.dump({"config": {"mode": "real", "backbone": "clip", "method": meth, "seed": s, "gamma": g, "limit": 0},
                       "eval": oe}, open(os.path.join(root, "results", "orig_%s_s%d.json" % (name, s)), "w"))

    for s in SEEDS:
        rng = np.random.RandomState(100 + s)
        G = {}
        for nm, mu in (("Rv", 2.0), ("Rt", 2.0), ("weak", 0.8)):
            G[nm] = (calib_logit(yv, mu, rng), calib_logit(yt, mu, rng))
        full = (G["Rv"][0] + G["Rt"][0], G["Rv"][1] + G["Rt"][1])
        # image_only = 천장 R(text 결손) / text_only = 천장 R(image 결손)
        write("image_only", s, {"full": full[0], "textmissing": G["Rv"][0], "imagemissing": G["weak"][0]},
              {"full": full[1], "textmissing": G["Rv"][1], "imagemissing": G["weak"][1]})
        write("text_only", s, {"full": full[0], "textmissing": G["weak"][0], "imagemissing": G["Rt"][0]},
              {"full": full[1], "textmissing": G["weak"][1], "imagemissing": G["Rt"][1]})
        # zf: text 결손 = R과 같은 순위, 로짓 −4 이동(순수 기준값 붕괴 → 양성 대조)
        #     image 결손 = R의 val 최적 기준값을 0.5로 옮긴 로짓(이미 최적 → H1 음성 대조, 붕괴 없음)
        thr = fit_thr(yv, sig(G["Rt"][0])); sh = -np.log(thr / (1 - thr))
        write("zerofill", s, {"full": full[0] - 2, "textmissing": G["Rv"][0] - 4, "imagemissing": G["Rt"][0] + sh},
              {"full": full[1] - 2, "textmissing": G["Rv"][1] - 4, "imagemissing": G["Rt"][1] + sh},
              mods={"full": 0.01} if s == 2 else None, orig=("zerofill", 2.0, 0.0 if s != 3 else 0.02))
        # γ8: text 결손 = 약한 신호(순위 붕괴) → 재보정을 해도 R에 못 미침
        write("gamma_g8.0", s, {"full": full[0], "textmissing": G["weak"][0], "imagemissing": G["Rt"][0]},
              {"full": full[1], "textmissing": G["weak"][1], "imagemissing": G["Rt"][1]}, orig=("gamma", 8.0, 0.0))
    print("가짜 로짓 루트:", root)
    print("기대: K1 경고 1건(zf s2 full, 일부러 +0.01) | K2 경고 1건(zf s3, 일부러 0.02) |")
    print("      text 결손: 원붕괴 큼, H1 개선, 잔여 붕괴 ≈ 0, 기준값 층 비율 ≈ 1, AUPRC 차 ≈ 0 (순수 기준값 붕괴),")
    print("                 H3 γ8(순위 붕괴)은 재보정 우위, AUPRC γ8 − zf 악화")
    print("      image 결손: E2 붕괴 없음(R과 zf 순위 같고 zf는 이미 최적 기준값), H1 ≈ 0 (구분 불가 또는 아주 작음)")
    print("      full(사후): zf = γ8 로짓 − 2 → F1@0.5 차 있음(γ8 > zf), 대칭 재보정 차 ≈ 0, AUPRC 차 ≈ 0")
    r = analyze(root, "clip")
    t, i = r["cells"]["text"], r["cells"]["image"]
    ok = [len(r["K1_bad"]) == 1, sum(abs(x["d_text"]) > 1e-3 for x in r["K2"]) == 1,
          t["H2"] is not None and abs(t["H2"]["resid"][0]) < 0.01 and t["H2"]["thr_share"] > 0.9 and abs(t["H2"]["ap_gap"][0]) < 1e-4,
          t["H1"][1] > 0, t["H3"]["g8_vs_zfr"][2] < 0, t["H3"]["ap_g8_vs_zf"][2] < 0,
          not i["E2"]["R_has"], abs(i["H1"][0]) < 0.01,
          r["full_posthoc"]["g8_vs_zf"][1] > 0, abs(r["full_posthoc"]["ap_g8_vs_zf"][0]) < 1e-4,
          abs(r["full_posthoc"]["g8x_vs_zfx"][0]) < 1e-9, abs(t["exact"]["resid"][0]) < 1e-9, abs(t["exact"]["H1"][0] - t["H1"][0]) > 0]
    import shutil; shutil.rmtree(root, ignore_errors=True)
    print("\n자체 시험 판정:", "모두 통과" if all(ok) else "★실패 %s" % ok)
    return all(ok)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    BB = sys.argv[1]
    r = analyze(".", BB)
    if r is not None:
        out = "results/thesis_round/V2/mmimdb_v2_%s.json" % BB
        json.dump(r, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("\n결과 저장:", out)
    print("=== ALL DONE (MM-IMDb v2 분석 %s) ===" % BB)
