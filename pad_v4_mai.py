"""추가 실험 v4 — 제안 방법 MAI(결손 인지 추론) PAD 확인 분석 (학습 없음, CPU, 저장된 로짓)
  python pad_v4_mai.py             # 서버 ~/Yechan/mmimdb_exp (conda mmimdb), v2_pad_heads_dump.py 실행 후
  python pad_v4_mai.py --selftest
규칙: thesis_round/ADDENDUM_v4_MAI_prereg.md (2026-11-11, 보조 분류기 출력 열람 전 고정). 이 스크립트는 그 문서 §4를 그대로 구현.
입력: V2/dumpH/g8_s{0,1,2}_f*.npz (융합 + 보조), V2/dump/{zf,md}_s{0,1,2}_f*.npz, onlymeta_s0·onlyimage_s0 (천장 R), [dumpH/gm_s0, 보고만]
셀: image 결손(메타 남음, 키 image_missing, 보조 headm, R = onlymeta의 image_missing) ← 주 판정
    meta 결손(이미지 남음, 키 meta_missing_input0, 보조 headi, R = onlyimage의 meta_missing_proj0)
MAI: z(α) = α·head + (1−α)·융합, α ∈ {0,…,1} 중 val macro OvR AUROC 최대(동률은 작은 α), 판단은 argmax(재보정 없음).
단위: fold마다 seed 0~2 평균(R은 seed 0) → fold 5개 짝지은 95% CI(t=2.776, ddof=1).
주 판정(image 결손): P1 MAI − γ8 융합 macro-F1 개선 그리고 P2 MAI − ModDrop macro-F1 악화 아님 → "확인"
  아니면 P1이 어느 셀·어느 지표(F1, AUROC)에서든 개선 → "부분 확인" / 그 밖 → "불확인"
"""
import os, sys, json, hashlib
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

DUMP, DUMPH = "results/thesis_round/V2/dump", "results/thesis_round/V2/dumpH"
PREREG = "thesis_round/ADDENDUM_v4_MAI_prereg.md"
FOLDS, SEEDS, T5 = range(5), (0, 1, 2), 2.776
ALPHAS = np.round(np.arange(0, 11) / 10.0, 1)
CASES = (("image", "image_missing", "headm", "onlymeta_s0", "image_missing"),
         ("meta", "meta_missing_input0", "headi", "onlyimage_s0", "meta_missing_proj0"))


def ci(d):
    d = np.asarray(d, float); m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d)); return float(m), float(m - T5 * se), float(m + T5 * se)


def verdict(lo, hi): return "개선" if lo > 0 else ("악화" if hi < 0 else "구분 불가")


def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def auroc(y, logit):
    p = softmax(np.asarray(logit, np.float64)); cs = [c for c in range(p.shape[1]) if 0 < (y == c).sum() < len(y)]
    return float(np.mean([roc_auc_score((y == c).astype(int), p[:, c]) for c in cs]))


def f1(y, logit): return float(f1_score(y, np.asarray(logit).argmax(1), average="macro", labels=list(range(6)), zero_division=0))


def pick_alpha(yv, fv, hv):
    sc = [auroc(yv, a * hv + (1 - a) * fv) for a in ALPHAS]
    best = max(sc); return float(ALPHAS[min(k for k, s in enumerate(sc) if s >= best - 1e-12)])


def run(root):
    L = lambda d, tag, f: dict(np.load(os.path.join(root, d, "%s_f%d.npz" % (tag, f))))
    need = [(DUMPH, "g8_s%d" % s) for s in SEEDS] + [(DUMP, "%s_s%d" % (m, s)) for m in ("zf", "md") for s in SEEDS] + \
           [(DUMP, "onlymeta_s0"), (DUMP, "onlyimage_s0")]
    miss = ["%s/%s_f%d" % (d, t, f) for d, t in need for f in FOLDS if not os.path.exists(os.path.join(root, d, "%s_f%d.npz" % (t, f)))]
    pr = os.path.join(root, PREREG); md5 = hashlib.md5(open(pr, "rb").read()).hexdigest() if os.path.exists(pr) else "없음"
    print("=" * 104); print("PAD MAI 확인 분석 | fold 5, seed %s, t=%.3f | 사전 고정 문서 md5 %s" % (SEEDS, T5, md5)); print("=" * 104)
    if miss:
        print("★ 없는 로짓: %s → 중단" % miss); return None
    has_gm = all(os.path.exists(os.path.join(root, DUMPH, "gm_s0_f%d.npz" % f)) for f in FOLDS)
    res = {"prereg_md5": md5, "cells": {}}
    for cell, key, hk, rtag, rkey in CASES:
        V = {k: {"f": [], "a": []} for k in ("R", "zf", "md", "g8", "head", "mai", "gm")}; alphas = []
        for f in FOLDS:
            Rd = L(DUMP, rtag, f); yt = Rd["test_y"]
            V["R"]["f"].append(f1(yt, Rd["test_" + rkey])); V["R"]["a"].append(auroc(yt, Rd["test_" + rkey]))
            acc = {k: {"f": [], "a": []} for k in ("zf", "md", "g8", "head", "mai")}
            for s in SEEDS:
                for m in ("zf", "md"):
                    d = L(DUMP, "%s_s%d" % (m, s), f)
                    assert np.array_equal(d["test_y"], yt), "K3 라벨 불일치"
                    acc[m]["f"].append(f1(yt, d["test_" + key])); acc[m]["a"].append(auroc(yt, d["test_" + key]))
                d = L(DUMPH, "g8_s%d" % s, f); assert np.array_equal(d["test_y"], yt), "K3 라벨 불일치"
                fv, hv, ft, ht = d["val_" + key], d["val_" + hk], d["test_" + key], d["test_" + hk]
                a = pick_alpha(d["val_y"], fv, hv); alphas.append(a); zt = a * ht + (1 - a) * ft
                for k, z in (("g8", ft), ("head", ht), ("mai", zt)):
                    acc[k]["f"].append(f1(yt, z)); acc[k]["a"].append(auroc(yt, z))
            for k in acc:
                for m in ("f", "a"):
                    V[k][m].append(float(np.mean(acc[k][m])))
            if has_gm:
                g = L(DUMPH, "gm_s0", f); V["gm"]["f"].append(f1(yt, g["test_" + key])); V["gm"]["a"].append(auroc(yt, g["test_" + key]))
        V = {k: {m: np.array(v) for m, v in d.items()} for k, d in V.items()}
        print("\n[%s 결손] (%s 경로가 남음, 보조 = %s) | 고른 α 분포 %s" % (
            cell, "메타" if cell == "image" else "이미지", hk, {float(a): alphas.count(a) for a in sorted(set(alphas))}))
        for k, lab in (("R", "천장 R"), ("zf", "zf"), ("md", "ModDrop"), ("g8", "γ8 융합"), ("head", "보조만"), ("mai", "MAI")) + \
                      ((("gm", "결합(seed0)"),) if has_gm else ()):
            print("  %-10s macro-F1 %.4f | AUROC %.4f" % (lab, V[k]["f"].mean(), V[k]["a"].mean()))
        P = {}
        for name, a, b in (("P1 MAI − γ8 융합", "mai", "g8"), ("P2 MAI − ModDrop", "mai", "md"), ("P3 MAI − zf", "mai", "zf"),
                           ("P4 MAI − 천장 R", "mai", "R"), ("P5 보조만 − γ8 융합", "head", "g8")):
            x, y = ci(V[a]["f"] - V[b]["f"]), ci(V[a]["a"] - V[b]["a"]); P[name] = {"f1": x, "auroc": y}
            print("  %-18s macro-F1 %+.4f [%+.4f,%+.4f] %-5s | AUROC %+.4f [%+.4f,%+.4f] %s" % (name, *x, verdict(x[1], x[2]), *y, verdict(y[1], y[2])))
        if has_gm:
            x = ci(V["mai"]["f"] - V["gm"]["f"]); P["참고 MAI − 결합(seed0)"] = {"f1": x}
            print("  %-18s macro-F1 %+.4f [%+.4f,%+.4f] %s (보고만)" % ("참고 MAI − 결합", *x, verdict(x[1], x[2])))
        res["cells"][cell] = {"P": P, "alphas": alphas, "means": {k: {m: float(v.mean()) for m, v in d.items()} for k, d in V.items() if len(d["f"])}}
    pi = res["cells"]["image"]["P"]
    conf = pi["P1 MAI − γ8 융합"]["f1"][1] > 0 and not (pi["P2 MAI − ModDrop"]["f1"][2] < 0)
    anyimp = any(res["cells"][c]["P"]["P1 MAI − γ8 융합"][m][1] > 0 for c in ("image", "meta") for m in ("f1", "auroc"))
    v = "확인" if conf else ("부분 확인" if anyimp else "불확인")
    res["verdict"] = v; print("\n▶ 주 판정(사전 고정): %s" % v)
    return res


def selftest():
    import tempfile, shutil
    out = []
    for case in ("pos", "neg"):
        root = tempfile.mkdtemp(prefix="padmai_"); os.makedirs(os.path.join(root, DUMP)); os.makedirs(os.path.join(root, DUMPH))
        r0 = np.random.RandomState(3)
        for f in FOLDS:
            yv = r0.choice(6, 300, p=[.3, .35, .03, .12, .1, .1]); yt = r0.choice(6, 460, p=[.3, .35, .03, .12, .1, .1])
            def lg(y, mu, rng): return mu * np.eye(6)[y] + rng.randn(len(y), 6)
            def save(d, tag, V, T):
                arr = {"val_y": yv, "test_y": yt}
                for k in V:
                    arr["val_" + k], arr["test_" + k] = V[k], T[k]
                np.savez_compressed(os.path.join(root, d, "%s_f%d.npz" % (tag, f)), **arr)
            for s in SEEDS:
                rng = np.random.RandomState(100 * f + s)
                mk = lambda mu: (lg(yv, mu, rng), lg(yt, mu, rng))
                weak, strong = mk(0.8), mk(2.5)
                base = lambda w: {"image_missing": w, "meta_missing_input0": w, "full": w}
                for m in ("zf", "md"):
                    save(DUMP, "%s_s%d" % (m, s), base(weak[0]), base(weak[1]))
                hm = strong if case == "pos" else mk(0.0)          # 음성: 정보 없는 보조 분류기(순수 잡음)
                save(DUMPH, "g8_s%d" % s, dict(base(weak[0]), headm=hm[0], headi=hm[0]), dict(base(weak[1]), headm=hm[1], headi=hm[1]))
            rng = np.random.RandomState(999 + f); st = (lg(yv, 2.5, rng), lg(yt, 2.5, rng))
            save(DUMP, "onlymeta_s0", {"image_missing": st[0]}, {"image_missing": st[1]})
            save(DUMP, "onlyimage_s0", {"meta_missing_proj0": st[0]}, {"meta_missing_proj0": st[1]})
        r = run(root); shutil.rmtree(root, ignore_errors=True)
        exp = "확인" if case == "pos" else "불확인"
        out.append(r["verdict"] == exp); print("[자체 시험 %s] 기대 %s → 얻음 %s" % (case, exp, r["verdict"]))
    print("\n자체 시험:", out); print("자체 시험 판정:", "모두 통과" if all(out) else "★실패")
    return all(out)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    r = run(".")
    if r is not None:
        out = "results/thesis_round/V2/pad_v4_mai.json"
        json.dump(r, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float); print("\n결과 저장:", out)
    print("=== ALL DONE (PAD MAI 확인) ===")
