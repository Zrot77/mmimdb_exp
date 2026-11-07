"""추가 실험 v2 — PAD 경로별 순위 분석(R4) + E4 PAD 입력 (학습 없음, CPU, 저장된 로짓 V2/dump 사용)
  python pad_v2_rank.py             # 서버 ~/Yechan/mmimdb_exp
  python pad_v2_rank.py --selftest
입력: results/thesis_round/V2/dump/{zf,g8,md}_s{0,1,2}_f{0..4}.npz, onlymeta_s0_f*.npz, onlyimage_s0_f*.npz (pad_v2_analysis와 같은 파일)

[고정 규칙 — 실행 전, 2026-11-07]
목적: MM-IMDb에서 본 '경로별 처방'(γ는 학습 가능한 이미지 경로, ModDrop은 고정 특징 텍스트 경로)의 원인 설명을
  두 경로가 모두 학습 가능한 PAD(이미지 = SigLIP 미세조정, 메타 = 2층 MLP)에서 순위 지표로 확인한다.
셀: image 결손(메타 경로가 남음, 키 image_missing, R = only_meta) / meta 결손(이미지 경로가 남음, 키 meta_missing_input0, R = only_image)
지표: macro one-vs-rest AUROC(softmax 확률, test에 양성·음성이 모두 있는 클래스만 평균) = 순위 지표. 참고로 argmax macro-F1.
단위: fold별로 seed{0,1,2} 평균(R은 seed 0만 있음), n=5 fold, t=2.776, ddof=1.
  Q1 AUROC R − zf (순위 층 격차)   Q2 AUROC γ8 − zf   Q3 AUROC MD − zf   Q4 AUROC γ8 − MD
해석 규칙: Q1이 개선(CI 하한>0)인 셀만 판단에 쓴다.
  그런 셀 모두에서 Q2 개선 → "학습 가능성 설명과 일치(두 경로 모두 학습 가능한 PAD에서 γ가 순위를 개선)"
  Q1 개선 셀이 있으나 그중 일부에서만 Q2 개선 → "부분 일치" / 그런 셀에서 Q2가 모두 개선 아님 → "불일치"
  Q1 개선 셀이 없음 → "판정 불가(순위 층 격차 없음)"
E4 입력(사전등록 v2 §E4): 붕괴 = E2 정의 R(seed 0) − zf(seed 0) macro-F1 ≥ 0.05, γ 이득 = γ8 − zf macro-F1(fold별 seed 평균, n=5).
"""
import os, sys, json
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

DUMP = "results/thesis_round/V2/dump"; FOLDS = range(5); SEEDS = (0, 1, 2); T5 = 2.776; THR = 0.05
CASES = (("image", "image_missing", "onlymeta_s0", "image_missing"), ("meta", "meta_missing_input0", "onlyimage_s0", "meta_missing_proj0"))


def ci(d):
    d = np.asarray(d, float); m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d)); return float(m), float(m - T5 * se), float(m + T5 * se)


def verdict(lo, hi): return "개선" if lo > 0 else ("악화" if hi < 0 else "구분 불가")


def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def auroc(y, logit):
    p = softmax(np.asarray(logit, np.float64)); cs = [c for c in range(p.shape[1]) if 0 < (y == c).sum() < len(y)]
    return float(np.mean([roc_auc_score((y == c).astype(int), p[:, c]) for c in cs]))


def f1(y, logit): return float(f1_score(y, np.asarray(logit).argmax(1), average="macro", labels=list(range(6))))


def run(root):
    L = lambda tag, f: dict(np.load(os.path.join(root, DUMP, "%s_f%d.npz" % (tag, f))))
    miss = [("%s_f%d" % (t, f)) for t in ["%s_s%d" % (m, s) for m in ("zf", "g8", "md") for s in SEEDS] + ["onlymeta_s0", "onlyimage_s0"]
            for f in FOLDS if not os.path.exists(os.path.join(root, DUMP, "%s_f%d.npz" % (t, f)))]
    print("=" * 100); print("PAD 경로별 순위 분석(R4) + E4 입력 | fold 5, seed %s" % (SEEDS,)); print("=" * 100)
    if miss:
        print("★ 없는 로짓: %s → 중단" % miss); return None
    res = {"cells": {}}; judged = []
    for cell, key, rtag, rkey in CASES:
        A = {k: [] for k in ("R", "zf", "g8", "md")}; F = {k: [] for k in ("R", "zf", "g8", "md", "zf0")}
        for f in FOLDS:
            Rd = L(rtag, f); A["R"].append(auroc(Rd["test_y"], Rd["test_" + rkey])); F["R"].append(f1(Rd["test_y"], Rd["test_" + rkey]))
            for m in ("zf", "g8", "md"):
                a, b = [], []
                for s in SEEDS:
                    d = L("%s_s%d" % (m, s), f); a.append(auroc(d["test_y"], d["test_" + key])); b.append(f1(d["test_y"], d["test_" + key]))
                    if m == "zf" and s == 0:
                        F["zf0"].append(b[-1])
                A[m].append(np.mean(a)); F[m].append(np.mean(b))
        A = {k: np.array(v) for k, v in A.items()}; F = {k: np.array(v) for k, v in F.items()}
        q = {"Q1 AUROC R − zf": ci(A["R"] - A["zf"]), "Q2 AUROC γ8 − zf": ci(A["g8"] - A["zf"]),
             "Q3 AUROC MD − zf": ci(A["md"] - A["zf"]), "Q4 AUROC γ8 − MD": ci(A["g8"] - A["md"])}
        print("\n[%s 결손] (%s 경로가 남음)" % (cell, "메타" if cell == "image" else "이미지"))
        print("  AUROC: R %.4f | zf %.4f | γ8 %.4f | MD %.4f   (참고 macro-F1: R %.4f | zf %.4f | γ8 %.4f | MD %.4f)" % (
            A["R"].mean(), A["zf"].mean(), A["g8"].mean(), A["md"].mean(), F["R"].mean(), F["zf"].mean(), F["g8"].mean(), F["md"].mean()))
        for k, c in q.items():
            print("  %-18s %+.4f [%+.4f,%+.4f] %s" % (k, *c, verdict(c[1], c[2])))
        col = float((F["R"] - F["zf0"]).mean()); g = ci(F["g8"] - F["zf"])
        print("  [E4 입력] 붕괴(R−zf, seed 0, F1) %+.4f → %s | γ8−zf F1 %+.4f [%+.4f,%+.4f] %s → 셀 %s" % (
            col, "있음" if col >= THR else "없음", *g, verdict(g[1], g[2]), "일치" if (col >= THR) == (g[1] > 0) else "★불일치"))
        res["cells"][cell] = {"auroc": {k: float(v.mean()) for k, v in A.items()}, "Q": q,
                              "E4": {"collapse": col, "has": bool(col >= THR), "g8_gain": g}}
        if q["Q1 AUROC R − zf"][1] > 0:
            judged.append(q["Q2 AUROC γ8 − zf"][1] > 0)
    v = ("판정 불가(순위 층 격차 없음)" if not judged else
         ("학습 가능성 설명과 일치(두 경로 모두 학습 가능한 PAD에서 γ가 순위를 개선)" if all(judged) else ("부분 일치" if any(judged) else "불일치")))
    res["verdict"] = v; print("\n▶ R4 해석: %s (판단에 쓴 셀 %d개)" % (v, len(judged)))
    return res


def selftest():
    import tempfile, shutil
    out = []
    for case in ("both", "none"):
        root = tempfile.mkdtemp(prefix="padrk_"); os.makedirs(os.path.join(root, DUMP))
        rng0 = np.random.RandomState(0)
        for f in FOLDS:
            yv = rng0.randint(0, 6, 300); yt = rng0.randint(0, 6, 450)
            def lg(y, mu, rng): z = rng.randn(len(y), 6); z[np.arange(len(y)), y] += mu; return z
            for s in SEEDS:
                rng = np.random.RandomState(100 * f + s)
                good = lambda y: lg(y, 2.0, rng); bad = lambda y: lg(y, 0.6, rng)
                for m in ("zf", "g8", "md"):
                    gm = (m == "g8" and case == "both")
                    d = {"val_y": yv, "test_y": yt}
                    for k in ("image_missing", "meta_missing_input0", "meta_missing_proj0"):
                        d["val_" + k] = (good if gm else bad)(yv); d["test_" + k] = (good if gm else bad)(yt)
                    np.savez(os.path.join(root, DUMP, "%s_s%d_f%d.npz" % (m, s, f)), **d)
            rng = np.random.RandomState(999 + f)
            for t in ("onlymeta_s0", "onlyimage_s0"):
                d = {"val_y": yv, "test_y": yt}
                for k in ("image_missing", "meta_missing_proj0"):
                    d["val_" + k] = lg(yv, 2.0, rng); d["test_" + k] = lg(yt, 2.0, rng)
                np.savez(os.path.join(root, DUMP, "%s_f%d.npz" % (t, f)), **d)
        r = run(root); shutil.rmtree(root, ignore_errors=True)
        exp = "학습 가능성 설명과 일치(두 경로 모두 학습 가능한 PAD에서 γ가 순위를 개선)" if case == "both" else "불일치"
        out.append(r["verdict"] == exp); print("[자체 시험 %s] 기대 '%s' → 얻음 '%s'" % (case, exp, r["verdict"]))
    print("\n자체 시험 판정:", "모두 통과" if all(out) else "★실패 %s" % out)
    return all(out)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    r = run(".")
    if r is not None:
        out = "results/thesis_round/V2/pad_v2_rank.json"
        json.dump(r, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float); print("결과 저장:", out)
    print("=== ALL DONE (PAD R4) ===")
