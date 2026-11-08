"""추가 실험 v2 — PAD 재보정 실패 원인 진단 (진단 전용, 학습 없음, CPU, 저장된 로짓 V2/dump)
  python pad_v2_recal_cause.py
  python pad_v2_recal_cause.py --selftest
[고정 규칙 — 실행 전, 2026-11-08]
배경: PAD에서 val로 맞춘 편향 재보정은 붕괴를 거의 못 메웠고(F1 기준 16%·28%), test로 맞춘 상한은 다 메웠음.
  MM-IMDb val 축소(R3) 결과 'val이 작아서'는 기각됨. 남은 두 설명을 가른다.
  (가) val–test 분포 차이: val이 test를 대표하지 못해 val에서 맞춘 기준값이 test에 맞지 않음
  (나) 추정 잡음: 이 규모(수백 장, 6클래스, 희귀 클래스)에서는 어떤 데이터로 맞춰도 편향 추정이 불안정함
방법: zf 모델(seed 0·1·2 × fold 0~4)의 결손 로짓에서 클래스별 편향 6개를 val F1 최대화(좌표 상승, 격자 −6~+6 0.1, 3회 — pad_recal_ub와 동일)로 맞춤.
  ① val 맞춤 → test 반쪽에 적용  ② test 교차 맞춤: test를 클래스 층화로 반씩 나눠(난수 7+fold) 한쪽에서 맞추고 다른 쪽에 적용(양방향 평균)
  ③ 참고: 같은 반쪽에서 맞추고 그 반쪽에 적용(낙관적 상한). 모든 값은 test 반쪽에서 계산하고, 같은 반쪽의 zf·R과 비교.
  회복 비율 = (재보정 − zf)/(R − zf), fold별 seed 평균, n=5.
판정(image 결손이 주, meta 결손은 보고), 위에서부터 차례로:
  ③ < 0.5 → "편향으로 메울 수 없는 붕괴(이 진단의 대상 아님)"
  ② ≥ 0.5 이고 ① < 0.25 → "(가) val–test 분포 차이 설명과 일치"
  ①·② 모두 < 0.25 → "(나) 추정 잡음 설명과 일치(같은 데이터에선 맞지만 다른 같은 크기 데이터로 옮겨지지 않음)"
  그 외 → "판단 보류"
참고 보고: fold별 val·test 클래스 비율의 총변동거리(TV), val 최소 클래스 수, MEL 제외 5클래스 macro-F1로 같은 계산.
test 반쪽을 맞춤에 쓰므로 진단 전용이며 선택·판정(E9)에는 쓰지 않는다.
"""
import os, sys, json
import numpy as np

DUMP = "results/thesis_round/V2/dump"; K = 6; FOLDS = range(5); SEEDS = (0, 1, 2); T5 = 2.776
GRID = np.round(np.linspace(-6, 6, 121), 2)
CASES = (("image", "image_missing", "onlymeta_s0", "image_missing"), ("meta", "meta_missing_input0", "onlyimage_s0", "meta_missing_proj0"))


def macro_f1(y, pred, labs=range(K)):
    f = []
    for k in labs:
        tp = np.sum((pred == k) & (y == k)); fp = np.sum((pred == k) & (y != k)); fn = np.sum((pred != k) & (y == k))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def F1(y, L, b=None, labs=range(K)): return macro_f1(y, (L if b is None else L + b).argmax(1), labs)


def fit_f1(L, y):
    b = np.zeros(K); best = F1(y, L, b)
    for _ in range(3):
        for k in range(K):
            cur = b[k]
            for v in GRID:
                b[k] = v; s = F1(y, L, b)
                if s > best + 1e-12 or (abs(s - best) <= 1e-12 and abs(v) < abs(cur)):
                    best, cur = s, v
            b[k] = cur
    return b


def halves(y, seed):
    rng = np.random.RandomState(seed); A = []
    for k in range(K):
        idx = np.where(y == k)[0]; rng.shuffle(idx); A += list(idx[: len(idx) // 2])
    A = np.array(sorted(A), int); B = np.setdiff1d(np.arange(len(y)), A)
    return A, B


def ci(d):
    d = np.asarray(d, float); m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d)); return float(m), float(m - T5 * se), float(m + T5 * se)


def run(root):
    L = lambda tag, f: dict(np.load(os.path.join(root, DUMP, "%s_f%d.npz" % (tag, f))))
    print("=" * 100); print("PAD 재보정 실패 원인 진단 (진단 전용) — zf seed %s × fold 5, test 반쪽 교차 맞춤" % (SEEDS,)); print("=" * 100)
    res = {}
    for cell, key, rtag, rkey in CASES:
        out = {lab: {k: [] for k in ("val", "xfit", "same")} for lab in ("all", "noMEL")}; tv, vmin, nv, nt = [], [], [], []
        for f in FOLDS:
            Rd = L(rtag, f); yt = Rd["test_y"]; A, B = halves(yt, 7 + f)
            accs = {lab: {k: [] for k in ("val", "xfit", "same")} for lab in out}
            for s in SEEDS:
                d = L("zf_s%d" % s, f); Lt, Lv, yv = d["test_" + key], d["val_" + key], d["val_y"]
                assert np.array_equal(d["test_y"], yt)
                if s == 0:
                    pv = np.bincount(yv, minlength=K) / len(yv); pt = np.bincount(yt, minlength=K) / len(yt)
                    tv.append(0.5 * np.abs(pv - pt).sum()); vmin.append(int(np.bincount(yv, minlength=K).min())); nv.append(len(yv)); nt.append(len(yt))
                bv = fit_f1(Lv, yv); bA = fit_f1(Lt[A], yt[A]); bB = fit_f1(Lt[B], yt[B])
                for lab, labs in (("all", range(K)), ("noMEL", [0, 1, 3, 4, 5])):      # 클래스 순서 ACK,BCC,MEL,NEV,SCC,SEK → MEL=2
                    def rec(bias_for_half):
                        r = []
                        for H, b in zip((A, B), bias_for_half):
                            zf = F1(yt[H], Lt[H], None, labs); Rr = F1(yt[H], Rd["test_" + rkey][H], None, labs)
                            r.append(((F1(yt[H], Lt[H], b, labs) - zf), (Rr - zf)))
                        return np.mean([x[0] for x in r]), np.mean([x[1] for x in r])
                    gv, col = rec((bv, bv)); gx, _ = rec((bB, bA)); gs, _ = rec((bA, bB))
                    accs[lab]["val"].append((gv, col)); accs[lab]["xfit"].append((gx, col)); accs[lab]["same"].append((gs, col))
            for lab in out:
                for k in out[lab]:
                    g = np.mean([x[0] for x in accs[lab][k]]); c = np.mean([x[1] for x in accs[lab][k]])
                    out[lab][k].append((g, c))
        print("\n[%s 결손] val 크기 %s, test 크기 %s, val 최소 클래스 수 %s, val–test 클래스 비율 TV %s" % (
            cell, nv, nt, vmin, ["%.3f" % x for x in tv]))
        cellres = {"tv": tv, "vmin": vmin}
        for lab in ("all", "noMEL"):
            line = []; rr = {}
            for k, nm in (("val", "① val 맞춤"), ("xfit", "② test 교차 맞춤"), ("same", "③ 같은 반쪽(상한)")):
                g = np.array([x[0] for x in out[lab][k]]); c = np.array([x[1] for x in out[lab][k]])
                ratio = float(g.mean() / c.mean()) if c.mean() > 0 else float("nan"); rr[k] = ratio
                line.append("%s 이득 %+.4f [%+.4f,%+.4f] 회복 %.2f" % ((nm,) + ci(g) + (ratio,)))
            print("  %s (붕괴 R−zf %+.4f): " % ("6클래스" if lab == "all" else "MEL 제외", np.mean([x[1] for x in out[lab]["val"]])) + " | ".join(line))
            cellres[lab] = rr
        res[cell] = cellres
    r = res["image"]["all"]
    v = ("편향으로 메울 수 없는 붕괴(이 진단의 대상 아님)" if r["same"] < 0.5 else
         ("(가) val–test 분포 차이 설명과 일치" if (r["xfit"] >= 0.5 and r["val"] < 0.25) else
          ("(나) 추정 잡음 설명과 일치" if (r["xfit"] < 0.25 and r["val"] < 0.25) else "판단 보류")))
    res["verdict"] = v; print("\n▶ 판정(image 결손, 6클래스): %s (① %.2f, ② %.2f, ③ %.2f)" % (v, r["val"], r["xfit"], r["same"]))
    return res


def selftest():
    import tempfile, shutil
    out = []
    for case in ("shift", "noise"):
        root = tempfile.mkdtemp(prefix="padrc_"); os.makedirs(os.path.join(root, DUMP)); rng0 = np.random.RandomState(0)
        for f in FOLDS:
            yv = rng0.randint(0, K, 280); yt = rng0.randint(0, K, 460)
            def lg(y, mu, off, rng):
                z = rng.randn(len(y), K); z[np.arange(len(y)), y] += mu; return z + off
            off_t = np.array([0, 0, -2.5, 0, 0, 0.]) if case == "shift" else np.zeros(K)      # test에만 MEL 쪽 편향 → 붕괴
            off_v = np.zeros(K)                                                              # val에는 편향 없음(분포 차이)
            for s in SEEDS:
                rng = np.random.RandomState(10 * f + s)
                if case == "shift":
                    d = {"val_y": yv, "test_y": yt, "val_image_missing": lg(yv, 2.0, off_v, rng), "test_image_missing": lg(yt, 2.0, off_t, rng)}
                else:   # 순위 자체가 약함(편향으로 못 메우는 붕괴)
                    d = {"val_y": yv, "test_y": yt, "val_image_missing": lg(yv, 0.7, 0, rng), "test_image_missing": lg(yt, 0.7, 0, rng)}
                d["val_meta_missing_input0"], d["test_meta_missing_input0"] = d["val_image_missing"], d["test_image_missing"]
                np.savez(os.path.join(root, DUMP, "zf_s%d_f%d.npz" % (s, f)), **d)
            rng = np.random.RandomState(99 + f)
            for t, k in (("onlymeta_s0", "image_missing"), ("onlyimage_s0", "meta_missing_proj0")):
                np.savez(os.path.join(root, DUMP, "%s_f%d.npz" % (t, f)), val_y=yv, test_y=yt, **{"test_" + k: lg(yt, 2.0, 0, rng)})
        r = run(root); shutil.rmtree(root, ignore_errors=True)
        exp = "(가) val–test 분포 차이 설명과 일치" if case == "shift" else "편향으로 메울 수 없는 붕괴(이 진단의 대상 아님)"
        out.append(r["verdict"] == exp); print("[자체 시험 %s] 기대 '%s' → 얻음 '%s'" % (case, exp, r["verdict"]))
    print("\n자체 시험 판정:", "모두 통과" if all(out) else "★실패 %s" % out)
    return all(out)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    r = run(".")
    out = "results/thesis_round/V2/pad_v2_recal_cause.json"
    json.dump(r, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float); print("결과 저장:", out)
    print("=== ALL DONE (PAD 재보정 원인 진단) ===")
