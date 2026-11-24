"""(사후) PAD meta 결손 정의 민감도 분석 — 학습 없음, CPU, 저장된 로짓만 사용
  python pad_meta_proj0_sens.py              # 서버 ~/Yechan/mmimdb_exp (conda mmimdb), pad_v4_mai.py와 같은 폴더
  python pad_meta_proj0_sens.py --selftest
배경(2026-11-24 확인):
  - train_pad.py의 주 결손 정의는 meta_missing = 투영 공간 0(pm=0), 천장 R(only_image)과 ModDrop 학습도 투영 0
  - 그런데 v2·v4 PAD 분석(E2 붕괴, R4 순위, 재보정 원인, 보조 분류기 혼합 추론 meta 셀)은 zf·γ8·ModDrop의 meta 결손에
    meta_missing_input0(메타 입력 0 → MLP 통과)을 썼고 천장 R만 투영 0으로 잼 → 같은 비교 안에서 정의가 섞임
  - 이 스크립트는 meta 결손을 투영 0(meta_missing_proj0)으로 통일했을 때 결론이 바뀌는지 본다. 사전 판정은 바꾸지 않는다(사후 보고)
image 결손 셀은 정의가 하나뿐이라 대상이 아님(비교용으로만 다시 출력)
출력: results/thesis_round/V2/pad_meta_proj0_sens.json (이미 있으면 덮어쓰지 않고 이름 뒤에 시각을 붙임)
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pad_v4_mai as P

KEYS = ("meta_missing_input0", "meta_missing_proj0")
OUT = "results/thesis_round/V2/pad_meta_proj0_sens.json"


def load(root, d, tag, f):
    return dict(np.load(os.path.join(root, d, "%s_f%d.npz" % (tag, f))))


def part_b(root):
    """meta 결손 셀: 붕괴(R − zf)와 처방 비교를 두 정의로 나란히 (fold별 seed 0~2 평균 → fold 5개 짝지음)"""
    print("\n" + "=" * 104); print("[B] meta 결손 셀 — 붕괴와 처방 비교, 두 정의 나란히 (R은 두 경우 모두 only_image의 투영 0)"); print("=" * 104)
    out = {}
    for key in KEYS:
        V = {k: {"f": [], "a": []} for k in ("R", "zf", "md", "g8")}
        for f in P.FOLDS:
            Rd = load(root, P.DUMP, "onlyimage_s0", f); yt = Rd["test_y"]
            V["R"]["f"].append(P.f1(yt, Rd["test_meta_missing_proj0"])); V["R"]["a"].append(P.auroc(yt, Rd["test_meta_missing_proj0"]))
            acc = {k: {"f": [], "a": []} for k in ("zf", "md", "g8")}
            for s in P.SEEDS:
                for m, d in (("zf", P.DUMP), ("md", P.DUMP), ("g8", P.DUMPH)):
                    z = load(root, d, "%s_s%d" % (m, s), f)
                    assert np.array_equal(z["test_y"], yt), "라벨 불일치"
                    acc[m]["f"].append(P.f1(yt, z["test_" + key])); acc[m]["a"].append(P.auroc(yt, z["test_" + key]))
            for m in acc:
                for q in ("f", "a"):
                    V[m][q].append(float(np.mean(acc[m][q])))
        V = {k: {q: np.array(v) for q, v in d.items()} for k, d in V.items()}
        print("\n  정의 = %s" % key)
        for k, lab in (("R", "천장 R(투영 0)"), ("zf", "zf"), ("md", "ModDrop"), ("g8", "γ8 융합")):
            print("    %-14s macro-F1 %.4f | AUROC %.4f" % (lab, V[k]["f"].mean(), V[k]["a"].mean()))
        res = {}
        for name, a, b in (("붕괴 R − zf", "R", "zf"), ("γ8 − zf", "g8", "zf"), ("ModDrop − zf", "md", "zf"), ("γ8 − ModDrop", "g8", "md")):
            x, y = P.ci(V[a]["f"] - V[b]["f"]), P.ci(V[a]["a"] - V[b]["a"]); res[name] = {"f1": x, "auroc": y}
            print("    %-14s macro-F1 %+.4f [%+.4f,%+.4f] %-5s | AUROC %+.4f [%+.4f,%+.4f] %s" % (
                name, *x, P.verdict(x[1], x[2]), *y, P.verdict(y[1], y[2])))
        out[key] = {"P": res, "means": {k: {q: float(v.mean()) for q, v in d.items()} for k, d in V.items()}}
    return out


def part_a(root):
    """보조 분류기 혼합 추론(pad_v4_mai.run)을 meta 셀만 투영 0으로 바꿔 다시 실행. image 셀은 그대로(주 판정 재현 확인용)"""
    print("\n" + "=" * 104); print("[A] 보조 분류기 혼합 추론 — meta 셀을 투영 0으로 바꿔 재실행 (image 셀은 사전 고정 그대로)"); print("=" * 104)
    orig = P.CASES
    P.CASES = (orig[0], ("meta", "meta_missing_proj0", "headi", "onlyimage_s0", "meta_missing_proj0"))
    try:
        r = P.run(root)
    finally:
        P.CASES = orig
    return r


def main(root="."):
    print("PAD meta 결손 정의 민감도 (사후) | %s" % time.strftime("%Y-%m-%d %H:%M"))
    b = part_b(root); a = part_a(root)
    res = {"note": "사후 민감도, 사전 판정 불변", "B_collapse_presc": b,
           "A_blend_meta_proj0": None if a is None else {"meta": a["cells"]["meta"], "image_verdict_reproduced": a["verdict"]}}
    return res


def selftest():
    """가짜 로짓: (1) 두 정의가 같으면 [B]의 두 결과가 같아야 함 (2) 끝까지 실행되어야 함"""
    import tempfile, shutil
    ok = []
    for same in (True, False):
        root = tempfile.mkdtemp(prefix="padsens_")
        os.makedirs(os.path.join(root, P.DUMP)); os.makedirs(os.path.join(root, P.DUMPH))
        r0 = np.random.RandomState(5)
        for f in P.FOLDS:
            yv = r0.choice(6, 300, p=[.3, .35, .03, .12, .1, .1]); yt = r0.choice(6, 460, p=[.3, .35, .03, .12, .1, .1])
            def lg(y, mu, rng): return mu * np.eye(6)[y] + rng.randn(len(y), 6)
            def save(d, tag, V, T):
                arr = {"val_y": yv, "test_y": yt}
                for k in V:
                    arr["val_" + k], arr["test_" + k] = V[k], T[k]
                np.savez_compressed(os.path.join(root, d, "%s_f%d.npz" % (tag, f)), **arr)
            for s in P.SEEDS:
                rng = np.random.RandomState(100 * f + s)
                w = (lg(yv, 0.8, rng), lg(yt, 0.8, rng)); w2 = w if same else (lg(yv, 1.5, rng), lg(yt, 1.5, rng))
                mk = lambda i: {"image_missing": w[i], "meta_missing_input0": w[i], "meta_missing_proj0": w2[i], "full": w[i]}
                for m in ("zf", "md"):
                    save(P.DUMP, "%s_s%d" % (m, s), mk(0), mk(1))
                h = (lg(yv, 2.0, rng), lg(yt, 2.0, rng))
                save(P.DUMPH, "g8_s%d" % s, dict(mk(0), headm=h[0], headi=h[0]), dict(mk(1), headm=h[1], headi=h[1]))
            rng = np.random.RandomState(999 + f); st = (lg(yv, 2.5, rng), lg(yt, 2.5, rng))
            save(P.DUMP, "onlymeta_s0", {"image_missing": st[0]}, {"image_missing": st[1]})
            save(P.DUMP, "onlyimage_s0", {"meta_missing_proj0": st[0]}, {"meta_missing_proj0": st[1]})
        r = main(root); shutil.rmtree(root, ignore_errors=True)
        b = r["B_collapse_presc"]
        eq = all(np.allclose(b[KEYS[0]]["P"][n]["f1"], b[KEYS[1]]["P"][n]["f1"]) for n in b[KEYS[0]]["P"])
        good = (eq if same else not eq) and r["A_blend_meta_proj0"] is not None
        ok.append(good); print("[자체 시험 same=%s] 두 정의 결과 같음=%s → %s" % (same, eq, "통과" if good else "★실패"))
    print("\n자체 시험 판정:", "모두 통과" if all(ok) else "★실패"); return all(ok)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    r = main(".")
    out = OUT if not os.path.exists(OUT) else OUT.replace(".json", "_%s.json" % time.strftime("%Y%m%d%H%M"))
    json.dump(r, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float); print("\n결과 저장:", out)
    print("=== ALL DONE (PAD meta 정의 민감도) ===")
