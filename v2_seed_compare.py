"""추가 실험 v2 — MM-IMDb seed 확장 대조: n=3(seed 1·2·3) 결과 json 대 n=5(seed 1~5) 결과 json (학습 없음, CPU)
  python v2_seed_compare.py              # 서버 ~/Yechan/mmimdb_exp (conda mmimdb)
  python v2_seed_compare.py --selftest   # 가짜 json 대조 시험
[고정 규칙 — 실행 전, 2026-11-10]
- 대상: 분석 5종(E2·E9, MD, GM, H, 견고성) × 백본 2종 = 10쌍. 기존 파일은 읽기만 함.
- 두 json을 같은 경로로 함께 내려가며 (평균, 하한, 상한) 3원소 CI를 찾아 판정(개선/구분 불가/악화)을 다시 계산하고,
  판정이 바뀐 항목을 모두 '★seed 확장 시 변경'으로 출력함. 문자열·참거짓 판정 값(주 판정, E4 셀 일치 등)도 바뀌면 출력함.
- 메움 비율·기준값 층 비율·R3 값은 판정이 아니라 기술값이므로 양쪽 값을 나란히 출력만 함.
- 한쪽에만 있는 경로는 따로 개수와 이름을 출력함(누락 검사).
- 해석 규칙(사전 고정, 문서 머리말과 같음): n=3 판정은 그대로 두고 n=5 결과를 함께 보고, 바뀐 판정은 '★seed 확장 시 변경'으로 표시함.
"""
import json, os, sys

D = "results/thesis_round/V2"
PAIRS = [("E2·E9", "mmimdb_v2_%s"), ("E9-MD", "mmimdb_v2_moddrop_%s"), ("E9-GM", "mmimdb_v2_gm_%s"),
         ("E9-H", "mmimdb_v2_heads_%s"), ("견고성", "mmimdb_v2_robust_%s")]
DESC_KEYS = ("ratio", "thr_share", "R3")


def verdict(lo, hi):
    return "개선" if lo > 0 else ("악화" if hi < 0 else "구분 불가")


def is_ci(x):
    return (isinstance(x, list) and len(x) == 3 and all(isinstance(v, float) for v in x)
            and x[1] <= x[0] + 1e-12 and x[0] <= x[2] + 1e-12)


def walk(a, b, path, out):
    if is_ci(a) or is_ci(b):
        if is_ci(a) and is_ci(b):
            out["n_ci"] += 1
            va, vb = verdict(a[1], a[2]), verdict(b[1], b[2])
            if va != vb:
                out["changed"].append((path, a, va, b, vb))
        else:
            out["only"].append(path + "  (한쪽만 CI)")
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            p = path + "/" + str(k)
            if k not in a or k not in b:
                out["only"].append(p + ("  (n=5에만)" if k not in a else "  (n=3에만)")); continue
            if any(t in p for t in DESC_KEYS) and isinstance(a[k], (int, float)) and not isinstance(a[k], bool):
                out["desc"].append((p, a[k], b[k]))
            walk(a[k], b[k], p, out)
        return
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and any(isinstance(v, (dict, list)) for v in a):
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, "%s[%d]" % (path, i), out)
        return
    if isinstance(a, (str, bool)) and isinstance(b, (str, bool)):
        out["n_str"] += 1
        if a != b:
            out["str_changed"].append((path, a, b))


def compare(a, b):
    out = {"n_ci": 0, "n_str": 0, "changed": [], "str_changed": [], "desc": [], "only": []}
    walk(a, b, "", out)
    return out


def report(name, out):
    print("  CI %d개, 판정 문자열 %d개 대조 | CI 판정 변경 %d | 문자열 판정 변경 %d | 한쪽에만 %d" % (
        out["n_ci"], out["n_str"], len(out["changed"]), len(out["str_changed"]), len(out["only"])))
    for p, a, va, b, vb in out["changed"]:
        print("  ★seed 확장 시 변경 %s\n      n=3 %+.4f [%+.4f,%+.4f] %s → n=5 %+.4f [%+.4f,%+.4f] %s" % (p, *a, va, *b, vb))
    for p, a, b in out["str_changed"]:
        print("  ★seed 확장 시 변경 %s: n=3 '%s' → n=5 '%s'" % (p, a, b))
    for p, a, b in out["desc"]:
        f = lambda v: ("%.3f" % v) if isinstance(v, float) else str(v)
        print("  (기술) %s: n=3 %s → n=5 %s" % (p, f(a), f(b)))
    for p in out["only"]:
        print("  (누락 검사) %s" % p)


def main():
    print("=" * 100); print("MM-IMDb seed 확장 대조 | n=3 json 대 n=5(_seeds12345) json | 판정 규칙: 하한>0 개선, 상한<0 악화"); print("=" * 100)
    tot = {"ci": 0, "chg": 0, "str": 0, "schg": 0}; res = {}
    for bb in ("clip", "siglip"):
        for name, pat in PAIRS:
            f3 = os.path.join(D, pat % bb + ".json"); f5 = os.path.join(D, pat % bb + "_seeds12345.json")
            print("\n[%s %s] %s ↔ %s" % (bb, name, os.path.basename(f3), os.path.basename(f5)))
            if not (os.path.exists(f3) and os.path.exists(f5)):
                print("  ★파일 없음: n=3 %s, n=5 %s" % (os.path.exists(f3), os.path.exists(f5))); continue
            a, b = json.load(open(f3, encoding="utf-8")), json.load(open(f5, encoding="utf-8"))
            o = compare(a, b); report(name, o)
            tot["ci"] += o["n_ci"]; tot["chg"] += len(o["changed"]); tot["str"] += o["n_str"]; tot["schg"] += len(o["str_changed"])
            res["%s/%s" % (bb, name)] = {"changed": [(p, a_, va, b_, vb) for p, a_, va, b_, vb in o["changed"]],
                                         "str_changed": o["str_changed"], "desc": o["desc"], "only": o["only"]}
    print("\n" + "=" * 100)
    print("합계: CI %d개 중 판정 변경 %d개, 판정 문자열 %d개 중 변경 %d개" % (tot["ci"], tot["chg"], tot["str"], tot["schg"]))
    out = os.path.join(D, "seed_compare_n3_n5.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("결과 저장:", out); print("=== ALL DONE (seed 확장 대조) ===")


def selftest():
    a = {"cells": {"text": {"H1": [0.2, 0.1, 0.3], "H2": {"resid": [0.017, -0.018, 0.051], "thr_share": 0.91},
                            "verdict": "메움", "seeds": [1, 2, 3], "R3": {"300": 0.8}},
                   "image": {"g8": [-0.026, -0.04, -0.01], "x": [0.01, 0.0, 0.02]}}}
    b = {"cells": {"text": {"H1": [0.21, 0.19, 0.24], "H2": {"resid": [0.018, 0.006, 0.030], "thr_share": 0.90},
                            "verdict": "메움", "seeds": [1, 2, 3, 4, 5], "R3": {"300": 0.82}},
                   "image": {"g8": [-0.017, -0.045, 0.010], "x": [0.01, 0.001, 0.02]}, "full": {"y": [0.0, -0.1, 0.1]}}}
    o = compare(a, b)
    paths = sorted(p for p, *_ in o["changed"])
    ok = [o["n_ci"] == 4,                                                   # H1·resid·g8·x (seed 목록은 CI 아님)
          paths == ["/cells/image/g8", "/cells/image/x", "/cells/text/H2/resid"],   # x: 하한 0.0(구분 불가) → 0.001(개선)
          o["str_changed"] == [],
          any(p.endswith("/full  (n=5에만)") for p in o["only"]),
          any(p == "/cells/text/H2/thr_share" for p, *_ in o["desc"]),
          any(p == "/cells/text/R3/300" for p, *_ in o["desc"])]
    o2 = compare({"v": "가"}, {"v": "나"}); ok.append(len(o2["str_changed"]) == 1)
    o3 = compare(a, a); ok.append(not o3["changed"] and not o3["str_changed"] and not o3["only"])
    print("자체 시험:", ok); print("자체 시험 판정:", "모두 통과" if all(ok) else "★실패")
    return all(ok)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    main()
