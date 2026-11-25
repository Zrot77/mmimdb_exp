"""심사 피드백 대응 확인 — 학습 없음, 집계 숫자만 출력 (서버 mmimdb_exp 폴더에서 실행)
  python review_check.py
  (1) FINAL/tables.json의 결손 없는(full) 조건 평균: zf·ModDrop·γ8·γ+ModDrop의 F1@0.5(f), 재보정 F1(x), AUPRC(a)
  (2) R1 부트스트랩: seed 기준 CI 폭과 부트스트랩 CI 폭 비교 (구간 폭 비교가 없다는 지적 대응)
  (3) γ 격자(v12·v13) 실험의 로짓 저장 여부 (그림 4.3을 재보정 후 값으로 다시 그릴 수 있는지)
"""
import glob, json, os, sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
R = "results/thesis_round"


def walk(o, path, hit):
    if isinstance(o, dict):
        hit(path, o)
        for k, v in o.items():
            walk(v, path + [str(k)], hit)
    elif isinstance(o, list):
        for i, v in enumerate(o[:50]):
            walk(v, path + ["[%d]" % i], hit)


def part1():
    p = os.path.join(R, "FINAL", "tables.json")
    print("=" * 80); print("(1) full 조건:", p); print("=" * 80)
    if not os.path.exists(p):
        print("없음"); return
    def hit(path, d):
        if "means" in d and isinstance(d["means"], dict) and "zf" in d["means"]:
            print("/".join(path))
            for k, v in d["means"].items():
                print("   %-4s  F1@0.5 %.4f   재보정 F1 %.4f   AUPRC %.4f" % (k, v.get("f", float("nan")), v.get("x", float("nan")), v.get("a", float("nan"))))
            for k, v in d.get("cmp", {}).items():
                print("   %s : %s" % (k, json.dumps(v, ensure_ascii=False)))
            if "gm_vs_g8" in d:
                print("   결합 − γ8 :", json.dumps(d["gm_vs_g8"], ensure_ascii=False))
    walk(json.load(open(p, encoding="utf-8")), [], hit)


def part2():
    fs = sorted(set(glob.glob(os.path.join(R, "**", "*robust*.json"), recursive=True)))
    print("=" * 80); print("(2) 부트스트랩 대 seed CI 폭:", fs); print("=" * 80)
    for p in fs:
        rows = []
        def hit(path, d):
            s, b = d.get("seed"), d.get("boot")
            if isinstance(b, (list, tuple)) and len(b) == 3:
                try:
                    if isinstance(s, dict):
                        lo, hi = s.get("lo", s.get("ci_lo")), s.get("hi", s.get("ci_hi"))
                    else:
                        lo, hi = s[1], s[2]
                    rows.append(("/".join(path)[-70:], float(hi) - float(lo), float(b[2]) - float(b[1])))
                except Exception:
                    rows.append(("/".join(path)[-70:] + "  (seed 형식: %s)" % type(s).__name__, float("nan"), float(b[2]) - float(b[1])))
        walk(json.load(open(p, encoding="utf-8")), [], hit)
        print(p, "비교 %d개" % len(rows))
        for n, ws, wb in rows:
            print("   %-72s seed폭 %.4f  boot폭 %.4f  비 %.2f" % (n, ws, wb, wb / ws if ws == ws and ws > 0 else float("nan")))
        ok = [r for r in rows if r[1] == r[1] and r[1] > 0]
        if ok:
            k = sum(1 for r in ok if r[2] < r[1])
            print("   → boot 폭 < seed 폭: %d / %d" % (k, len(ok)))


def part3():
    print("=" * 80); print("(3) γ 격자 로짓 저장 여부"); print("=" * 80)
    pats = ["results/**/*v13*.npz", "results/**/*v12*.npz", "results/**/*gamma*.npz", "results/**/*_g16*.npz", "results/**/*g64*.npz"]
    fs = sorted(set(f for q in pats for f in glob.glob(q, recursive=True)))
    print("npz %d개" % len(fs))
    for f in fs[:15]:
        print("  ", f)
    js = sorted(set(glob.glob("results/**/*v13*.json", recursive=True)))
    print("v13 json %d개, 첫 파일의 최상위 키:" % len(js))
    if js:
        d = json.load(open(js[0], encoding="utf-8"))
        print("  ", js[0]); print("  ", list(d.keys())[:40] if isinstance(d, dict) else type(d).__name__)


if __name__ == "__main__":
    part1(); part2(); part3()
    print("\n=== ALL DONE (review_check) ===")
