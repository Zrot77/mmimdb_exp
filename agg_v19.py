"""v19 TLP 집계 — η별 γ8(baseline) vs γ8+TLP, seed 평균±std, K별.
읽는 파일: results/v19_tlp_{missing}_K{K}_s*.json
게이트: TLP_miss가 baseline_miss(γ8)를 seed std 이상 넘고 comp(complete)가 안 깎이면 통과."""
import glob, json
from collections import defaultdict


def stats(v):
    v = [x for x in v if x == x]  # NaN 제거
    if not v:
        return float("nan"), 0.0
    n = len(v); m = sum(v) / n
    s = (sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return m, s


for missing in ("text", "image", "both"):
    for K in (1, 5):
        files = sorted(glob.glob("results/v19_tlp_%s_K%d_s*.json" % (missing, K)))
        if not files:
            continue
        agg = defaultdict(lambda: defaultdict(list))
        for f in files:
            for eta, d in json.load(open(f))["results"].items():
                for k, v in d.items():
                    agg[eta][k].append(v)
        print("\n== TLP | missing={} | K={} | seeds={} ==".format(missing, K, len(files)))
        print("  {:8s} | miss결손 base→TLP (Δ, ±std)        | all base→TLP    | comp base→TLP".format("η"))
        for eta in sorted(agg):
            a = agg[eta]
            bm, bms = stats(a["baseline_miss"]); tm, tms = stats(a["tlp_miss"])
            ba, _ = stats(a["baseline_all"]); ta, _ = stats(a["tlp_all"])
            bc, _ = stats(a["baseline_comp"]); tc, _ = stats(a["tlp_comp"])
            print("  {:8s} | {:.3f}→{:.3f} (Δ{:+.3f}, ±{:.3f}) | {:.3f}→{:.3f} | {:.3f}→{:.3f}".format(
                eta, bm, tm, tm - bm, (bms ** 2 + tms ** 2) ** 0.5, ba, ta, bc, tc))
        print("  게이트: miss결손 Δ가 ±std 이상 양수 + comp 안 깎임 → 통과 / Δ 신뢰구간 0 포함 → 기각(TLP도 γ 못 넘음)")
