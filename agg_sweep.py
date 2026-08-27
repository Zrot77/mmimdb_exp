"""v7 파인튜닝량 스윕 집계 — 레벨(L0~L3)×조건(supervised/infonce/infonce_noise) suff(z_image)
+ 누수 + noise 회복 분율 + distortion(freeze 0.378 대비) + 과적합 격차.
파일명 규약: results/ft_{L0..L3}_{supervised|infonce|infonce_noise}_s{seed}.json
실행: python agg_sweep.py"""
import glob
import json
import os
import re
import collections
import statistics as st

FREEZE_IMG = 0.378   # 2주차 freeze판 정렬후 이미지 suff (distortion 기준선)
LEVELS = ["L0", "L1", "L2", "L3"]
OBJS = ["supervised", "infonce", "infonce_noise"]

# (level,obj) -> {test:[...], val:[...], overfit:[...]}
data = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(os.path.join("results", "ft_*_s*.json")):
    m = re.match(r"ft_(L\d)_(supervised|infonce_noise|infonce)_s\d+\.json$", os.path.basename(f))
    if not m:
        continue
    lv, ob = m.group(1), m.group(2)
    s = json.load(open(f, encoding="utf-8"))["suff_z_image"]
    data[(lv, ob)]["test"].append(s["test"])
    data[(lv, ob)]["val"].append(s.get("val", s["test"]))
    data[(lv, ob)]["overfit"].append(s.get("overfit_gap", 0.0))


def mean(a):
    return sum(a) / len(a) if a else float("nan")


def sd(a):
    return st.pstdev(a) if len(a) > 1 else 0.0


print("\n=== v7 파인튜닝량 스윕: suff(z_image) test (mean±std) ===")
hdr = "  lvl  " + " ".join("{:>16s}".format(o) for o in OBJS) + "   누수   회복분율  distortion"
print(hdr)
for lv in LEVELS:
    cells, vals = [], {}
    for ob in OBJS:
        t = data[(lv, ob)]["test"]
        vals[ob] = mean(t) if t else None
        cells.append("{:.3f}±{:.3f}".format(mean(t), sd(t)) if t else "   -   ")
    line = "  {:4s} ".format(lv) + " ".join("{:>16s}".format(c) for c in cells)
    if vals.get("supervised") is not None and vals.get("infonce") is not None:
        leak = vals["supervised"] - vals["infonce"]
        frac = ((vals["infonce_noise"] - vals["infonce"]) / leak
                if vals.get("infonce_noise") is not None and abs(leak) > 1e-6 else float("nan"))
        dist = FREEZE_IMG - vals["infonce"]   # +면 파인튜닝이 freeze보다 더 깎음
        line += "  {:+.3f}   {:>6}   {:+.3f}".format(
            leak, "{:.0f}%".format(100 * frac) if frac == frac else "  -", dist)
    print(line)

print("\n과적합 격차 (train−val, 클수록 과적합; 큰 레벨은 판정에서 주의):")
for lv in LEVELS:
    for ob in OBJS:
        o = data[(lv, ob)]["overfit"]
        if o:
            print("  {} {:15s}: {:+.3f}".format(lv, ob, mean(o)))

print("\n판정 가이드:")
print("  회복분율이 L0→L3로 ~100%에 접근 → 가설A(shortcut 전이, 학습량 관건).")
print("  회복분율이 계속 낮음(~40%↓ 유지) → 가설B(distortion regime, γ 처방).")
print("  누수↑ & infonce가 freeze(0.378) 아래로(distortion +) → 가설B 추가 증거.")
print("  단, 과적합 격차 큰 레벨의 값은 신뢰 낮음(val 기준으로 재판단).")
