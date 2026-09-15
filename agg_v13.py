"""v13 집계 — H1(정규화 총량 대조: γ·p 스윕 vs gm) + H3(표현 suff vs 사용 usage_share).
읽는 파일:
  results/v13_gamma_g*_s*.json    (H1 γ 스윕)
  results/v13_moddrop_p*_s*.json  (H1 p 스윕)
  results/v13_key_{zerofill,gamma,moddrop,gm}_s*.json  (H3 표 + gm 앵커)
"""
import glob, json, re
from collections import defaultdict


def stats(v):
    n = len(v); m = sum(v) / n
    s = (sum((x - m) ** 2 for x in v) / n) ** 0.5 if n > 1 else 0.0
    return m, s


def load_group(pattern, keyfn):
    g = defaultdict(list)
    for f in glob.glob(pattern):
        g[keyfn(f)].append(json.load(open(f)))
    return g


# ---------- H1: γ 스윕 ----------
print("== H1  γ 단독 스윕 (text결손 macro / full macro, 3seed) ==")
gg = load_group("results/v13_gamma_g*_s*.json", lambda f: float(re.search(r"_g([0-9.]+)_s", f).group(1)))
gbest = None
for gv in sorted(gg):
    r = gg[gv]; tm, ts = stats([x["eval"]["text"]["macro"] for x in r]); fm, fs = stats([x["eval"]["full"]["macro"] for x in r])
    print(f"  γ={gv:<6}: text {tm:.3f}±{ts:.3f} | full {fm:.3f}±{fs:.3f}")
    if gbest is None or tm > gbest[1]:
        gbest = (gv, tm, ts)

print("== H1  ModDrop p 단독 스윕 ==")
pg = load_group("results/v13_moddrop_p*_s*.json", lambda f: float(re.search(r"_p([0-9.]+)_s", f).group(1)))
pbest = None
for pv in sorted(pg):
    r = pg[pv]; tm, ts = stats([x["eval"]["text"]["macro"] for x in r]); fm, fs = stats([x["eval"]["full"]["macro"] for x in r])
    print(f"  p={pv:<6}: text {tm:.3f}±{ts:.3f} | full {fm:.3f}±{fs:.3f}")
    if pbest is None or tm > pbest[1]:
        pbest = (pv, tm, ts)

gm = load_group("results/v13_key_gm_s*.json", lambda f: "gm")
gm_tm = stats([x["eval"]["text"]["macro"] for x in gm["gm"]]) if gm else (float("nan"), 0)

print("\n== H1 판정: 단독 최대 vs gm ==")
if gbest and pbest:
    print(f"  γ 단독 최대: γ={gbest[0]}  text {gbest[1]:.3f}±{gbest[2]:.3f}")
    print(f"  p 단독 최대: p={pbest[0]}  text {pbest[1]:.3f}±{pbest[2]:.3f}")
    print(f"  gm         :          text {gm_tm[0]:.3f}±{gm_tm[1]:.3f}")
    sm = max(gbest[1], pbest[1])
    print(f"  → 단독최대 {sm:.3f} vs gm {gm_tm[0]:.3f} : gm−단독 = {gm_tm[0]-sm:+.3f}")
    print("     (gm이 유의하게 높음 → H1 지지=상보 실재 / 근접 → 환상·부분상보)")

# ---------- H3: 표현 vs 사용 ----------
print("\n== H3  표현(suff) vs 사용(img_usage_share), 3seed ==")
key = load_group("results/v13_key_*_s*.json", lambda f: re.search(r"v13_key_([a-z]+)_s", f).group(1))
print(f"  {'조건':10s} suff_img       suff_txt       img_usage_share")
for k in ["zerofill", "gamma", "moddrop", "gm"]:
    if k not in key:
        continue
    r = key[k]
    si = stats([x["suff"]["z_image"] for x in r]); st = stats([x["suff"]["z_text"] for x in r])
    us = stats([x["usage"]["img_usage_share"] for x in r])
    print(f"  {k:10s} {si[0]:.3f}±{si[1]:.3f}   {st[0]:.3f}±{st[1]:.3f}   {us[0]:.3f}±{us[1]:.3f}")
print("  예측 패턴: γ→suff_img↑·usage 소 / ModDrop→suff 소·usage↑ / gm→둘 다↑")
