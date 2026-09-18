"""v14c 집계 — SigLIP-FT 위 6종 비교 (text결손 macro 순위), v12 CLIP 순위와 대조.
읽는 파일: results/v14b_siglip_{none,g8.0}_s*.json (재활용) + results/v14c_siglip_{moddrop,mmin,kd,ib,gm}_s*.json
"""
import glob, json, re
from collections import defaultdict


def stats(v):
    n = len(v); m = sum(v) / n
    s = (sum((x - m) ** 2 for x in v) / n) ** 0.5 if n > 1 else 0.0
    return m, s


files = (glob.glob("results/v14b_siglip_none_s*.json")
         + glob.glob("results/v14b_siglip_g8.0_s*.json")
         + glob.glob("results/v14c_siglip_*_s*.json"))
g = defaultdict(list)
for f in files:
    name = re.search(r"v14[bc]_siglip_(.+?)_s\d+\.json", f).group(1)
    g[name].append(json.load(open(f)))

label = {"none": "zerofill", "g8.0": "gamma(γ8)", "moddrop": "moddrop",
         "mmin": "mmin", "kd": "kd", "ib": "ib", "gm": "gm(γ8+MD)"}
rows = []
for k, r in g.items():
    tm = stats([x["eval"]["text"]["macro"] for x in r])
    fm = stats([x["eval"]["full"]["macro"] for x in r])
    rows.append((label.get(k, k), tm, fm, len(r)))
rows.sort(key=lambda x: -x[1][0])

print("== SigLIP-FT 6종 비교 (text결손 macro 내림차순, 3seed) ==")
for name, tm, fm, n in rows:
    print("  {:12s} | text {:.3f}±{:.3f} | full {:.3f}±{:.3f} | n={}".format(name, tm[0], tm[1], fm[0], fm[1], n))
print("\n  [대조] CLIP-B/32(v12, γ2) 순위: gm 0.437 > kd 0.394 > gamma 0.384 ≈ moddrop 0.383 > mmin 0.366 > ib 0.328 > zerofill 0.256")
print("  주의: SigLIP의 gamma/gm/ib는 γ8(v13 교훈 반영), v12는 γ2 — gamma 계열 절대값은 직접 비교 말고 '순위·구도'로 읽을 것")
