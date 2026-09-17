"""v14 Stage B 집계 — SigLIP fine-tune(이미지만 교체) full/text결손, CLIP-FT 천장과 대조.
읽는 파일: results/v14b_siglip_{none,gGG}_s*.json (train_v12.py --backbone siglip 산출)"""
import glob, json, re
from collections import defaultdict


def stats(v):
    n = len(v); m = sum(v) / n
    s = (sum((x - m) ** 2 for x in v) / n) ** 0.5 if n > 1 else 0.0
    return m, s


g = defaultdict(list)
for f in glob.glob("results/v14b_*_s*.json"):
    g[re.sub(r"_s\d+\.json$", "", f)].append(json.load(open(f)))

print("== SigLIP-FT (이미지만 SigLIP, 텍스트 CLIP) — full / text결손 macro/micro ==")
for key in sorted(g):
    r = g[key]; name = key.split("results/v14b_")[-1]
    fm = stats([x["eval"]["full"]["macro"] for x in r])
    tm = stats([x["eval"]["text"]["macro"] for x in r])
    tmi = stats([x["eval"]["text"]["micro"] for x in r])
    print("  {:16s} | full {:.3f}±{:.3f} | text {:.3f}±{:.3f} | text_micro {:.3f}".format(
        name, fm[0], fm[1], tm[0], tm[1], tmi[0]))

print("\n  기준(CLIP-B/32 FT, v13): text결손 γ8=0.446 / γ16=0.468 / 포화천장≈0.469")
print("  판정: SigLIP text결손이 0.469를 유의하게 넘으면 → 천장 돌파 확정(이미지 표현이 병목이었음)")
