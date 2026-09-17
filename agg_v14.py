"""v14 Stage B 집계 — CLIP-frozen vs SigLIP-mix(이미지만 교체), presc/γ별 full·text결손.
읽는 파일: results/v14_{clip,siglip}_{none,gGG}_s*.json (train_v9.py 산출)
핵심 질문: SigLIP 이미지가 text결손을 CLIP 천장(≈0.469)/CLIP-frozen 통제 위로 올리는가."""
import glob, json, re
from collections import defaultdict


def stats(v):
    n = len(v); m = sum(v) / n
    s = (sum((x - m) ** 2 for x in v) / n) ** 0.5 if n > 1 else 0.0
    return m, s


g = defaultdict(list)
for f in glob.glob("results/v14_*_s*.json"):
    g[re.sub(r"_s\d+\.json$", "", f)].append(json.load(open(f)))

print("{:24s} | full macro   | text macro   | text micro".format("조건(인코더_처방)"))
for key in sorted(g):
    r = g[key]; name = key.split("results/v14_")[-1]
    fm = stats([x["eval"]["full"]["macro"] for x in r])
    tm = stats([x["eval"]["text"]["macro"] for x in r])
    tmi = stats([x["eval"]["text"]["micro"] for x in r])
    print("{:24s} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:.3f}±{:.3f}".format(
        name, fm[0], fm[1], tm[0], tm[1], tmi[0], tmi[1]))
print("\n  기준선: γ CLIP-B/32 FT text결손 천장 ≈ 0.469 / CLIP frozen 선형 이미지 0.407 / SigLIP frozen 선형 0.545")
print("  판정: siglip_* text macro가 clip_* 및 0.469를 넘으면 → 천장 돌파 파이프라인 확인")
