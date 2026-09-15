"""v13 H2 — 샘플 분해 (겹침 vs 고유).
text결손 테스트셋에서 baseline(zerofill) 대비 sample-F1이 개선된 샘플로 γ·ModDrop·gm 집합을 만들고
(a)γ만/(b)MD만/(c)둘다/(d)둘다아님 분해 + Jaccard 겹침 + gm 흡수율 계산.
읽는 파일: results/preds_{zerofill,gamma,moddrop,gm}_s*.pt  (train_v12.py --save_preds 산출)
"""
import glob, re, statistics as st
from collections import defaultdict

import torch


def sample_f1(pred, y):
    pred = pred.float(); y = y.float()
    tp = (pred * y).sum(1); fp = (pred * (1 - y)).sum(1); fn = ((1 - pred) * y).sum(1)
    return 2 * tp / (2 * tp + fp + fn + 1e-9)


def ms(v):
    return (sum(v) / len(v), (st.pstdev(v) if len(v) > 1 else 0.0))


conds = ["zerofill", "gamma", "moddrop", "gm"]
seeds = sorted(set(int(re.search(r"_s(\d+)\.pt", f).group(1)) for f in glob.glob("results/preds_*_s*.pt")))
agg = defaultdict(list)

for s in seeds:
    try:
        d = {c: torch.load(f"results/preds_{c}_s{s}.pt") for c in conds}
    except FileNotFoundError:
        print(f"  (seed {s} 파일 누락 — 스킵)"); continue
    f1 = {c: sample_f1(d[c]["pred_text"], d[c]["y"]) for c in conds}
    base = f1["zerofill"]
    g = f1["gamma"] > base + 1e-9
    m = f1["moddrop"] > base + 1e-9
    gmi = f1["gm"] > base + 1e-9
    a = int((g & ~m).sum()); b = int((m & ~g).sum()); c = int((g & m).sum()); dd = int((~g & ~m).sum())
    union = g | m
    jac = c / max(1, a + b + c)
    absorb = float((union & gmi).sum()) / max(1, int(union.sum()))
    for k, v in dict(a=a, b=b, c=c, d=dd, jac=jac, absorb=absorb).items():
        agg[k].append(v)

print("== H2 샘플 분해 (text결손, baseline 대비 sample-F1 개선, {}seed 평균) ==".format(len(seeds)))
print(f"  (a) γ만 개선   : {ms(agg['a'])[0]:.0f}")
print(f"  (b) ModDrop만  : {ms(agg['b'])[0]:.0f}")
print(f"  (c) 둘 다 개선 : {ms(agg['c'])[0]:.0f}")
print(f"  (d) 둘 다 아님 : {ms(agg['d'])[0]:.0f}")
print(f"  Jaccard 겹침  c/(a+b+c) = {ms(agg['jac'])[0]:.3f}  (낮음=서로 다른 샘플 개선=상보, 높음=중복)")
print(f"  gm 흡수율  |(γ∪MD)∩gm|/|γ∪MD| = {ms(agg['absorb'])[0]:.3f}  (높음=gm이 둘의 고유 개선을 모두 흡수)")
