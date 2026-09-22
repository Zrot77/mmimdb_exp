"""v15 인과 분석 집계 — IB(nuisance 제거)·MMIN(재구성 중복성) probe를 CLIP vs SigLIP 대조.
읽는 파일: results/v15_{clip,siglip}_{ib,mmin}_s*.json (train_v12.py --causal 산출)"""
import glob, json, re
from collections import defaultdict


def stats(v):
    n = len(v); m = sum(v) / n
    s = (sum((x - m) ** 2 for x in v) / n) ** 0.5 if n > 1 else 0.0
    return m, s


def load(method):
    g = {}
    for bb in ("clip", "siglip"):
        fs = glob.glob("results/v15_{}_{}_s*.json".format(bb, method))
        if fs:
            g[bb] = [json.load(open(f)) for f in fs]
    return g


print("== IB 인과 (병목 f*가 라벨 보존하며 nuisance 제거하나) ==")
ib = load("ib")
for bb, r in ib.items():
    c = [x["checks"] for x in r if x.get("checks")]
    if not c:
        continue
    def m(k): return stats([x[k] for x in c])[0]
    print("  [{}] suff_f {:.3f} → suff_f* {:.3f} (라벨 {:+.3f}) | var {:.1f}→{:.1f} (×{:.2f}) | effrank {:.1f}→{:.1f} (×{:.2f}) | text {:.3f}".format(
        bb, m("suff_f"), m("suff_fstar"), m("suff_fstar") - m("suff_f"),
        m("var_f"), m("var_fstar"), m("var_fstar") / max(1e-9, m("var_f")),
        m("effrank_f"), m("effrank_fstar"), m("effrank_fstar") / max(1e-9, m("effrank_f")),
        stats([x["eval"]["text"]["macro"] for x in r])[0]))
print("  가설: SigLIP=라벨 거의 보존인데 var·effrank 크게↓(nuisance 제거) / CLIP=라벨도 손실(뺄 게 적음)")

print("\n== MMIN 인과 (이미지→텍스트 재구성이 새 라벨정보를 주나) ==")
mm = load("mmin")
for bb, r in mm.items():
    c = [x["checks"] for x in r if x.get("checks")]
    if not c:
        continue
    def m(k): return stats([x[k] for x in c])[0]
    print("  [{}] suff_pi {:.3f} | suff_pt(real) {:.3f} | suff_pt_hat(재구성) {:.3f} | suff(pi+pt_hat) {:.3f} | text {:.3f}".format(
        bb, m("suff_pi"), m("suff_pt"), m("suff_pt_hat"), m("suff_pi_plus_pthat"),
        stats([x["eval"]["text"]["macro"] for x in r])[0]))
print("  가설: suff_pt_hat≈suff_pi 이고 suff(pi+pt_hat)≈suff_pi 면 재구성은 중복(새 정보 없음)")
