"""v16 인과 분리 집계 — 표현 강도(frozen probe) vs IB 값어치, SigLIP 패밀리 여러 축.
독립변수 = 각 인코더의 이미지 essence(frozen 지도 선형 probe 천장)
종속변수 = IB 값어치 = IB text결손 − γ text결손 (+ 보조: 병목 분산 제거비 var(f*)/var(f))
핵심 판정 = 여러 축(해상도/크기/목표)에 걸쳐 '표현강도 ↔ IB값어치'가 일관 양의 상관이면 → 교란변수 아니라 강도에 귀속

읽는 파일:
  feats_<tag>/feats_{train,test}.pt       (extract_siglip.py --out feats_<tag>)
  results/v16_<tag>_{gamma,ib}_s*.json    (train_v12.py --backbone siglip --siglip_name ...)
"""
import glob, json, os
from collections import defaultdict

import torch

from utils import multilabel_probe

BACKBONES = [                       # (tag, 표시라벨, 축)
    ("sig_b224", "base-224", "res"),
    ("sig_b256", "base-256", "res/size"),
    ("sig_b384", "base-384", "res"),
    ("sig_b512", "base-512", "res"),
    ("sig_l256", "large-256", "size"),
    ("sig2_b224", "siglip2-224", "obj"),
]


def stats(v):
    n = len(v); m = sum(v) / n
    s = (sum((x - m) ** 2 for x in v) / n) ** 0.5 if n > 1 else 0.0
    return m, s


def frozen_strength(tag, device):
    d = "feats_%s" % tag
    tr, te = os.path.join(d, "feats_train.pt"), os.path.join(d, "feats_test.pt")
    if not (os.path.exists(tr) and os.path.exists(te)):
        return None
    a = torch.load(tr, map_location="cpu"); b = torch.load(te, map_location="cpu")
    vals = [multilabel_probe(a["img"].float(), a["y"].float(), b["img"].float(), b["y"].float(),
                             device=device, seed=s)["f1_macro"] for s in range(3)]
    return sum(vals) / len(vals)


def method_text(tag, method):
    fs = glob.glob("results/v16_%s_%s_s*.json" % (tag, method))
    if not fs:
        return None, []
    runs = [json.load(open(f)) for f in fs]
    return stats([r["eval"]["text"]["macro"] for r in runs])[0], runs


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys)) ** 0.5
    return cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    def fmt(x):
        return "%.3f" % x if isinstance(x, float) else "  -  "

    print("== v16 인과 분리: 표현강도(frozen) vs IB 값어치 ==")
    print("{:14s} {:>6s} {:>8s} {:>7s} {:>7s} {:>9s} {:>10s}".format(
        "backbone", "축", "표현강도", "γ_text", "IB_text", "IB값어치", "var비f*/f"))
    for tag, label, axis in BACKBONES:
        strg = frozen_strength(tag, device)
        g, _ = method_text(tag, "gamma")
        ib, ibruns = method_text(tag, "ib")
        if strg is None and g is None and ib is None:
            continue
        ibval = (ib - g) if (ib is not None and g is not None) else None
        varr = None
        if ibruns:
            cs = [r["checks"] for r in ibruns if r.get("checks") and "var_f" in r["checks"]]
            if cs:
                varr = stats([c["var_fstar"] / max(1e-9, c["var_f"]) for c in cs])[0]
        rows.append((label, axis, strg, g, ib, ibval, varr))
        print("{:14s} {:>6s} {:>8s} {:>7s} {:>7s} {:>9s} {:>10s}".format(
            label, axis, fmt(strg), fmt(g), fmt(ib),
            fmt(ibval) if ibval is not None else "  -  ", fmt(varr) if varr is not None else "  -  "))

    pts = [(r[2], r[5]) for r in rows if isinstance(r[2], float) and isinstance(r[5], float)]
    if len(pts) >= 2:
        r = pearson([p[0] for p in pts], [p[1] for p in pts])
        print("\n  표현강도 ↔ IB값어치(IB−γ) Pearson r = %.3f (n=%d)" % (r, len(pts)))
        print("  해석: 여러 축에 걸쳐 강한 양(+)이면 → IB값어치는 해상도/크기/목표가 아니라 표현 강도에 귀속(주장 지지)")
        print("        축마다 강도-값어치가 어긋나거나 특정 요인에서만 오르면 → 그 요인이 진짜 원인(주장 수정)")
    else:
        print("\n  (상관엔 표현강도+IB값어치 둘 다 있는 백본 2개↑ 필요 — 아직 데이터 부족)")


if __name__ == "__main__":
    main()
