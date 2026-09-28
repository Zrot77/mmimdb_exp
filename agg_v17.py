"""v17 — 2×2 factorial interaction (γ × ModDrop), 새 학습 없이 기존 결과 재사용.
seed-paired 계산: 각 seed에서 interaction_s = gm − gamma − moddrop + zerofill, 이후 seed 평균±std + t.
두 factorial:
  CLIP γ2   : v13_key_{zerofill,gamma,moddrop,gm}_s*.json  (gamma·gm = γ2)
  SigLIP γ8 : v14b_siglip_{none,g8.0}_s*.json + v14c_siglip_{moddrop,gm}_s*.json
값 없는 조건은 '데이터 없음'.
"""
import glob, json, re, os
from collections import defaultdict


def load_by_seed(pattern, metric_path):
    """pattern 매칭 json에서 seed→값. metric_path = (regime, metric)."""
    out = {}
    for f in glob.glob(pattern):
        m = re.search(r"_s(\d+)\.json$", f)
        if not m:
            continue
        r = json.load(open(f))
        reg, met = metric_path
        out[int(m.group(1))] = r["eval"][reg][met]
    return out


def stats(v):
    n = len(v); mu = sum(v) / n
    sd = (sum((x - mu) ** 2 for x in v) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return mu, sd, n


def factorial(name, globs, reg="text", met="macro"):
    z = load_by_seed(globs["zerofill"], (reg, met))
    g = load_by_seed(globs["gamma"], (reg, met))
    m = load_by_seed(globs["moddrop"], (reg, met))
    gm = load_by_seed(globs["gm"], (reg, met))
    seeds = sorted(set(z) & set(g) & set(m) & set(gm))
    print("\n== {} | {}결손 {} | 공통 seed {} ==".format(name, reg, met, seeds))
    if not seeds:
        print("  (데이터 없음 — 4조건 공통 seed 없음)")
        return
    def ms(d): mu, sd, n = stats([d[s] for s in seeds]); return mu, sd
    for lab, d in [("zerofill", z), ("gamma", g), ("moddrop", m), ("gm(γ+MD)", gm)]:
        mu, sd = ms(d)
        print("  {:10s} {:.3f}±{:.3f}".format(lab, mu, sd))
    # seed-paired 대비
    gamma_main = [(g[s] - z[s] + gm[s] - m[s]) / 2 for s in seeds]
    md_main = [(m[s] - z[s] + gm[s] - g[s]) / 2 for s in seeds]
    inter = [gm[s] - g[s] - m[s] + z[s] for s in seeds]
    md_on_gamma = [gm[s] - g[s] for s in seeds]      # γ 위에서 ModDrop 한계효과
    for lab, v in [("γ 주효과", gamma_main), ("ModDrop 주효과", md_main),
                   ("ModDrop|γon(gm−gamma)", md_on_gamma), ("interaction", inter)]:
        mu, sd, n = stats(v)
        se = sd / (n ** 0.5) if n > 0 else 0.0
        t = mu / se if se > 0 else float("nan")
        print("  {:24s} = {:+.3f}  (SE {:.3f}, t {:+.2f})".format(lab, mu, se, t))
    print("  판정: interaction 신뢰구간이 0 포함 → 시너지 없음 / 유의 양수 → 시너지 / 유의 음수 → 잠식(중복)")


CLIP_G2 = {"zerofill": "results/v13_key_zerofill_s*.json", "gamma": "results/v13_key_gamma_s*.json",
           "moddrop": "results/v13_key_moddrop_s*.json", "gm": "results/v13_key_gm_s*.json"}
SIG_G8 = {"zerofill": "results/v14b_siglip_none_s*.json", "gamma": "results/v14b_siglip_g8.0_s*.json",
          "moddrop": "results/v14c_siglip_moddrop_s*.json", "gm": "results/v14c_siglip_gm_s*.json"}

for reg in ("text", "full", "image"):
    factorial("CLIP γ2", CLIP_G2, reg=reg)
    factorial("SigLIP γ8", SIG_G8, reg=reg)
