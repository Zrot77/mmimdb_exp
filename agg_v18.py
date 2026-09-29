"""v18 B1 게이트 집계 — γ 단독(포화) vs B1 vs γ+B1.
1차 게이트: text결손이 γ를 seed std 이상 유의하게 넘는가 + suff(이미지)가 γ보다 높은가.
읽는 파일:
  results/v14b_siglip_g8.0_s*.json  (γ8 단독 = 기준, 재사용)
  results/v18_b1_l1_s*.json         (B1 단독, λ_B=1)
  results/v18_gb1_l1_s*.json        (γ+B1, γ8·λ_B=1)
"""
import glob, json
from collections import defaultdict


def stats(v):
    n = len(v); mu = sum(v) / n
    sd = (sum((x - mu) ** 2 for x in v) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return mu, sd


def load(pat, path):
    out = []
    for f in glob.glob(pat):
        r = json.load(open(f))
        cur = r
        for k in path:
            cur = cur[k]
        out.append(cur)
    return out


ROWS = [("γ8 단독(기준)", "results/v14b_siglip_g8.0_s*.json"),
        ("B1 단독(λ1)", "results/v18_b1_l1_s*.json"),
        ("γ+B1(λ1)", "results/v18_gb1_l1_s*.json")]

print("== v18 B1 게이트: γ 단독 vs B1 vs γ+B1 (SigLIP so400m, γ 포화 γ8) ==")
print("{:16s} {:>14s} {:>14s} {:>14s}".format("조건", "text결손macro", "full macro", "suff(이미지)"))
data = {}
for lab, pat in ROWS:
    t = load(pat, ["eval", "text", "macro"])
    fu = load(pat, ["eval", "full", "macro"])
    si = load(pat, ["suff", "z_image"])
    if not t:
        print("  {:16s} (데이터 없음)".format(lab)); continue
    data[lab] = (stats(t), stats(fu), stats(si))
    tm, fm, sm = data[lab]
    print("  {:16s} {:.3f}±{:.3f}   {:.3f}±{:.3f}   {:.3f}±{:.3f}".format(
        lab, tm[0], tm[1], fm[0], fm[1], sm[0], sm[1]))

if "γ8 단독(기준)" in data:
    g = data["γ8 단독(기준)"]
    print("\n== 게이트 판정 (vs γ8 단독) ==")
    for lab in ("B1 단독(λ1)", "γ+B1(λ1)"):
        if lab not in data:
            continue
        d = data[lab]
        dt = d[0][0] - g[0][0]; ds = d[2][0] - g[2][0]
        print("  {:12s}: text {:+.3f} (γ {:.3f}→{:.3f}) | suff(img) {:+.3f} (γ {:.3f}→{:.3f})".format(
            lab, dt, g[0][0], d[0][0], ds, g[2][0], d[2][0]))
    print("  통과 = text가 γ를 std 이상 넘고 suff(이미지)도 γ보다↑ / 기각 = 신뢰구간 겹침(B1=γ 우회)")
