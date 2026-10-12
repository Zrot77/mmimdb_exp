"""A0 감사 — 기존 results/*.json을 config 기준으로 묶어 ddof=1로 재계산(정본표·per-seed·seed목록).
새 학습 없음. 출력: results/thesis_round/A0/recompute.csv + 화면 요약.
재현점검(--repro)도 지원: 두 json의 text결손 macro 차이.
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def mean_std(v, ddof=1):
    n = len(v)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(v) / n
    if n - ddof <= 0:
        return m, 0.0
    s = (sum((x - m) ** 2 for x in v) / (n - ddof)) ** 0.5
    return m, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/thesis_round/A0/recompute.csv")
    ap.add_argument("--repro", nargs=2, default=None,
                    help="재현점검: 새json 기존json → text결손 macro Δ")
    args = ap.parse_args()

    if args.repro:
        a = json.load(open(args.repro[0]))["eval"]["text"]["macro"]
        b = json.load(open(args.repro[1]))["eval"]["text"]["macro"]
        print("재현점검 text결손 macro: 새 {:.4f} vs 기존 {:.4f} | |Δ|={:.4f} → {}".format(
            a, b, abs(a - b), "통과(≤0.010)" if abs(a - b) <= 0.010 else "★초과(중단·보고)"))
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    groups = defaultdict(dict)        # key -> {seed: metrics}
    all_seeds = set()
    for f in sorted(glob.glob(os.path.join(args.results, "*.json"))):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if "eval" not in r or "config" not in r:
            continue
        c = r["config"]
        key = (c.get("backbone"), os.path.basename(str(c.get("siglip_name", "-"))),
               c.get("method"), c.get("gamma"), c.get("moddrop_p"), c.get("lam_b"))
        seed = c.get("seed")
        all_seeds.add(seed)
        ev = r.get("eval", {})
        m = {}
        for reg in ("full", "text", "image", "both"):
            e = ev.get(reg)
            if isinstance(e, dict) and "macro" in e:
                m[reg] = e["macro"]
        if "text" not in m:                       # 이 포맷은 건너뜀(구포맷/비교대상 아님)
            continue
        if "suff" in r:
            m["suff_img"] = r["suff"].get("z_image"); m["suff_txt"] = r["suff"].get("z_text")
        groups[key][seed] = m

    print("발견 seed 목록:", sorted(s for s in all_seeds if s is not None))
    rows = []
    hdr = ["backbone", "siglip", "method", "gamma", "moddrop_p", "lam_b", "n_seed",
           "text_mean", "text_std", "full_mean", "image_mean", "suffimg_mean", "seeds"]
    print("\n{:7s} {:22s} {:8s} {:>5s} {:>4s} {:>2s} | text(mean±std) full image suffimg".format(
        "bb", "siglip", "method", "γ", "p", "n"))
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        bb, sg, me, ga, mp, lb = key
        d = groups[key]
        seeds = sorted(s for s in d if s is not None)
        def col(name):
            return [d[s][name] for s in seeds if name in d[s]]
        tm, ts = mean_std(col("text")); fm, _ = mean_std(col("full"))
        im, _ = mean_std(col("image")); sm, _ = mean_std(col("suff_img"))
        rows.append([bb, sg, me, ga, mp, lb, len(seeds), tm, ts, fm, im, sm, ",".join(map(str, seeds))])
        print("{:7s} {:22s} {:8s} {:>5s} {:>4s} {:>2d} | {:.3f}±{:.3f}  {:.3f} {:.3f} {}".format(
            str(bb), str(sg)[:22], str(me), str(ga), str(mp), len(seeds),
            tm, ts, fm, im, ("%.3f" % sm) if sm == sm else "-"))
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(hdr); w.writerows(rows)
    print("\n>>> saved", args.out, "| ddof=1 재계산. 문서값과 0.002 초과 차이는 정본표에서 mismatch로 표기")


if __name__ == "__main__":
    main()
