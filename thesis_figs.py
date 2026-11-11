"""논문용 그림 생성 (학습 없음, CPU) — 입력은 thesis_tables.py가 만든 FINAL/tables.json과 견고성 json(R3)
  python thesis_figs.py              # 서버 ~/Yechan/mmimdb_exp, thesis_tables.py 실행 후
  python thesis_figs.py --selftest
출력: results/thesis_round/FINAL/figs/fig{1,3,4,5,7}.{png,pdf}  (그림 번호는 문서 7절 그림 목록)
  fig1 층 분해 폭포 막대(CLIP text 결손 대 SigLIP image 결손) / fig3 4셀 기준값 층·순위 층(±95% CI)과 붕괴
  fig4 처방별 순위 개선(AUPRC 차, ±95% CI)과 메워야 할 순위 층 / fig5 융합·보조·천장 AUPRC / fig7 val 크기별 재보정 이득(R3)
그림 2(구조도)·6(γ 스윙)·8(ESBL)은 이 스크립트 밖에서 만든다.
글자는 영문(서버 한글 글꼴 의존 제거), 색은 검증된 범주 팔레트 앞 3칸(파랑·주황·청록)과 회색만 사용, 이중 축 없음.
"""
import json, os, sys
import numpy as np

FINAL = "results/thesis_round/FINAL"
C1, C2, C3, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9890", "#0b0b0b", "#52514e"
CELLS = [("clip", "text", "CLIP\ntext missing"), ("clip", "image", "CLIP\nimage missing"),
         ("siglip", "text", "SigLIP\ntext missing"), ("siglip", "image", "SigLIP\nimage missing")]


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e5e0",
                         "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300})
    return plt


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, "%s.%s" % (name, ext)), bbox_inches="tight")


def err(c):
    return np.array([[c[0] - c[1]], [c[2] - c[0]]])


def fig1(plt, J, out):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, (bb, reg, title) in zip(axs, [("clip", "text", "CLIP, text missing"), ("siglip", "image", "SigLIP, image missing")]):
        c = J[bb][reg]; zf, R = c["means"]["zf"]["f"], c["means"]["R"]["f"]
        thr, rank = c["decomp"]["thr"][0], c["decomp"]["rank"][0]
        steps = [("zero-fill\n(thr 0.5)", 0, zf, GRAY), ("threshold\nlayer", zf, thr, C1), ("ranking\nlayer", zf + thr, rank, C2), ("ceiling R\n(thr 0.5)", 0, R, GRAY)]
        for i, (lab, base, h, col) in enumerate(steps):
            ax.bar(i, h, bottom=base, color=col, width=0.6, edgecolor="white", linewidth=1)
            top = base + h
            ax.text(i, max(top, base) + 0.008, ("%.3f" % h) if i in (0, 3) else ("%+.3f" % h), ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(range(4)); ax.set_xticklabels([s[0] for s in steps]); ax.set_title(title, fontsize=9, color=INK)
        ax.axhline(0, color=MUTED, linewidth=0.8)
    axs[0].set_ylabel("macro-F1")
    fig.suptitle("Collapse = threshold layer + ranking layer (MM-IMDb, 5 seeds)", fontsize=9.5, color=INK, y=1.03)
    save(fig, out, "fig1_waterfall"); plt.close(fig)


def fig3(plt, J, out):
    fig, ax = plt.subplots(figsize=(7.2, 3.0)); x = np.arange(len(CELLS)); w = 0.32
    for k, (key, lab, col) in enumerate((("thr", "threshold layer", C1), ("rank", "ranking layer", C2))):
        for i, (bb, reg, _) in enumerate(CELLS):
            c = J[bb][reg]["decomp"][key]
            ax.bar(x[i] + (k - 0.5) * w, c[0], w, color=col, label=lab if i == 0 else None, edgecolor="white", linewidth=1)
            ax.errorbar(x[i] + (k - 0.5) * w, c[0], yerr=err(c), fmt="none", ecolor=INK, elinewidth=0.9, capsize=2)
    for i, (bb, reg, _) in enumerate(CELLS):
        ax.plot(x[i], J[bb][reg]["decomp"]["collapse"][0], marker="D", color=INK, markersize=5, label="collapse (R − zf)" if i == 0 else None, linestyle="none")
    ax.axhline(0, color=MUTED, linewidth=0.8); ax.set_xticks(x); ax.set_xticklabels([c[2] for c in CELLS])
    ax.set_ylabel("macro-F1 difference"); ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Layer decomposition of missing-modality collapse (mean ± 95% CI, 5 seeds)", fontsize=9.5, color=INK)
    save(fig, out, "fig3_layers"); plt.close(fig)


def fig4(plt, J, out):
    fig, ax = plt.subplots(figsize=(7.2, 3.0)); x = np.arange(len(CELLS)); w = 0.24
    for k, (key, lab, col) in enumerate((("g8", "γ (aux loss, γ=8)", C1), ("md", "ModDrop", C2), ("gm", "γ + ModDrop", C3))):
        for i, (bb, reg, _) in enumerate(CELLS):
            c = J[bb][reg]["presc"][key]["a"]
            ax.bar(x[i] + (k - 1) * w, c[0], w, color=col, label=lab if i == 0 else None, edgecolor="white", linewidth=1)
            ax.errorbar(x[i] + (k - 1) * w, c[0], yerr=err(c), fmt="none", ecolor=INK, elinewidth=0.9, capsize=2)
    for i, (bb, reg, _) in enumerate(CELLS):
        g = J[bb][reg]["decomp"]["ap_gap"][0]
        ax.hlines(g, x[i] - 1.6 * w, x[i] + 1.6 * w, colors=INK, linestyles="dashed", linewidth=1, label="ranking gap to fill (AUPRC R − zf)" if i == 0 else None)
    ax.axhline(0, color=MUTED, linewidth=0.8); ax.set_xticks(x); ax.set_xticklabels([c[2] for c in CELLS])
    ax.set_ylabel("AUPRC gain over zero-fill"); ax.legend(loc="upper left", fontsize=7.5, ncol=2)
    ax.set_title("Which remedy fills the ranking layer (mean ± 95% CI, 5 seeds)", fontsize=9.5, color=INK)
    save(fig, out, "fig4_remedies"); plt.close(fig)


def fig5(plt, J, out):
    fig, ax = plt.subplots(figsize=(7.2, 2.8)); x = np.arange(len(CELLS))
    for i, (bb, reg, _) in enumerate(CELLS):
        m = J[bb][reg]["means"]
        fu, hd, R = m["g8"]["a"], m["g8_head"]["a"], m["R"]["a"]
        ax.plot([x[i], x[i]], [fu, hd], color=GRAY, linewidth=1.5, zorder=1)
        ax.scatter([x[i]], [fu], s=40, color=C1, zorder=3, label="fused classifier (missing input)" if i == 0 else None, edgecolors="white", linewidths=1.5)
        ax.scatter([x[i]], [hd], s=40, color=C2, zorder=3, label="auxiliary head of remaining modality" if i == 0 else None, edgecolors="white", linewidths=1.5)
        ax.hlines(R, x[i] - 0.18, x[i] + 0.18, colors=INK, linewidth=1.4, label="ceiling R" if i == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels([c[2] for c in CELLS]); ax.set_xlim(-0.5, len(CELLS) - 0.5)
    ax.set_ylabel("AUPRC (γ=8 model)"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=7.5)
    ax.set_title("The fused classifier does not use what the auxiliary head knows (mean, 5 seeds)", fontsize=9.5, color=INK)
    save(fig, out, "fig5_aux_vs_fused"); plt.close(fig)


def fig7(plt, R3, out):
    fig, ax = plt.subplots(figsize=(5.2, 3.0)); ns = ["300", "600", "1200", "full"]; xs = np.arange(len(ns))
    styles = {("clip", "text"): (C1, "-"), ("clip", "image"): (C1, "--"), ("siglip", "text"): (C2, "-"), ("siglip", "image"): (C2, "--")}
    for (bb, reg, lab) in CELLS:
        if (bb, reg) not in R3:
            continue
        v = [R3[(bb, reg)][n] for n in ns]; col, ls = styles[(bb, reg)]
        ax.plot(xs, v, color=col, linestyle=ls, linewidth=2, marker="o", markersize=4, label=lab.replace("\n", ", "))
        dy = {("siglip", "text"): 0.006, ("siglip", "image"): -0.006}.get((bb, reg), 0.0)
        ax.text(xs[-1] + 0.08, v[-1] + dy, lab.replace("\n", " "), fontsize=7, color=MUTED, va="center")
    ax.set_xticks(xs); ax.set_xticklabels(["300", "600", "1200", "full val"]); ax.set_xlabel("validation images used for recalibration")
    ax.set_ylabel("recalibration gain of zero-fill\n(macro-F1)"); ax.set_xlim(-0.2, len(ns) - 1 + 0.9)
    ax.set_title("Recalibration needs little validation data (MM-IMDb, 20 draws)", fontsize=9.5, color=INK)
    save(fig, out, "fig7_valsize"); plt.close(fig)


def load_r3(root):
    R3 = {}
    for bb in ("clip", "siglip"):
        p = os.path.join(root, "results/thesis_round/V2/mmimdb_v2_robust_%s_seeds12345.json" % bb)
        if not os.path.exists(p):
            continue
        r = json.load(open(p, encoding="utf-8"))
        for reg in ("text", "image"):
            try:
                R3[(bb, reg)] = {n: float(r["cells"][reg]["R3"][n]["c1"]) for n in ("300", "600", "1200", "full")}
            except (KeyError, TypeError):
                pass
    return R3


def main(root="."):
    plt = setup()
    J = json.load(open(os.path.join(root, FINAL, "tables.json"), encoding="utf-8"))
    out = os.path.join(root, FINAL, "figs")
    fig1(plt, J, out); fig3(plt, J, out); fig4(plt, J, out); fig5(plt, J, out)
    R3 = load_r3(root)
    print("R3 셀 %d개 읽음" % len(R3))
    if R3:
        fig7(plt, R3, out)
    print("저장:", sorted(os.listdir(out))); print("=== ALL DONE (논문용 그림) ===")


def selftest():
    import tempfile, shutil
    root = tempfile.mkdtemp(prefix="tfig_")
    rng = np.random.RandomState(0)
    c3 = lambda m: [m, m - 0.01, m + 0.01]
    J = {}
    for bb in ("clip", "siglip"):
        J[bb] = {}
        for reg in ("text", "image"):
            thr, rank = rng.uniform(-0.04, 0.18), rng.uniform(0.01, 0.08)
            J[bb][reg] = {"means": {k: {"f": rng.uniform(.3, .5), "x": .5, "a": rng.uniform(.4, .6)} for k in ("R", "zf", "g8", "g8_head")},
                          "decomp": {"thr": c3(thr), "rank": c3(rank), "collapse": c3(thr + rank), "ap_gap": c3(rank * 1.2)},
                          "presc": {k: {"a": c3(rng.uniform(-.02, .07))} for k in ("g8", "md", "gm")}}
    os.makedirs(os.path.join(root, FINAL)); json.dump(J, open(os.path.join(root, FINAL, "tables.json"), "w"))
    os.makedirs(os.path.join(root, "results/thesis_round/V2"))
    for bb in ("clip", "siglip"):
        json.dump({"cells": {reg: {"R3": {n: {"c1": v} for n, v in zip(("300", "600", "1200", "full"), (.08, .09, .1, .11))}} for reg in ("text", "image")}},
                  open(os.path.join(root, "results/thesis_round/V2/mmimdb_v2_robust_%s_seeds12345.json" % bb), "w"))
    main(root)
    files = sorted(os.listdir(os.path.join(root, FINAL, "figs")))
    ok = len(files) == 10 and all(os.path.getsize(os.path.join(root, FINAL, "figs", f)) > 5000 for f in files)
    keep = os.environ.get("KEEP_FIGS")
    if keep:
        shutil.copytree(os.path.join(root, FINAL, "figs"), keep, dirs_exist_ok=True)
    shutil.rmtree(root, ignore_errors=True)
    print("자체 시험:", files); print("자체 시험 판정:", "모두 통과" if ok else "★실패")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    main(".")
