"""추가 실험 v2 — MM-IMDb seed별 값 점검: 새 seed 4·5가 이상 run이 아닌지 (학습 없음, CPU, 저장된 로짓만 사용)
  V2_SEEDS=1,2,3,4,5 python v2_seed_perseed.py     # 서버 ~/Yechan/mmimdb_exp (conda mmimdb)
  python v2_seed_perseed.py --selftest
[고정 규칙 — 실행 전, 2026-11-10] 기술 통계만, 판정 없음
- 목적: seed 확장으로 판정이 바뀐 비교가 seed 4·5 중 한 run의 이상값 하나 때문인지 확인.
- 입력: E2_mmimdb(R) · E9_mmimdb(zf, γ8, MD, GM) run, 계산 함수는 mmimdb_v2_analysis.py 것을 그대로 씀(재보정 = 정확 탐색).
- 출력 1(절대값): 방법·조건(full/text결손/image결손)별 seed 1~5의 F1@0.5, F1@재보정, AUPRC.
- 출력 2(짝지은 차이): 판정이 바뀐 비교의 seed별 차이, s1~3 평균, s4·5 평균.
- 표시 규칙: s4 또는 s5 값이 s1~3의 [최소 − 범위, 최대 + 범위] 밖이면 '◆범위 밖'(범위 = 최대 − 최소). 판정 아님.
  기준 seed가 3개뿐이라 정상 run도 지표 하나에서는 우연히 ◆가 붙을 수 있음(가짜 무작위 run 시험에서 약 20%).
  그래서 run별 ◆ 개수(절대값 9칸 = 조건 3 × 지표 3)를 따로 세어, 한 run이 9칸 중 5칸 이상이면 '★이상 run 의심'으로 표시하고
  그 run의 학습 로그를 확인함(선별용: 이상 없는 가짜 run 24묶음에서 우연 의심 1건, 심어 둔 이상 run은 9/9로 검출).
  같은 run의 지표들은 서로 상관돼 있어 정상이지만 다른 seed도 걸릴 수 있으므로, 의심만으로 판정·run을 바꾸지 않음. 의심 run이 없으면 "판정 변화는 특정 run의 이상값이 아니라 seed 변동"으로 기술함.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmimdb_v2_analysis import f1_at05, f1_recal_exact, auprc, load_run

SEEDS = [1, 2, 3, 4, 5]
NAMES = {"R_img": ("E2_mmimdb", "image_only"), "R_txt": ("E2_mmimdb", "text_only"), "zf": ("E9_mmimdb", "zerofill"),
         "g8": ("E9_mmimdb", "gamma_g8.0"), "md": ("E9_mmimdb", "moddrop"), "gm": ("E9_mmimdb", "gamma_moddrop_g8.0")}
CONDS = (("full", "full"), ("text결손", "textmissing"), ("image결손", "imagemissing"))
# (조건, 이름, 앞 run, 뒤 run, 지표)  f=F1@0.5, x=F1@재보정(정확), a=AUPRC — 판정이 바뀐 비교 + 주 판정 비교
DIFFS = [("text결손", "잔여 붕괴 R+재보정 − zf+재보정", "R_img", "zf", "x"), ("text결손", "AUPRC R − zf", "R_img", "zf", "a"),
         ("text결손", "GM+재보정 − γ8+재보정", "gm", "g8", "x"),
         ("image결손", "γ8+재보정 − zf+재보정", "g8", "zf", "x"), ("image결손", "AUPRC γ8 − zf", "g8", "zf", "a"),
         ("image결손", "MD − zf (F1@0.5)", "md", "zf", "f"), ("image결손", "MD+재보정 − zf+재보정", "md", "zf", "x"),
         ("image결손", "MD+재보정 − γ8+재보정", "md", "g8", "x"), ("image결손", "AUPRC MD − γ8", "md", "g8", "a"),
         ("image결손", "GM+재보정 − MD+재보정", "gm", "md", "x"), ("image결손", "AUPRC GM − MD", "gm", "md", "a"),
         ("full", "γ8 − zf (F1@0.5)", "g8", "zf", "f"), ("full", "γ8+재보정 − zf+재보정", "g8", "zf", "x"),
         ("full", "GM+재보정 − γ8+재보정", "gm", "g8", "x")]


def outside(v):
    """v: seed 1~5 값. s4·s5 중 s1~3의 [min−range, max+range] 밖인 seed 목록"""
    a = np.asarray(v[:3], float); lo, hi = a.min(), a.max(); r = hi - lo
    return [s for s, x in zip((4, 5), v[3:5]) if x < lo - r or x > hi + r]


def metrics(z, nk):
    yv, yt = z["val_y"], z["test_y"]
    return {"f": f1_at05(yt, z["test_" + nk]), "x": f1_recal_exact(yv, z["val_" + nk], yt, z["test_" + nk]),
            "a": auprc(yt, z["test_" + nk])}


def fmt(v):
    return " ".join("%+.4f" % x for x in v)


def analyze(root, bb):
    V = {}
    for k, (d, m) in NAMES.items():
        for s in SEEDS:
            r = load_run(os.path.join(root, "results/thesis_round/V2", d, "%s_%s_s%d" % (bb, m, s)))
            if r is None:
                sys.exit("★ 없는 run: %s/%s_%s_s%d → 중단" % (d, bb, m, s))
            z = r[1]
            for cn, nk in CONDS:
                if "test_" + nk in z:
                    V[(k, cn, s)] = metrics(z, nk)
    print("=" * 110); print("MM-IMDb [%s] seed별 값 (기술 통계, 판정 없음) | ◆ = s4·s5가 s1~3 [최소−범위, 최대+범위] 밖" % bb); print("=" * 110)
    print("[절대값]  seed 1 2 3 4 5 순서")
    nflag = 0; per_run = {}
    for cn, _ in CONDS:
        for k in NAMES:
            if (k, cn, 1) not in V:
                continue
            for mk, ml in (("f", "F1@0.5"), ("x", "F1@재보정"), ("a", "AUPRC")):
                v = [V[(k, cn, s)][mk] for s in SEEDS]; o = outside(v); nflag += bool(o)
                for s in o:
                    per_run[(k, s)] = per_run.get((k, s), 0) + 1
                print("  %-9s %-6s %-9s %s%s" % (cn, k, ml, " ".join("%.4f" % x for x in v), ("  ◆범위 밖 s%s" % o) if o else ""))
    print("\n[짝지은 차이]  seed 1 2 3 4 5 | s1~3 평균 | s4·5 평균")
    out = {}
    for cn, name, a, b, mk in DIFFS:
        if (a, cn, 1) not in V or (b, cn, 1) not in V:
            print("  %-9s %-28s 해당 없음" % (cn, name)); continue
        v = [V[(a, cn, s)][mk] - V[(b, cn, s)][mk] for s in SEEDS]; o = outside(v); nflag += bool(o)
        print("  %-9s %-28s %s | %+.4f | %+.4f%s" % (cn, name, fmt(v), np.mean(v[:3]), np.mean(v[3:]), ("  ◆범위 밖 s%s" % o) if o else ""))
        out["%s/%s" % (cn, name)] = v
    print("\n◆ 표시 %d개 (판정 아님)" % nflag)
    sus = sorted((k, s, c) for (k, s), c in per_run.items() if c >= 5)
    print("run별 절대값 ◆ 개수(9칸 중): " + (", ".join("%s s%d %d" % (k, s, c) for (k, s), c in sorted(per_run.items())) or "없음"))
    print("▶ " + ("★이상 run 의심: " + ", ".join("%s s%d (%d/9)" % x for x in sus) + " → 학습 로그 확인" if sus
                  else "이상 run 의심 없음(9칸 중 5칸 이상 ◆인 run 없음) → 판정 변화는 특정 run의 이상값이 아니라 seed 변동"))
    return out, sus


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    for bb in ("clip", "siglip"):
        analyze(root, bb); print()
    print("=== ALL DONE (seed별 값 점검) ===")


def selftest():
    ok = [outside([0.1, 0.2, 0.3, 0.25, 0.15]) == [],
          outside([0.1, 0.2, 0.3, 0.55, 0.15]) == [4],          # 0.3 + 0.2 = 0.5 초과
          outside([0.1, 0.2, 0.3, 0.5, -0.11]) == [5],          # 경계 0.5는 안, −0.1 미만은 밖
          outside([0.2, 0.2, 0.2, 0.2, 0.21]) == [5]]           # 범위 0이면 같지 않은 값은 밖
    # 실제 계산 경로: 가짜 run으로 metrics가 F1@0.5·재보정·AUPRC를 돌려주는지
    rng = np.random.RandomState(0); y = (rng.rand(400, 5) < 0.3).astype(np.float32)
    lg = (y * 2 - 1) * 1.5 + rng.randn(400, 5)
    z = {"val_y": y[:200], "test_y": y[200:], "val_full": lg[:200], "test_full": lg[200:]}
    m = metrics(z, "full"); ok.append(all(0.5 < m[k] <= 1.0 for k in "fxa"))
    zb = dict(z, test_full=-lg[200:]); mb = metrics(zb, "full"); ok.append(mb["a"] < 0.5)   # 순위 뒤집으면 AUPRC 낮음
    print("자체 시험:", ok); print("자체 시험 판정:", "모두 통과" if all(ok) else "★실패")
    return all(ok)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    main()
