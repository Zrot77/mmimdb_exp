"""추가 실험 v2 — 이상 run 의심 점검: 설정·코드·학습 로그·val 성능을 같은 방법의 seed 1~5와 나란히 비교 (학습 없음, CPU)
  python v2_run_inspect.py                       # 기본: clip moddrop, siglip zerofill (seed별 값 점검에서 의심된 run)
  python v2_run_inspect.py clip:moddrop siglip:zerofill
  python v2_run_inspect.py --selftest
[고정 규칙 — 실행 전, 2026-11-10, 의심 run의 수치를 본 뒤지만 이 점검 결과를 보기 전에 정함]
- 결함 기준(하나라도 해당하면 '★결함 run'):
  (1) config가 seed 1과 seed·out·save_logits 외에 다름   (2) 로그에 그 run의 블록은 있으나 종료코드 0으로 끝난 블록이 없음
  (3) 학습 loss에 nan/inf   (4) 정상 블록에 val 조기중단 선택 줄('조기중단 선택')이 없음
- '확인 필요'(결함 아님, 사람이 판단): 로그 블록을 못 찾음(로그 위치 문제일 수 있음, V2_LOGS 환경변수로 추가 위치 지정),
  git_commit이 seed 1~3과 다름 → train_v12.py의 두 commit 사이 차이를 출력함. 차이가 학습 뒤 저장 코드(로짓·보조 분류기 출력 저장)나
  `--unfreeze_last` 금지 확인뿐이면 학습에 영향 없음으로 보고, 학습 경로(모델·손실·데이터·최적화)가 바뀌었으면 결함으로 봄.
- 결함이 없으면 '정상 run' = seed 변동으로 봄: run을 빼지 않고 n=5 결과를 그대로 보고, seed 1~3 평균 대 seed 4·5 평균을 각주로 둠.
  run을 결과를 보고 빼는 것은 사후 선택이므로 하지 않음.
- 결함이 있으면 그 run만 같은 seed로 새 폴더에 다시 학습해 대조하고, 원래 run은 지우지 않음(사용자와 결정).
- val 성능은 해석 참고용: test에서 약한 run이 val에서도 약하면 '모델 자체가 약하게 학습된 seed'로 기술함(판정 아님).
"""
import glob, json, math, os, re, sys

V2 = "results/thesis_round/V2"
DIRS = {"zerofill": "E9_mmimdb", "gamma_g8.0": "E9_mmimdb", "moddrop": "E9_mmimdb", "gamma_moddrop_g8.0": "E9_mmimdb",
        "image_only": "E2_mmimdb", "text_only": "E2_mmimdb"}
SKIP = {"seed", "out", "save_logits"}
DIFFS = {}


def log_blocks(stem_json, logs):
    """로그들에서 '$ ... --out <stem_json>' 로 시작해 다음 '$ ' 또는 '→ 종료코드' 까지의 블록 목록"""
    found = []
    for f in logs:
        try:
            lines = open(f, encoding="utf-8", errors="replace").read().splitlines()
        except Exception:
            continue
        i = 0
        while i < len(lines):
            if lines[i].startswith("$ ") and ("--out " + stem_json) in lines[i]:
                j = i + 1; blk = []
                while j < len(lines) and not lines[j].startswith("$ "):
                    blk.append(lines[j])
                    if lines[j].strip().startswith("→ 종료코드"):
                        break
                    j += 1
                found.append((f, blk)); i = j
            i += 1
    return found


def parse_block(blk):
    losses = [float(m.group(1)) if m.group(1) not in ("nan", "inf", "-inf") else float(m.group(1))
              for m in (re.search(r"loss=([-+0-9.eE]+|nan|inf|-inf)", l) for l in blk) if m]
    rc = [int(m.group(1)) for m in (re.search(r"종료코드 (-?\d+)", l) for l in blk) if m]
    es = [int(m.group(1)) for m in (re.search(r"조기중단 선택: epoch (\d+)", l) for l in blk) if m]
    vals = [l.strip() for l in blk if "[val]" in l]
    return {"losses": losses, "rc": rc[-1] if rc else None, "best_ep": es[-1] if es else None, "vals": vals,
            "nonfinite": any(not math.isfinite(x) for x in losses)}


def inspect(bb, m, logs, seeds=(1, 2, 3, 4, 5), target=None):
    d = os.path.join(V2, DIRS[m]); J = {}
    for s in seeds:
        p = os.path.join(d, "%s_%s_s%d.json" % (bb, m, s))
        J[s] = (p, json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None)
    print("=" * 110); print("[%s %s] seed %s%s" % (bb, m, list(seeds), (" | 의심 run: s%d" % target) if target else "")); print("=" * 110)
    c1 = (J[seeds[0]][1] or {}).get("config", {}); flags = {}
    commits13 = {(J[s][1] or {}).get("run", {}).get("git_commit") for s in seeds[:3] if J[s][1]}
    for s in seeds:
        p, j = J[s]; fl = []
        if j is None:
            print("  s%d: ★json 없음 %s" % (s, p)); flags[s] = ["json 없음"]; continue
        c = j.get("config", {})
        diff = {k: (c1.get(k), c.get(k)) for k in set(c1) | set(c) if k not in SKIP and c1.get(k) != c.get(k)}
        if diff:
            fl.append("(1) config 다름 %s" % diff)
        run = j.get("run", {}); gc = run.get("git_commit")
        chk = []
        if s not in seeds[:3] and gc not in commits13:
            chk.append("git_commit %s ∉ seed1~3 %s" % (gc, sorted(map(str, commits13))))
            for c0 in sorted(x for x in commits13 if x):
                diffcmd = ["git", "diff", "--unified=0", c0, gc, "--", "train_v12.py"]
                try:
                    import subprocess
                    out = subprocess.check_output(diffcmd, text=True, stderr=subprocess.STDOUT)
                    DIFFS.setdefault((c0, gc), out)
                except Exception as e:
                    DIFFS.setdefault((c0, gc), "git diff 실패: %r" % e)
        blks = log_blocks(p, logs)
        P = [parse_block(b) for _, b in blks]
        ok_blks = [x for x in P if x["rc"] == 0]
        if not blks:
            chk.append("로그 블록 못 찾음")
        elif not ok_blks:
            fl.append("(2) 종료코드 0 블록 없음(블록 %d개, 로그 %s)" % (len(blks), sorted({os.path.basename(f) for f, _ in blks})))
        last = ok_blks[-1] if ok_blks else (P[-1] if P else None)
        if last and last["nonfinite"]:
            fl.append("(3) loss nan/inf")
        if ok_blks and last["best_ep"] is None:
            fl.append("(4) 조기중단 선택 줄 없음")
        ev, va = j.get("eval", {}), j.get("val", {})
        g = lambda e, r: e.get(r, {}).get("macro", float("nan"))
        print("  s%d | commit %s | gpu %s | %ss | 로그 블록 %d(정상 %d) | best ep %s | loss %s" % (
            s, gc, run.get("gpu"), run.get("elapsed_sec"), len(blks), len(ok_blks), last["best_ep"] if last else "-",
            " ".join("%.4f" % x for x in last["losses"]) if last else "-"))
        print("       val  macro full %.4f text결손 %.4f image결손 %.4f | test macro full %.4f text결손 %.4f image결손 %.4f | suff img %.3f txt %.3f" % (
            g(va, "full"), g(va, "text"), g(va, "image"), g(ev, "full"), g(ev, "text"), g(ev, "image"),
            j.get("suff", {}).get("z_image", float("nan")), j.get("suff", {}).get("z_text", float("nan"))))
        if last:
            for v in last["vals"]:
                print("       " + v)
        print("       → " + ("★결함 기준 해당: " + " | ".join(fl) if fl else "결함 기준 해당 없음")
              + ((" | 확인 필요: " + " | ".join(chk)) if chk else ""))
        if blks:
            print("       로그: " + ", ".join(sorted({f for f, _ in blks})))
        flags[s] = fl
    return flags


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")] or ["clip:moddrop:4", "siglip:zerofill:5"]
    extra = [x for pat in os.environ.get("V2_LOGS", "").split(",") if pat for x in glob.glob(os.path.expanduser(pat), recursive=True)]
    logs = sorted(set(glob.glob(V2 + "/*.log") + glob.glob(V2 + "/**/*.log", recursive=True) + glob.glob("*.log")
                      + glob.glob("nohup*.out") + glob.glob("logs/*.log") + extra))
    print("검색한 로그 %d개" % len(logs))
    verdict = []
    for a in args:
        parts = a.split(":"); bb, m = parts[0], parts[1]; t = int(parts[2]) if len(parts) > 2 else None
        fl = inspect(bb, m, logs, target=t)
        if t is not None:
            verdict.append("%s %s s%d: %s" % (bb, m, t, "★결함 run → 같은 seed 재학습 대조 필요" if fl.get(t) else
                                                     "결함 기준 해당 없음 → (확인 필요 항목이 해소되면) 정상 run: seed 변동으로 보고 n=5 결과 그대로, s1~3 대 s4·5 각주"))
    for (a, b), out in DIFFS.items():
        print("\n[train_v12.py 차이 %s → %s] (%d줄, 앞 120줄만)" % (a, b, len(out.splitlines())))
        print("\n".join(out.splitlines()[:120]) or "(차이 없음)")
    print("\n" + "=" * 110)
    for v in verdict:
        print("▶ " + v)
    print("=== ALL DONE (이상 run 점검) ===")


def selftest():
    import tempfile
    t = tempfile.mkdtemp(); stem = os.path.join(t, "x_s4.json")
    open(os.path.join(t, "a.log"), "w", encoding="utf-8").write("\n".join([
        "$ python train_v12.py --out %s --seed 4" % stem, "  epoch  1  loss=0.3000", "    [val] ep 4 full_micro=0.6 (best 0.6@4)",
        "  => 조기중단 선택: epoch 4", "  → 종료코드 0",
        "$ python train_v12.py --out %s --seed 5" % stem.replace("s4", "s5"), "  epoch  1  loss=nan", "  → 종료코드 1"]))
    lg = [os.path.join(t, "a.log")]
    b4 = [parse_block(b) for _, b in log_blocks(stem, lg)]; b5 = [parse_block(b) for _, b in log_blocks(stem.replace("s4", "s5"), lg)]
    ok = [len(b4) == 1 and b4[0]["rc"] == 0 and b4[0]["best_ep"] == 4 and not b4[0]["nonfinite"] and len(b4[0]["vals"]) == 1,
          len(b5) == 1 and b5[0]["rc"] == 1 and b5[0]["nonfinite"] and b5[0]["best_ep"] is None,
          log_blocks(os.path.join(t, "none.json"), lg) == []]
    print("자체 시험:", ok); print("자체 시험 판정:", "모두 통과" if all(ok) else "★실패")
    return all(ok)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    main()
