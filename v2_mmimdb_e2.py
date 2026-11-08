"""추가 실험 v2 — MM-IMDb 실행기 (E2 단일모달 천장 / E9 재보정용 zf·γ8 재학습).
기존 zero-fill 결과 json의 config를 그대로 읽어, method(와 γ)만 바꿔 train_v12.py를 실행한다.
(명령줄 인자를 추측하지 않기 위함. seed 1·2·3, 같은 seed의 zf config 사용)
모든 run은 val·test 로짓(full/text결손/image결손)도 함께 저장한다(--save_logits; json 저장 이후 실행 → 결과 불변).
  python v2_mmimdb_e2.py clip --dry                                   # 원본 json·명령만 출력(실행 안 함)
  python v2_mmimdb_e2.py clip                                         # E2: image_only, text_only
  python v2_mmimdb_e2.py clip --methods zerofill,gamma --gamma 8.0 --outdir results/thesis_round/V2/E9_mmimdb   # E9
이미 결과 json과 로짓이 모두 있는 run은 건너뛴다(중단 후 같은 명령으로 이어서 실행 가능).
"""
import glob, json, os, re, subprocess, sys


def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


BB = sys.argv[1]
DRY = "--dry" in sys.argv
METHODS = opt("--methods", "image_only,text_only").split(",")
GAMMA = opt("--gamma", None)
OUTDIR = opt("--outdir", "results/thesis_round/V2/E2_mmimdb")
BASE_SEEDS = [1, 2, 3]                                   # zero-fill 결과 json이 있는 기존 seed
SEEDS = [int(x) for x in opt("--seeds", "1,2,3").split(",")]   # seed 확장: 새 seed는 seed 1~3과 같은 설정(seed만 변경)
SKIP_KEYS = {"out", "save_preds", "save_model", "seed", "method", "causal", "smoke", "save_logits"}
GAMMA_METHODS = ("gamma", "gamma_moddrop", "gamma_b1")   # --gamma는 이 방법들에만 적용(zerofill은 원 config 그대로 → 재현성 확인 가능)
SIGLIP_DEFAULT = "siglip-so400m-patch14-384"

known = set(re.findall(r'add_argument\("--([a-z_0-9]+)"', open("train_v12.py", encoding="utf-8").read()))
assert "save_logits" in known, "train_v12.py에 --save_logits가 없음 → git pull 필요"
cands = {}
for f in sorted(glob.glob("results/*.json")):
    try:
        r = json.load(open(f, encoding="utf-8"))
    except Exception:
        continue
    c = r.get("config")
    if not isinstance(c, dict) or "text" not in r.get("eval", {}):
        continue
    if c.get("mode") != "real" or c.get("backbone") != BB or c.get("method") != "zerofill" or c.get("limit"):
        continue
    if BB == "siglip" and os.path.basename(str(c.get("siglip_name"))) != SIGLIP_DEFAULT:
        continue
    cands.setdefault(c.get("seed"), []).append((f, c, r["eval"]["text"]["macro"], r["eval"]["image"]["macro"]))

print("=" * 90); print("v2 MM-IMDb 실행기 [%s] methods=%s γ=%s → %s" % (BB, METHODS, GAMMA, OUTDIR)); print("=" * 90)
chosen = {}
for s in BASE_SEEDS:
    L = cands.get(s, [])
    if not L:
        sys.exit("★ seed %d의 zero-fill 결과 json이 없음 → 중단(추측 실행 금지)" % s)
    mats = [{k: v for k, v in c.items() if k not in SKIP_KEYS} for _, c, _, _ in L]
    if any(m != mats[0] for m in mats[1:]):
        print("★ seed %d 후보 %d개의 설정이 서로 다름 → 중단. 후보:" % (s, len(L)))
        for (f, c, t, i), m in zip(L, mats):
            print("   ", f, {k: m[k] for k in m if any(mm.get(k) != m.get(k) for mm in mats)})
        sys.exit(1)
    f, c, t, i = L[0]
    chosen[s] = c
    print("seed %d ← %s (후보 %d개, 설정 동일) | zf text결손 macro %.4f, image결손 macro %.4f" % (s, f, len(L), t, i))
NEW = [s for s in SEEDS if s not in chosen]
if NEW:   # 새 seed: 기존 seed 1~3의 설정이 seed 외에 모두 같을 때만 그 설정을 사용
    base = [{k: v for k, v in chosen[b].items() if k not in SKIP_KEYS} for b in BASE_SEEDS]
    if any(m != base[0] for m in base[1:]):
        sys.exit("★ seed 1~3의 zero-fill 설정이 seed 외에도 서로 다름 → 새 seed 설정을 정할 수 없어 중단")
    for s in NEW:
        chosen[s] = chosen[BASE_SEEDS[0]]
        print("seed %d ← seed 1~3과 같은 설정(seed만 %d로 변경, 새 seed)" % (s, s))

os.makedirs(OUTDIR, exist_ok=True)
for m in METHODS:
    for s in SEEDS:
        stem = "%s/%s_%s%s_s%d" % (OUTDIR, BB, m, ("_g" + GAMMA) if (GAMMA and m in ("gamma", "gamma_moddrop")) else "", s)
        out, lg = stem + ".json", stem + "_logits.npz"
        override_g = GAMMA is not None and m in GAMMA_METHODS
        cmd = [sys.executable, "-u", "train_v12.py"]   # -u: 로그 즉시 기록(모니터링용)
        for k, v in chosen[s].items():
            if k in SKIP_KEYS or k not in known or v is None or v is False or (override_g and k == "gamma"):
                continue
            cmd += ["--" + k] if v is True else ["--" + k, str(v)]
        if override_g:
            cmd += ["--gamma", GAMMA]
        # 재현성 대조용: 같은 seed·방법(·γ)의 기존 결과 json이 있으면 설정 차이와 기존 수치를 출력
        if m != "zerofill":
            for f in sorted(glob.glob("results/*.json")):
                try:
                    r = json.load(open(f, encoding="utf-8"))
                except Exception:
                    continue
                c = r.get("config")
                if (not isinstance(c, dict) or c.get("mode") != "real" or c.get("backbone") != BB or c.get("method") != m
                        or c.get("seed") != s or c.get("limit") or "text" not in r.get("eval", {})):
                    continue
                if override_g and abs(float(c.get("gamma", -1)) - float(GAMMA)) > 1e-9:
                    continue
                diff = {k: (chosen[s].get(k), c.get(k)) for k in set(chosen[s]) | set(c)
                        if k not in SKIP_KEYS and k != "gamma" and chosen[s].get(k) != c.get(k)}
                print("  [기존 %s] text결손 %.4f image결손 %.4f | zf config와 다른 키: %s" % (
                    f, r["eval"]["text"]["macro"], r["eval"]["image"]["macro"], diff if diff else "없음"))
        cmd += ["--method", m, "--seed", str(s), "--out", out, "--save_logits", lg]
        print("\n$ " + " ".join(cmd), flush=True)
        if DRY:
            continue
        if os.path.exists(out) and os.path.exists(lg):
            print("  (결과·로짓 모두 있음, 건너뜀)"); continue
        rc = subprocess.call(cmd)
        print("  → 종료코드 %d" % rc, flush=True)
        if rc != 0:   # 실패(GPU 점유 등) 시 다음 run으로 넘어가 헛돌지 않도록 즉시 중단 → 원인 해결 후 같은 명령으로 재개
            sys.exit("★ run 실패(종료코드 %d) → 실행기 중단. 원인 확인 후 같은 명령으로 다시 실행하면 완료된 run은 건너뜀" % rc)
print("\n=== ALL DONE (V2 MM-IMDb %s %s%s) ===" % (BB, ",".join(METHODS), " DRY" if DRY else ""))
