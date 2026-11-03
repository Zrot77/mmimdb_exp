"""추가 실험 v2 — MM-IMDb E2 실행기 (사전등록 thesis_round/ADDENDUM_v2_prereg.md §E2).
기존 zero-fill 결과 json의 config를 그대로 읽어, method만 image_only / text_only로 바꿔 train_v12.py를 실행한다.
(명령줄 인자를 추측하지 않기 위함. seed 1·2·3, 같은 seed의 zf config 사용)
  python v2_mmimdb_e2.py clip --dry     # 원본 json·명령만 출력(실행 안 함)
  python v2_mmimdb_e2.py clip           # 실행 (이미 있는 결과는 건너뜀)
  python v2_mmimdb_e2.py siglip
"""
import glob, json, os, re, subprocess, sys

BB = sys.argv[1]
DRY = "--dry" in sys.argv
SEEDS = [1, 2, 3]
METHODS = ["image_only", "text_only"]
OUTDIR = "results/thesis_round/V2/E2_mmimdb"
SKIP_KEYS = {"out", "save_preds", "save_model", "seed", "method", "causal", "smoke"}
SIGLIP_DEFAULT = "siglip-so400m-patch14-384"

known = set(re.findall(r'add_argument\("--([a-z_0-9]+)"', open("train_v12.py", encoding="utf-8").read()))
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

print("=" * 90); print("v2 E2 MM-IMDb [%s] — 원본 zero-fill json 탐색" % BB); print("=" * 90)
chosen = {}
for s in SEEDS:
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
unknown = sorted(k for k in chosen[SEEDS[0]] if k not in known and k not in SKIP_KEYS)
if unknown:
    print("참고: 현재 train_v12에 없는 config 키(무시):", unknown)

os.makedirs(OUTDIR, exist_ok=True)
for m in METHODS:
    for s in SEEDS:
        out = "%s/%s_%s_s%d.json" % (OUTDIR, BB, m, s)
        cmd = [sys.executable, "train_v12.py"]
        for k, v in chosen[s].items():
            if k in SKIP_KEYS or k not in known or v is None or v is False:
                continue
            cmd += ["--" + k] if v is True else ["--" + k, str(v)]
        cmd += ["--method", m, "--seed", str(s), "--out", out]
        print("\n$ " + " ".join(cmd), flush=True)
        if DRY:
            continue
        if os.path.exists(out):
            print("  (이미 있음, 건너뜀)"); continue
        rc = subprocess.call(cmd)
        print("  → 종료코드 %d" % rc, flush=True)
print("\n=== ALL DONE (V2 E2 MM-IMDb %s%s) ===" % (BB, " DRY" if DRY else ""))
