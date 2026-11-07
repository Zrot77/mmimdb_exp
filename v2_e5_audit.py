"""추가 실험 v2 — E5 수치 감사: Phase A(MM-IMDb L0) 수치를 원본 결과 json에서 config의 seed 기준으로 다시 계산 (학습 없음)
  python v2_e5_audit.py             # 서버 ~/Yechan/mmimdb_exp
  python v2_e5_audit.py --selftest
[고정 규칙 — 실행 전, 2026-11-07]
- 대상: results/ 아래 모든 json(이번 V2 재학습 산출물 results/thesis_round/V2 는 제외 → 원본 출처만 사용).
- 조건 매칭: mode=real, backbone, method와 조건값(γ, moddrop_p, noise_sigma)이 맞고, 핵심 키
  (unfreeze_last, dim, epochs, lr, head_lr, batch, w_align, beta, temp)가 같은 백본의 zero-fill 기준 config와 모두 '있고 같아야' 함
  (다른 파이프라인·스모크·p 탐색 run 배제). seed는 파일 이름이 아니라 config의 seed를 쓴다.
- 같은 조건·seed에 파일이 여러 개면 값(text·image·full macro)이 모두 같은지 확인하고, 다르면 '★값 불일치'로 표시(첫 파일 사용).
- 통계: ddof=1, paired CI n=3 t=4.303. 문서값과 함께 출력(판정을 새로 하지 않음 — 감사 목적).
- 계산: 조건별 mean±sd 표 / A1 I·M(CLIP γ8) / A4 M(CLIP γ16) / A2 gap = 학습 후 이미지 suff(z_image) − text 결손 (γ2·8·16·32)
        / A3 noise(σ 0.3) − zf, − γ8, − γ16 (CLIP·SigLIP) / 기존 factorial I(CLIP γ2, SigLIP γ8).
"""
import glob, json, os, sys
import numpy as np

T3 = 4.303
ESS = ("unfreeze_last", "dim", "epochs", "lr", "head_lr", "batch", "w_align", "beta", "temp")
CONDS = {  # 이름: (backbone, method, {조건 키: 값})
    "clip_zf": ("clip", "zerofill", {}), "clip_g2": ("clip", "gamma", {"gamma": 2.0}), "clip_g8": ("clip", "gamma", {"gamma": 8.0}),
    "clip_g16": ("clip", "gamma", {"gamma": 16.0}), "clip_g32": ("clip", "gamma", {"gamma": 32.0}),
    "clip_md": ("clip", "moddrop", {"moddrop_p": 0.3}), "clip_gm2": ("clip", "gamma_moddrop", {"gamma": 2.0, "moddrop_p": 0.3}),
    "clip_gm8": ("clip", "gamma_moddrop", {"gamma": 8.0, "moddrop_p": 0.3}), "clip_gm16": ("clip", "gamma_moddrop", {"gamma": 16.0, "moddrop_p": 0.3}),
    "clip_noise": ("clip", "noise", {"noise_sigma": 0.3}),
    "sig_zf": ("siglip", "zerofill", {}), "sig_g8": ("siglip", "gamma", {"gamma": 8.0}), "sig_g16": ("siglip", "gamma", {"gamma": 16.0}),
    "sig_md": ("siglip", "moddrop", {"moddrop_p": 0.3}), "sig_gm8": ("siglip", "gamma_moddrop", {"gamma": 8.0, "moddrop_p": 0.3}),
    "sig_noise": ("siglip", "noise", {"noise_sigma": 0.3}),
}
SIG = "siglip-so400m-patch14-384"


def ci(d):
    d = np.asarray(d, float); m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d)); return float(m), float(m - T3 * se), float(m + T3 * se)


def collect(root):
    v2 = os.path.normpath(os.path.join(root, "results/thesis_round/V2")); recs = []
    for f in sorted(glob.glob(os.path.join(root, "results", "**", "*.json"), recursive=True)):
        if os.path.normpath(f).startswith(v2):
            continue
        try:
            r = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        c = r.get("config") if isinstance(r, dict) else None
        ev = r.get("eval", {}) if isinstance(r, dict) else {}
        if not isinstance(c, dict) or c.get("mode") != "real" or not all(k in ev for k in ("text", "image", "full")) or c.get("limit"):
            continue
        if c.get("backbone") == "siglip" and os.path.basename(str(c.get("siglip_name"))) != SIG:
            continue
        recs.append((os.path.relpath(f, root), c, r))
    return recs


def run(root):
    recs = collect(root)
    print("=" * 100); print("E5 수치 감사 — Phase A 재계산 (원본 json %d개 후보, config seed 기준, ddof=1)" % len(recs)); print("=" * 100)
    ref = {}
    for bb in ("clip", "siglip"):
        zs = [c for _, c, _ in recs if c.get("backbone") == bb and c.get("method") == "zerofill" and all(k in c for k in ESS)]
        if not zs:
            print("★ %s zero-fill 기준 config 없음 → %s 조건은 건너뜀" % (bb, bb)); continue
        ref[bb] = {k: zs[0][k] for k in ESS}
        bad = [c for c in zs if any(c[k] != ref[bb][k] for k in ESS)]
        print("기준 config[%s] %s %s" % (bb, ref[bb], "" if not bad else "(★ zero-fill 중 핵심 키가 다른 run %d개 — 제외됨)" % len(bad)))
    V = {}
    for name, (bb, meth, cond) in CONDS.items():
        by = {}
        if bb not in ref:
            V[name] = {}; continue
        for rel, c, r in recs:
            if c.get("backbone") != bb or c.get("method") != meth or not all(k in c and c[k] == ref[bb][k] for k in ESS):
                continue
            if any(abs(float(c.get(k, -999)) - v) > 1e-9 for k, v in cond.items()):
                continue
            by.setdefault(int(c["seed"]), []).append((rel, r))
        V[name] = {}
        for s, lst in sorted(by.items()):
            vals = [tuple(round(x["eval"][k]["macro"], 6) for k in ("text", "image", "full")) for _, x in lst]
            flag = "" if len(set(vals)) == 1 else " ★값 불일치 %s" % sorted(set(vals))
            r0 = lst[0][1]
            V[name][s] = {"text": r0["eval"]["text"]["macro"], "image": r0["eval"]["image"]["macro"], "full": r0["eval"]["full"]["macro"],
                          "suff_img": (r0.get("suff") or {}).get("z_image"), "files": [x for x, _ in lst]}
            if flag:
                print("  [%s s%d] 파일 %d개%s" % (name, s, len(lst), flag))
    print("\n[1] 조건별 mean ± sd(ddof=1) — seed: text 결손 / image 결손 / full")
    for name, d in V.items():
        ss = sorted(d)
        if not ss:
            print("  %-11s 결과 없음" % name); continue
        row = " | ".join("%s %.4f±%.4f" % (k, np.mean([d[s][k] for s in ss]), np.std([d[s][k] for s in ss], ddof=1) if len(ss) > 1 else 0)
                         for k in ("text", "image", "full"))
        print("  %-11s seed %s (%s) | %s" % (name, ss, ", ".join(sorted({os.path.dirname(f) or "." for s in ss for f in d[s]["files"]})), row))
    S = [1, 2, 3]
    have = lambda *ns: all(all(s in V[n] for s in S) for n in ns)
    x = lambda n, k="text": np.array([V[n][s][k] for s in S])
    print("\n[2] 재계산 (문서값과 비교)")
    if have("clip_zf", "clip_g8", "clip_md", "clip_gm8"):
        I = ci(x("clip_gm8") - x("clip_g8") - x("clip_md") + x("clip_zf")); M = ci(x("clip_gm8") - x("clip_g8"))
        print("  A1 CLIP γ8: I %+.4f [%+.4f,%+.4f] (문서 −0.110) | M %+.4f [%+.4f,%+.4f] (문서 +0.018)" % (*I, *M))
    if have("clip_g16", "clip_gm16"):
        M = ci(x("clip_gm16") - x("clip_g16")); print("  A4 CLIP γ16: M %+.4f [%+.4f,%+.4f] (문서 +0.008 [−0.003, +0.020])" % M)
    for g, doc in (("clip_g2", "+0.051"), ("clip_g8", "-"), ("clip_g16", "−0.005"), ("clip_g32", "-")):
        if have(g) and all(V[g][s]["suff_img"] is not None for s in S):
            gap = np.array([V[g][s]["suff_img"] - V[g][s]["text"] for s in S])
            print("  A2 %s gap(suff_img − text) %+.4f ± %.4f (문서 %s)" % (g, gap.mean(), gap.std(ddof=1), doc))
    for bb, zf, g8, g16, nz, doc in (("CLIP", "clip_zf", "clip_g8", "clip_g16", "clip_noise", "−0.210"), ("SigLIP", "sig_zf", "sig_g8", "sig_g16", "sig_noise", "−0.116")):
        if have(nz, zf):
            parts = ["noise − zf %+.4f [%+.4f,%+.4f]" % ci(x(nz) - x(zf))]
            for g in (g8, g16):
                if have(g):
                    parts.append("noise − %s %+.4f [%+.4f,%+.4f]" % ((g.split("_")[1],) + ci(x(nz) - x(g))))
            print("  A3 %s: %s (문서 noise − γ %s)" % (bb, " | ".join(parts), doc))
    if have("clip_zf", "clip_g2", "clip_md", "clip_gm2"):
        I = ci(x("clip_gm2") - x("clip_g2") - x("clip_md") + x("clip_zf")); print("  기존 factorial CLIP γ2: I %+.4f [%+.4f,%+.4f] (문서 −0.074)" % I)
    if have("sig_zf", "sig_g8", "sig_md", "sig_gm8"):
        I = ci(x("sig_gm8") - x("sig_g8") - x("sig_md") + x("sig_zf")); print("  기존 factorial SigLIP γ8: I %+.4f [%+.4f,%+.4f] (문서 −0.094)" % I)
        Ii = ci(x("sig_gm8", "image") - x("sig_g8", "image") - x("sig_md", "image") + x("sig_zf", "image"))
        print("  기존 factorial SigLIP γ8 image 결손: I %+.4f [%+.4f,%+.4f] (문서 +0.068)" % Ii)
    print("\n※ 문서값이 다르게 나오면 원인(seed 짝, ddof, 파일 선택)을 확인할 것. frozen 파이프라인(3차) 수치는 대상 아님.")
    return V


def selftest():
    import tempfile, shutil
    root = tempfile.mkdtemp(prefix="e5_"); os.makedirs(os.path.join(root, "results/thesis_round/A1")); os.makedirs(os.path.join(root, "results/thesis_round/V2"))
    base = {"mode": "real", "backbone": "clip", "unfreeze_last": 3, "dim": 256, "epochs": 8, "lr": 1e-5, "head_lr": 1e-3, "batch": 64,
            "w_align": 1.0, "beta": 0.3, "temp": 0.07, "gamma": 2.0, "moddrop_p": 0.3, "noise_sigma": 0.0}
    vals = {"zerofill": [.24, .24, .29], ("gamma", 8.0): [.44, .45, .45], "moddrop": [.38, .37, .40], ("gamma_moddrop", 8.0): [.461, .461, .470]}
    def w(path, cfg, t):
        json.dump({"config": cfg, "eval": {"text": {"macro": t}, "image": {"macro": .45}, "full": {"macro": .46}}, "suff": {"z_image": t + .01}},
                  open(os.path.join(root, path), "w"))
    for i, s in enumerate((1, 2, 3)):
        w("results/v12_zerofill_s%d.json" % s, dict(base, method="zerofill", seed=s), vals["zerofill"][i])
        w("results/v13_gamma_g8.0_s%d.json" % s, dict(base, method="gamma", gamma=8.0, seed=s), vals[("gamma", 8.0)][i])
        w("results/v12_moddrop_p0.3_s%d.json" % s, dict(base, method="moddrop", seed=s), vals["moddrop"][i])
        w("results/v12_moddrop_p0.5_s%d.json" % s, dict(base, method="moddrop", moddrop_p=0.5, seed=s), .99)      # p 탐색 run → 제외돼야 함
        w("results/v9_frozen_zerofill_s%d.json" % s, {"mode": "real", "backbone": "clip", "method": "zerofill", "seed": s}, .99)   # 다른 파이프라인 → 제외
        w("results/thesis_round/V2/clip_gamma_moddrop_g8.0_s%d.json" % s, dict(base, method="gamma_moddrop", gamma=8.0, seed=s), .99)  # V2 → 제외
    # A1: 파일 이름과 seed가 뒤바뀐 경우 재현(s1 파일에 seed 3, s3 파일에 seed 1)
    for fname, s in (("clip_g8_md_s1.json", 3), ("clip_g8_md_s2.json", 2), ("clip_g8_md_s3.json", 1)):
        w("results/thesis_round/A1/" + fname, dict(base, method="gamma_moddrop", gamma=8.0, seed=s), vals[("gamma_moddrop", 8.0)][s - 1])
    V = run(root); shutil.rmtree(root, ignore_errors=True)
    gm = V["clip_gm8"]; zf = V["clip_zf"]; md = V["clip_md"]
    ok = [gm[1]["text"] == .461 and gm[3]["text"] == .470, all(md[s]["text"] < .9 for s in md), all(zf[s]["text"] < .9 for s in zf),
          all("thesis_round/V2" not in f for s in gm for f in gm[s]["files"])]
    print("\n자체 시험 판정:", "모두 통과" if all(ok) else "★실패 %s" % ok)
    return all(ok)


if __name__ == "__main__":
    sys.exit(0 if selftest() else 1) if "--selftest" in sys.argv else run(".")
    print("=== ALL DONE (E5 감사) ===")
