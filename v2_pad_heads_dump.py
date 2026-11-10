"""추가 실험 v4 — PAD 기존 체크포인트에서 보조 분류기(head_i·head_m) 출력 추출 (학습 없음, GPU 추론)
  python v2_pad_heads_dump.py          # 서버 ~/Yechan/mmimdb_exp (conda mmimdb), GPU 1개
규칙: thesis_round/ADDENDUM_v4_MAI_prereg.md — 이 스크립트는 그 문서를 커밋한 뒤 실행한다.
출력: results/thesis_round/V2/dumpH/<tag>_f<fold>.npz
  키: {val,test}_{full, image_missing, meta_missing_input0, meta_missing_proj0, headi, headm, y}
  headi = head_i(pi) (이미지만 보는 보조 분류기, meta 결손 때 남는 모달), headm = head_m(pm) (메타만, image 결손 때 남는 모달)
블라인드: 보조 분류기 성능은 출력하지 않는다. K0(융합 로짓이 기존 V2/dump와 같은지)와 개수만 출력.
대상: γ8 seed 0~2 × fold 0~4 (v2_pad_dump.py의 CKPTS와 같은 경로). 결합(γ8+ModDrop) 체크포인트가 B3/ckpt에 하나로 특정되면 gm_s0으로 함께 추출.
이미 있는 출력은 건너뜀(재실행 안전). 기존 파일은 읽기만 함.
"""
import glob, hashlib, json, os, sys
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from train_pad import PadModel, PadDS, img_index, extract, SIGLIP
from data_pad import build_meta, DIAGNOSES, LABEL

DATA = os.path.expanduser("~/Yechan/mmimdb_data/data")
OUT, OLD = "results/thesis_round/V2/dumpH", "results/thesis_round/V2/dump"
PREREG = "thesis_round/ADDENDUM_v4_MAI_prereg.md"
CKPTS = {"g8_s0": "results/thesis_round/B3/ckpt/gamma_g8.0_f{f}.pt",
         "g8_s1": "results/thesis_round/V2/E3/ckpt/gamma_g8.0_s1_f{f}.pt",
         "g8_s2": "results/thesis_round/V2/E3/ckpt/gamma_g8.0_s2_f{f}.pt"}
FUSED = ("full", "image_missing", "meta_missing_input0", "meta_missing_proj0")


def gm_pattern():
    c = sorted(set(glob.glob("results/thesis_round/B3/ckpt/gamma_moddrop*_f0.pt")))
    return (c[0][:-len("_f0.pt")] + "_f{f}.pt") if len(c) == 1 else None, c


def main():
    if not os.path.exists(PREREG):
        sys.exit("★ 사전 고정 문서 %s 없음 → 중단(먼저 git pull)" % PREREG)
    print("사전 고정 문서 md5 %s" % hashlib.md5(open(PREREG, "rb").read()).hexdigest())
    gp, cand = gm_pattern()
    if gp:
        CKPTS["gm_s0"] = gp; print("결합 체크포인트 패턴:", gp)
    else:
        print("결합 체크포인트를 하나로 특정 못 함(후보 %s) → 결합은 추출하지 않음" % cand)
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = json.load(open("thesis_round/B0/splits.json"))
    df = pd.read_csv(os.path.join(DATA, "metadata.csv")).reset_index(drop=True)
    y = df[LABEL].map({d: i for i, d in enumerate(DIAGNOSES)}).values.astype(np.int64)
    iindex = img_index(DATA)
    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(SIGLIP)
    done, skipped, worst = 0, [], 0.0
    for f in range(5):
        fold = sp["folds"][f]
        meta, _ = build_meta(df, np.array(fold["train"]))
        mk = lambda rows: DataLoader(PadDS(df, np.array(rows), meta, y, iindex, proc), batch_size=16, shuffle=False, num_workers=4)
        loaders = None
        for tag, pat in CKPTS.items():
            path, out = pat.format(f=f), os.path.join(OUT, "%s_f%d.npz" % (tag, f))
            if os.path.exists(out):
                continue
            if not os.path.exists(path):
                skipped.append(path); continue
            loaders = loaders or {"val": mk(fold["val"]), "test": mk(fold["test"])}
            model = PadModel(meta.shape[1], len(DIAGNOSES), siglip_name=SIGLIP).to(device)
            sd = torch.load(path, map_location=device); res = model.load_state_dict(sd, strict=False)
            frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
            assert not res.unexpected_keys, ("예상 밖 키", res.unexpected_keys[:5])
            bad = [k for k in res.missing_keys if k not in frozen and not k.endswith("position_ids")]
            assert not bad, ("체크포인트에 없는 학습 파라미터", bad[:5])
            assert "head_i.weight" in sd and "head_m.weight" in sd, "체크포인트에 보조 분류기 없음"
            arr = {}
            for split, ld in loaders.items():
                _, pi, pm, yy = extract(model, ld, device)
                pi, pm = pi.to(device), pm.to(device)
                with torch.no_grad():
                    pm_in0 = model.meta_mlp(torch.zeros(pi.size(0), meta.shape[1], device=device))
                    arr[split + "_full"] = model.classify(pi, pm).cpu().numpy()
                    arr[split + "_image_missing"] = model.classify(torch.zeros_like(pi), pm).cpu().numpy()
                    arr[split + "_meta_missing_input0"] = model.classify(pi, pm_in0).cpu().numpy()
                    arr[split + "_meta_missing_proj0"] = model.classify(pi, torch.zeros_like(pm)).cpu().numpy()
                    arr[split + "_headi"] = model.head_i(pi).cpu().numpy()
                    arr[split + "_headm"] = model.head_m(pm).cpu().numpy()
                arr[split + "_y"] = yy.numpy()
            k0 = "K0 기존 로짓 없음(gm 등)"
            old = os.path.join(OLD, "%s_f%d.npz" % (tag, f))
            if os.path.exists(old):
                o = dict(np.load(old))
                assert all(np.array_equal(o[s + "_y"], arr[s + "_y"]) for s in ("val", "test")), "K3 라벨 불일치"
                d = max(float(np.abs(o["%s_%s" % (s, k)].astype(np.float64) - arr["%s_%s" % (s, k)].astype(np.float64)).max())
                        for s in ("val", "test") for k in FUSED)
                worst = max(worst, d); k0 = "K0 융합 로짓 최대 |차| %.2e%s" % (d, "" if d <= 1e-4 else " ★경고")
            np.savez_compressed(out, **arr); done += 1
            print(">>> %s f%d 저장 | %s" % (tag, f, k0), flush=True)
            del model; torch.cuda.empty_cache()
    print("새로 추출 %d개 | 체크포인트 없음 %d개 | K0 최대 |차| %.2e%s" % (done, len(skipped), worst, "" if worst <= 1e-4 else " ★경고"))
    for p in skipped:
        print("   없음:", p)
    n = len(glob.glob(os.path.join(OUT, "g8_s*_f*.npz")))
    print("γ8 출력 파일 %d/15개" % n)
    print("=== ALL DONE (v4 PAD 보조 분류기 추출) ===")


if __name__ == "__main__":
    main()
