"""추가 실험 v2/v3 — PAD 체크포인트에서 val·test 로짓 추출 (E1-PAD 부분결손, E9-PAD 재보정 분석용).
학습 없음(평가만). 이미 추출된 파일은 건너뛰므로 E3·E2가 끝난 뒤 다시 실행하면 새 체크포인트만 처리한다.
  python v2_pad_dump.py
출력: results/thesis_round/V2/dump/<tag>_f<fold>.npz  (val/test × full, image_missing, meta_missing_input0, meta_missing_proj0 로짓 + 라벨)
"""
import glob, json, os, sys
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from train_pad import PadModel, PadDS, img_index, extract, SIGLIP
from data_pad import build_meta, DIAGNOSES, LABEL

DATA = os.path.expanduser("~/Yechan/mmimdb_data/data")
OUT = "results/thesis_round/V2/dump"
CKPTS = {  # tag: 경로 패턴(fold 자리 {f})
    "zf_s0": "results/thesis_round/B3/ckpt/zerofill_f{f}.pt",
    "g8_s0": "results/thesis_round/B3/ckpt/gamma_g8.0_f{f}.pt",
    "md_s0": "results/thesis_round/B3/ckpt/moddrop_f{f}.pt",
    "zf_s1": "results/thesis_round/V2/E3/ckpt/zerofill_s1_f{f}.pt",
    "g8_s1": "results/thesis_round/V2/E3/ckpt/gamma_g8.0_s1_f{f}.pt",
    "md_s1": "results/thesis_round/V2/E3/ckpt/moddrop_s1_f{f}.pt",
    "zf_s2": "results/thesis_round/V2/E3/ckpt/zerofill_s2_f{f}.pt",
    "g8_s2": "results/thesis_round/V2/E3/ckpt/gamma_g8.0_s2_f{f}.pt",
    "md_s2": "results/thesis_round/V2/E3/ckpt/moddrop_s2_f{f}.pt",
    "onlymeta_s0": "results/thesis_round/V2/E2/ckpt/only_meta_f{f}.pt",
    "onlyimage_s0": "results/thesis_round/V2/E2/ckpt/only_image_f{f}.pt",
}


def main():
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = json.load(open("thesis_round/B0/splits.json"))
    df = pd.read_csv(os.path.join(DATA, "metadata.csv")).reset_index(drop=True)
    y = df[LABEL].map({d: i for i, d in enumerate(DIAGNOSES)}).values.astype(np.int64)
    iindex = img_index(DATA)
    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(SIGLIP)
    done, skipped = 0, []
    for f in range(5):
        fold = sp["folds"][f]
        meta, _ = build_meta(df, np.array(fold["train"]))
        mk = lambda rows: DataLoader(PadDS(df, np.array(rows), meta, y, iindex, proc), batch_size=16, shuffle=False, num_workers=4)
        loaders = None
        for tag, pat in CKPTS.items():
            path, out = pat.format(f=f), os.path.join(OUT, "%s_f%d.npz" % (tag, f))
            if os.path.exists(out): continue
            if not os.path.exists(path): skipped.append(path); continue
            loaders = loaders or {"val": mk(fold["val"]), "test": mk(fold["test"])}
            model = PadModel(meta.shape[1], len(DIAGNOSES), siglip_name=SIGLIP).to(device)
            sd = torch.load(path, map_location=device)
            res = model.load_state_dict(sd, strict=False)
            frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
            assert not res.unexpected_keys, ("예상 밖 키", res.unexpected_keys[:5])
            bad = [k for k in res.missing_keys if k not in frozen and not k.endswith("position_ids")]
            assert not bad, ("체크포인트에 없는 학습 파라미터", bad[:5])
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
                arr[split + "_y"] = yy.numpy()
            np.savez_compressed(out, **arr); done += 1
            print(">>> %s f%d 저장 (test full macro-F1 확인용 argmax 정확도 %.3f)" % (
                tag, f, (arr["test_full"].argmax(1) == arr["test_y"]).mean()), flush=True)
            del model; torch.cuda.empty_cache()
    print("새로 추출 %d개 | 체크포인트 없음(아직 학습 전이면 정상) %d개" % (done, len(skipped)))
    for p in skipped: print("   없음:", p)
    print("=== ALL DONE (V2 PAD dump) ===")


if __name__ == "__main__":
    main()
