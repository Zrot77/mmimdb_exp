"""B0 — PAD-UFES-20 분할·전처리·누설검증.
- 제외: 식별자(patient_id,lesion_id,img_id), biopsed(라벨 누설), diagnostic(라벨)
- 수치형: age,diameter_1,diameter_2,fitspatrick → fold-train 표준화 + 결측 0 + 결측지시자
- 범주형: 나머지 → one-hot, 결측/UNK 별도 범주
- 분할: 5-fold StratifiedGroupKFold(group=patient_id, stratify=diagnostic, random_state 고정),
        각 fold train 내 환자단위 15% val. 누설검증(환자 교집합 0).
전처리 통계는 해당 fold train에서만 fit. build_meta()는 B1에서 재사용.

실행(분할 생성+검증): python data_pad.py --data_dir data --out thesis_round/B0
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

DIAGNOSES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
EXCLUDE = ["patient_id", "lesion_id", "img_id", "diagnostic", "biopsed"]
NUMERIC = ["age", "diameter_1", "diameter_2", "fitspatrick"]
LABEL, GROUP, IMG = "diagnostic", "patient_id", "img_id"
SEED = 42


def categorical_cols(df):
    return [c for c in df.columns if c not in EXCLUDE and c not in NUMERIC]


def build_meta(df, train_idx):
    """fold train(train_idx)에서 통계 fit → 전체 df 변환. 반환: (meta[np.float32], feature_names)."""
    cats = categorical_cols(df)
    cols, names = [], []
    tr = df.loc[train_idx]
    # 수치형: train 평균/표준편차로 표준화, 결측 0 + 지시자
    for c in NUMERIC:
        x = pd.to_numeric(df[c], errors="coerce")
        xt = pd.to_numeric(tr[c], errors="coerce")
        mu, sd = xt.mean(), xt.std(ddof=0)
        sd = sd if sd and not np.isnan(sd) else 1.0
        miss = x.isna().astype(np.float32).values
        z = ((x - mu) / sd).fillna(0.0).astype(np.float32).values
        cols += [z, miss]; names += [c + "_z", c + "_missing"]
    # 범주형: train 범주로 one-hot(+결측/UNK), test의 미지 범주는 UNK
    for c in cats:
        vals = tr[c].astype("object").where(tr[c].notna(), "MISSING").astype(str)
        levels = sorted(vals.unique().tolist())
        if "MISSING" not in levels:
            levels.append("MISSING")
        levels.append("UNK")
        col = df[c].astype("object").where(df[c].notna(), "MISSING").astype(str)
        col = col.where(col.isin(levels), "UNK")
        for lv in levels:
            cols.append((col == lv).astype(np.float32).values); names.append("%s=%s" % (c, lv))
    return np.stack(cols, 1).astype(np.float32), names


def make_splits(df):
    y = df[LABEL].values
    g = df[GROUP].values
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = []
    for k, (trv, te) in enumerate(sgk.split(df, y, g)):
        # train 내 환자단위 15% val
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
        tr_rel, va_rel = next(gss.split(df.iloc[trv], y[trv], g[trv]))
        tr = trv[tr_rel]; va = trv[va_rel]
        folds.append({"fold": k, "train": tr.tolist(), "val": va.tolist(), "test": te.tolist()})
    return folds


def leakage_check(df, folds):
    ok = True
    lines = []
    for f in folds:
        ps = {s: set(df.iloc[f[s]][GROUP]) for s in ("train", "val", "test")}
        tv = ps["train"] & ps["val"]; tt = ps["train"] & ps["test"]; vt = ps["val"] & ps["test"]
        good = not (tv or tt or vt)
        ok = ok and good
        lines.append("fold%d: train환자 %d / val %d / test %d | 교집합 tr∩val %d tr∩te %d val∩te %d → %s" % (
            f["fold"], len(ps["train"]), len(ps["val"]), len(ps["test"]), len(tv), len(tt), len(vt),
            "OK" if good else "★누설"))
    return ok, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="thesis_round/B0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(os.path.join(args.data_dir, "metadata.csv")).reset_index(drop=True)

    # 기대값 검증
    exp = dict(rows=2298, patients=1373, lesions=1641)
    got = dict(rows=len(df), patients=df[GROUP].nunique(), lesions=df["lesion_id"].nunique())
    print("기대값 검증:", got, "기대", exp, "→", "OK" if got == exp else "★불일치(중단)")
    assert set(df[LABEL].unique()) == set(DIAGNOSES), "진단 6종 불일치"

    folds = make_splits(df)
    ok, lines = leakage_check(df, folds)
    print("\n누설검증:")
    for ln in lines:
        print(" ", ln)
    print("전체 누설검증:", "통과" if ok else "★실패")

    # fold별 test 클래스 분포
    print("\nfold별 test 진단 분포:")
    for f in folds:
        vc = df.iloc[f["test"]][LABEL].value_counts().to_dict()
        print("  fold%d:" % f["fold"], {d: vc.get(d, 0) for d in DIAGNOSES})

    # meta 차원 확인(fold0 train 기준)
    meta, names = build_meta(df, np.array(folds[0]["train"]))
    print("\nmeta 차원:", meta.shape, "(수치 %d·결측지시자 포함 + 범주 one-hot)" % (2 * len(NUMERIC)))

    json.dump({"seed": SEED, "folds": folds, "meta_dim": meta.shape[1], "feature_names": names},
              open(os.path.join(args.out, "splits.json"), "w"))
    with open(os.path.join(args.out, "split_check.md"), "w", encoding="utf-8") as fh:
        fh.write("# B0 split_check\n\n기대값: %s\n\n" % str(got))
        fh.write("\n".join(lines) + "\n\n전체: %s\nmeta_dim=%d\n" % ("통과" if ok else "실패", meta.shape[1]))
    print("\n>>> saved", args.out, "/splits.json, split_check.md")


if __name__ == "__main__":
    main()
