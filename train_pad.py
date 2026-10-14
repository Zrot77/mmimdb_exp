"""B1·B2·B3 — PAD-UFES-20 (이미지 + 임상메타) 결손 강건 학습.
이미지 = SigLIP so400m L0(마지막 3블록+head+post_ln FT), 메타 = 2층 MLP(h256,ReLU,do0.1)→256.
융합 = concat(이미지proj256, 메타256) → Linear(512,6) softmax, 손실 CE.
γ = 각 모달 표현에 6클래스 보조 head CE × γ. (결정: InfoNCE·align 제외 — 이미지-메타 정렬 근거 약함, 지시서 §B1 CE 명세 따름)
결손(zero-fill, MM-IMDb infer_logits와 동일하게 투영공간 0): full / image_missing(pi=0) / meta_missing(pm=0).
  보조기록(판정 미사용): meta_missing_input0 = meta_mlp(입력 0벡터) — B2 json의 meta_missing 정의(v1, 2026-10-14 변경 전).
suff probe = 다항 로지스틱 회귀(train fit, test macro-F1): image_raw(학습전 고정 이미지특징)·meta·학습후(pi,pm).
평가: macro-F1(주), balanced_acc, macro AUROC(ovr), 클래스별 F1. 반복=5 fold(fold당 seed 0 고정 = B2와 동일, --seed), fold 짝지음.
γ 선택용: val(최적 epoch 체크포인트)에서도 동일 결손평가를 res['val_eval']에 저장 (테스트로 선택 금지).
기록: elapsed_sec, gpu, torch/numpy 버전, git commit, checkpoint 경로(--save_model 시 학습가능 파라미터만 저장).

실행: python train_pad.py --data_dir $HOME/Yechan/mmimdb_data/data --fold 0 --method zerofill --out results/thesis_round/B2/pad_zerofill_f0.json
"""
import argparse, glob, json, os, subprocess, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score

from data_pad import build_meta, DIAGNOSES, LABEL, IMG
import pandas as pd

SIGLIP = "google/siglip-so400m-patch14-384"


def img_index(data_dir):
    idx = {}
    for p in glob.glob(os.path.join(data_dir, "image", "imgs_part_*", "*.png")):
        idx[os.path.basename(p)] = p
    return idx


class PadDS(Dataset):
    def __init__(self, df, rows, meta, y, iindex, proc):
        self.df = df.iloc[rows].reset_index(drop=True)
        self.meta = meta[rows]; self.y = y[rows]; self.ii = iindex; self.proc = proc
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(self.ii[self.df[IMG][i]]).convert("RGB")
        px = self.proc(images=img)["pixel_values"][0]              # numpy (3,H,W), 프로세서 내부 텐서변환 우회
        px = torch.as_tensor(np.asarray(px, dtype=np.float32))
        return px, torch.as_tensor(self.meta[i], dtype=torch.float32), int(self.y[i])


class PadModel(nn.Module):
    def __init__(self, meta_dim, n_cls=6, unfreeze_last=3, dim=256, siglip_name=SIGLIP):
        super().__init__()
        from transformers import AutoModel
        self.siglip = AutoModel.from_pretrained(siglip_name)
        for p in self.siglip.parameters(): p.requires_grad_(False)
        vm = self.siglip.vision_model
        for blk in vm.encoder.layers[-unfreeze_last:]:
            for p in blk.parameters(): p.requires_grad_(True)
        for nm in ("post_layernorm", "head"):
            m = getattr(vm, nm, None)
            if m is not None:
                for p in m.parameters(): p.requires_grad_(True)
        img_dim = self.siglip.config.vision_config.hidden_size
        self.pi = nn.Linear(img_dim, dim)
        self.meta_mlp = nn.Sequential(nn.Linear(meta_dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, dim))
        self.cls = nn.Linear(2 * dim, n_cls)
        self.head_i = nn.Linear(dim, n_cls)
        self.head_m = nn.Linear(dim, n_cls)

    def enc_img(self, px): return self.siglip.get_image_features(pixel_values=px)
    def proj(self, zi, meta): return self.pi(zi), self.meta_mlp(meta)
    def classify(self, a, b): return self.cls(torch.cat([a, b], 1))


@torch.no_grad()
def extract(model, loader, device, raw=False):
    model.eval(); zis, pis, pms, ys = [], [], [], []
    for px, meta, y in loader:
        zi = model.enc_img(px.to(device))
        pi, pm = model.proj(zi, meta.to(device).float())
        zis.append(zi.cpu()); pis.append(pi.cpu()); pms.append(pm.cpu()); ys.append(y)
    return torch.cat(zis), torch.cat(pis), torch.cat(pms), torch.cat(ys)


def suff(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")
    clf.fit(Xtr.numpy(), ytr.numpy())
    return float(f1_score(yte.numpy(), clf.predict(Xte.numpy()), average="macro"))


def metrics(logits, y, n_cls=6):
    prob = torch.softmax(logits, 1).numpy(); pred = prob.argmax(1); yt = y.numpy()
    out = {"f1_macro": float(f1_score(yt, pred, average="macro")),
           "balanced_acc": float(balanced_accuracy_score(yt, pred))}
    try:
        out["auroc_macro"] = float(roc_auc_score(np.eye(n_cls)[yt], prob, average="macro", multi_class="ovr"))
    except Exception:
        out["auroc_macro"] = float("nan")
    out["f1_per_class"] = f1_score(yt, pred, average=None, labels=list(range(n_cls))).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--splits", default="thesis_round/B0/splits.json")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--method", choices=["zerofill", "gamma", "moddrop", "gamma_moddrop"], default="zerofill")
    ap.add_argument("--gamma", type=float, default=8.0)
    ap.add_argument("--moddrop_p", type=float, default=0.3)
    ap.add_argument("--siglip_name", default=SIGLIP)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="스모크: 각 split 앞 N개만")
    ap.add_argument("--out", default="results/thesis_round/B/pad.json")
    ap.add_argument("--seed", type=int, default=0, help="B2와 동일하게 0 (변경 금지: B2-B3 연결)")
    ap.add_argument("--save_model", default="", help="경로 지정 시 학습가능 파라미터(+head)만 저장")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t_start = time.time()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sp = json.load(open(args.splits)); fold = sp["folds"][args.fold]
    if args.limit:                                   # 스모크: 각 split 앞 N개
        for k in ("train", "val", "test"):
            fold[k] = fold[k][:args.limit]
    df = pd.read_csv(os.path.join(args.data_dir, "metadata.csv")).reset_index(drop=True)
    y = df[LABEL].map({d: i for i, d in enumerate(DIAGNOSES)}).values.astype(np.int64)
    meta, names = build_meta(df, np.array(fold["train"]))
    iindex = img_index(args.data_dir)
    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(args.siglip_name)

    def mk(rows, shuf): return DataLoader(PadDS(df, rows, meta, y, iindex, proc), batch_size=args.batch,
                                          shuffle=shuf, num_workers=args.workers)
    tr = mk(np.array(fold["train"]), True); va = mk(np.array(fold["val"]), False); te = mk(np.array(fold["test"]), False)

    model = PadModel(meta.shape[1], len(DIAGNOSES), siglip_name=args.siglip_name).to(device)

    # ---- raw suff (학습 전): image_raw(pretrained 고정 특징), meta ----
    zi_tr0, _, _, ytr = extract(model, mk(np.array(fold["train"]), False), device)
    zi_te0, _, _, yte = extract(model, te, device)
    suff_img_raw = suff(zi_tr0, ytr, zi_te0, yte)
    mtr = torch.tensor(meta[np.array(fold["train"])]); mte = torch.tensor(meta[np.array(fold["test"])])
    suff_meta = suff(mtr, ytr, mte, yte)

    # ---- 학습 ----
    head_mods = [model.pi, model.meta_mlp, model.cls, model.head_i, model.head_m]
    hp = [p for m in head_mods for p in m.parameters()]; hid = set(id(p) for p in hp)
    bb = [p for p in model.parameters() if p.requires_grad and id(p) not in hid]
    opt = torch.optim.Adam([{"params": bb, "lr": args.lr}, {"params": hp, "lr": args.head_lr}], weight_decay=args.wd)

    @torch.no_grad()
    def eval_full_val(loader):
        model.eval(); L, Y = [], []
        for px, mt, yy in loader:
            zi = model.enc_img(px.to(device)); pi, pm = model.proj(zi, mt.to(device).float())
            L.append(model.classify(pi, pm).cpu()); Y.append(yy)
        return metrics(torch.cat(L), torch.cat(Y))["f1_macro"]

    best = {"val": -1, "state": None}
    for ep in range(args.epochs):
        model.train(); tot = 0
        for px, mt, yy in tr:
            px, mt, yy = px.to(device), mt.to(device).float(), yy.to(device)
            zi = model.enc_img(px); pi, pm = model.proj(zi, mt)
            a, b = pi, pm
            if args.method in ("moddrop", "gamma_moddrop"):
                n = pi.size(0)
                di = (torch.rand(n, 1, device=device) < args.moddrop_p).float()
                dm = (torch.rand(n, 1, device=device) < args.moddrop_p).float()
                both = ((di + dm) > 1.5).float(); di, dm = di * (1 - both), dm * (1 - both)
                a, b = pi * (1 - di), pm * (1 - dm)
            loss = F.cross_entropy(model.classify(a, b), yy)
            if args.method in ("gamma", "gamma_moddrop"):
                loss = loss + args.gamma * (F.cross_entropy(model.head_i(pi), yy) + F.cross_entropy(model.head_m(pm), yy))
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach())
        vf = eval_full_val(va)
        if vf > best["val"]:
            best.update(val=vf, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        print("fold%d ep%d loss%.3f val_f1 %.3f (best %.3f)" % (args.fold, ep + 1, tot / len(tr), vf, best["val"]))
    model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})

    # ---- 결손 평가: full / image_missing(pi=0) / meta_missing(pm=0) [+보조 meta_missing_input0] ----
    @torch.no_grad()
    def eval_cases(pi_, pm_, y_):
        pi_, pm_ = pi_.to(device), pm_.to(device)
        pm_in0 = model.meta_mlp(torch.zeros(pi_.size(0), meta.shape[1], device=device))
        return {"full": metrics(model.classify(pi_, pm_).cpu(), y_),
                "image_missing": metrics(model.classify(torch.zeros_like(pi_), pm_).cpu(), y_),
                "meta_missing": metrics(model.classify(pi_, torch.zeros_like(pm_)).cpu(), y_),
                "meta_missing_input0": metrics(model.classify(pi_, pm_in0).cpu(), y_)}

    _, pi_te, pm_te, yte2 = extract(model, te, device)
    ev = eval_cases(pi_te, pm_te, yte2)
    _, pi_va, pm_va, yva = extract(model, va, device)
    ev_val = eval_cases(pi_va, pm_va, yva)          # γ 선택 등 모든 선택은 이것으로만

    # ---- 학습후 suff ----
    pi_tr, pm_tr = extract(model, mk(np.array(fold["train"]), False), device)[1:3]
    suff_img_after = suff(pi_tr, ytr, pi_te, yte2)
    suff_meta_after = suff(pm_tr, ytr, pm_te, yte2)

    res = {"method": args.method, "fold": args.fold, "config": vars(args), "eval": ev,
           "suff": {"image_raw": suff_img_raw, "meta": suff_meta,
                    "image_after": suff_img_after, "meta_after": suff_meta_after},
           "val_best_f1": best["val"], "val_eval": ev_val, "classes": DIAGNOSES}
    ckpt = None
    if args.save_model:
        frozen = {n for n, p_ in model.named_parameters() if not p_.requires_grad}
        sd = {k: v for k, v in best["state"].items() if k not in frozen}
        os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
        torch.save(sd, args.save_model); ckpt = args.save_model
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    res["run"] = {"elapsed_sec": round(time.time() - t_start, 1),
                  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                  "torch": torch.__version__, "numpy": np.__version__, "git_commit": commit,
                  "checkpoint": ckpt, "seed": args.seed}
    json.dump(res, open(args.out, "w"), indent=2, default=float)
    print(">>> saved", args.out)
    print("  suff image_raw %.3f | meta %.3f || full %.3f img_miss %.3f meta_miss %.3f (input0 %.3f) | %.0fs" % (
        suff_img_raw, suff_meta, ev["full"]["f1_macro"], ev["image_missing"]["f1_macro"], ev["meta_missing"]["f1_macro"],
        ev["meta_missing_input0"]["f1_macro"], res["run"]["elapsed_sec"]))


if __name__ == "__main__":
    main()
