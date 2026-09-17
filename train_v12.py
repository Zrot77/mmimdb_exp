"""v12 — 6종 비교 라운드 (결손 강건 + 두 축 커버리지).
v11 고정 backbone(CLIP L0 이미지 타워 + 텍스트 고정 앵커 + proj(256) + fusion 분류기) 위에
각 방법의 '핵심 메커니즘만' 이식해 공정 비교한다. 주 지표 = 결손 강건 F1, 대표 = text결손 macro.

방법(--method):
  zerofill : 결손 처리 없음(=v11 baseline). 하한 참조.
  gamma    : (ours) 모달별 label head CE × γ, sufficiency 직접 강화. essence 늘리기.
  moddrop  : 학습 중 각 모달 표현을 확률 p로 zero → 결손 내성. 결손 처리(masking).
  mmin     : CRA(cascade residual AE)로 없는 모달 표현 재구성 + 결손패턴 학습. 결손 처리(imputation).
  kd       : full 교사(먼저 학습·freeze) → 결손 학생을 교사 logit에 증류. essence 늘리기(distill).
  ib       : DMIB — concat f → 마스크+병목 → f*, sufficiency KL[p(y|f)‖p(y|f*)] + 브랜치 supervision. nuisance 줄이기(IB).
  gamma_moddrop : (흡수 단서) γ + ModDrop 조합.

공통 프로토콜: 3-seed, zero-fill 결손 평가(full/image/text/both), f1 macro/micro/weighted,
  val full-micro 조기중단, 차등 lr(백본 1e-5 / head 1e-3), weight decay.

실행:
  python train_v12.py --mode synthetic --backbone stub --method mmin --smoke
  python train_v12.py --mode real --backbone clip --raw_root ~/Yechan/mmimdb_data/mmimdb \
      --feat_dir feats --method moddrop --moddrop_p 0.3 --epochs 8 --early_stop --wd 1e-4 --seed 1
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_ft import ImageTextFT, SyntheticFT
from train_ft import StubVision
from utils import info_nce, align_cos, f1_all, multilabel_probe, N_GENRE

CLIP_NAME = "openai/clip-vit-base-patch32"
SIGLIP_NAME = "google/siglip-so400m-patch14-384"


# ============================ CRA (MMIN 핵심) ============================
class ResAE(nn.Module):
    """residual autoencoder 블록 — 잔차로 표현을 정제(cascade의 한 단)."""
    def __init__(self, d, h):
        super().__init__()
        self.enc = nn.Linear(d, h)
        self.dec = nn.Linear(h, d)

    def forward(self, x):
        return x + self.dec(F.relu(self.enc(x)))


class CRA(nn.Module):
    """Cascade Residual AutoEncoder (Tran+ CVPR2017, MMIN 핵심).
    관측 모달 표현(in_d) → 없는 모달 표현(out_d)을 잔차 캐스케이드로 재구성."""
    def __init__(self, in_d, out_d, hidden=256, n_blocks=3):
        super().__init__()
        self.inp = nn.Linear(in_d, out_d)
        self.blocks = nn.ModuleList([ResAE(out_d, hidden) for _ in range(n_blocks)])

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return h


# ============================ 모델 ============================
class V12Model(nn.Module):
    def __init__(self, method="zerofill", backbone="clip", unfreeze_last=3, dim=256,
                 n_cls=N_GENRE, cra_blocks=3, ib_bottleneck=128, clip_name=CLIP_NAME,
                 siglip_name=SIGLIP_NAME):
        super().__init__()
        self.method = method
        self.backbone = backbone
        self.dim = dim
        txt_dim = 512                                   # 텍스트 앵커 = CLIP 텍스트(512) 유지(강한 채로, 이미지만 교체)
        if backbone == "clip":
            from transformers import CLIPModel
            self.clip = CLIPModel.from_pretrained(clip_name)
            for p in self.clip.parameters():
                p.requires_grad_(False)
            for blk in self.clip.vision_model.encoder.layers[-unfreeze_last:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            for m in (self.clip.visual_projection, self.clip.vision_model.post_layernorm):
                for p in m.parameters():
                    p.requires_grad_(True)
            img_dim = 512
        elif backbone == "siglip":
            from transformers import SiglipModel
            self.siglip = SiglipModel.from_pretrained(siglip_name)
            for p in self.siglip.parameters():
                p.requires_grad_(False)
            vm = self.siglip.vision_model                # v11 L0 방식: 마지막 블록들 + 풀링헤드 + post_layernorm FT
            for blk in vm.encoder.layers[-unfreeze_last:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            ft_mods = [vm.post_layernorm]
            if getattr(vm, "head", None) is not None:
                ft_mods.append(vm.head)                  # 어텐션 풀링 헤드(so400m)
            for m in ft_mods:
                for p in m.parameters():
                    p.requires_grad_(True)
            img_dim = self.siglip.config.vision_config.hidden_size   # so400m = 1152
        else:
            self.vision = StubVision(512)
            img_dim = 512
        self.img_dim, self.txt_dim = img_dim, txt_dim
        # 공통 proj + fusion 분류기 (+ 브랜치 head: gamma/ib용)
        self.pi = nn.Linear(img_dim, dim)
        self.pt = nn.Linear(txt_dim, dim)
        self.cls = nn.Linear(2 * dim, n_cls)
        self.head_i = nn.Linear(dim, n_cls)
        self.head_t = nn.Linear(dim, n_cls)
        # MMIN: 양방향 CRA (text→image, image→text)
        if method == "mmin":
            self.cra_t2i = CRA(dim, dim, dim, cra_blocks)
            self.cra_i2t = CRA(dim, dim, dim, cra_blocks)
        # IB(DMIB): 마스크 게이트 + 병목(mu/logvar) + 병목 분류기
        if method == "ib":
            self.ib_gate = nn.Parameter(torch.zeros(2 * dim))          # sigmoid → soft 마스크
            self.ib_mu = nn.Linear(2 * dim, ib_bottleneck)
            self.ib_logvar = nn.Linear(2 * dim, ib_bottleneck)
            self.cls_ib = nn.Linear(ib_bottleneck, n_cls)

    def encode_image(self, px):
        if self.backbone == "clip":
            return self.clip.get_image_features(pixel_values=px)
        if self.backbone == "siglip":
            return self.siglip.get_image_features(pixel_values=px)
        return self.vision(px)

    def proj(self, zi, zt):
        return self.pi(zi), self.pt(zt)

    def classify(self, a, b):
        return self.cls(torch.cat([a, b], dim=1))

    # ---- IB(DMIB) 경로 ----
    def ib_forward(self, a, b, sample=True):
        f = torch.cat([a, b], dim=1) * torch.sigmoid(self.ib_gate)
        mu, logvar = self.ib_mu(f), self.ib_logvar(f)
        fstar = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        comp = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return self.cls_ib(fstar), comp


# ============================ 데이터 ============================
def make_loader(args, split, shuffle):
    if args.mode == "synthetic":
        ds = SyntheticFT(n={"train": 400, "dev": 200, "test": 200}[split],
                         seed={"train": 0, "dev": 1, "test": 2}[split])
    else:
        if args.backbone == "siglip":
            from transformers import AutoProcessor       # SigLIP 이미지 전처리(384px), 텍스트는 캐시 CLIP 앵커 사용
            proc = AutoProcessor.from_pretrained(args.siglip_name)
        else:
            from transformers import CLIPProcessor
            proc = CLIPProcessor.from_pretrained(CLIP_NAME)
        ds = ImageTextFT(args.feat_dir, args.raw_root, split, proc)
    if getattr(args, "limit", 0) and args.mode == "real":       # 스모크: 앞 N개만(SigLIP 경로 빠른 점검)
        from torch.utils.data import Subset
        ds = Subset(ds, list(range(min(args.limit, len(ds)))))
    return DataLoader(ds, batch_size=args.batch, shuffle=shuffle, num_workers=args.workers)


@torch.no_grad()
def extract_proj(model, loader, device):
    """clean proj 특징 (pi, pt) + 라벨 — 결손 평가·suff용 (개입 없이)."""
    model.eval()
    pis, pts, ys = [], [], []
    for px, txt, y in loader:
        zi = model.encode_image(px.to(device))
        pi, pt = model.proj(zi, txt.to(device))
        pis.append(pi.cpu()); pts.append(pt.cpu()); ys.append(y)
    return torch.cat(pis), torch.cat(pts), torch.cat(ys)


def bernoulli_kl(p, q, eps=1e-6):
    """다중라벨 Bernoulli KL[p‖q] (p=교사 확률 detach, q=학생 확률).
    라벨 평균(mean)으로 스케일을 BCE(요소평균)와 맞춤 — 안 그러면 KL이 23배 커져 KD/IB가 학습 지배."""
    p = p.clamp(eps, 1 - eps); q = q.clamp(eps, 1 - eps)
    return (p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()).mean()


# ============================ 방법별 추론(결손 평가) ============================
@torch.no_grad()
def infer_logits(model, pi, pt, regime, device):
    """regime = full/image/text/both. 방법에 맞게 결손 모달을 처리(zero 또는 impute 또는 IB)."""
    z0i, z0t = torch.zeros_like(pi), torch.zeros_like(pt)
    m = model.method
    if m == "mmin":
        if regime == "full":
            a, b = pi, pt
        elif regime == "image":                       # image 결손 → pt에서 pi 재구성
            a, b = model.cra_t2i(pt), pt
        elif regime == "text":                        # text 결손 → pi에서 pt 재구성
            a, b = pi, model.cra_i2t(pi)
        else:                                         # both 결손 → 입력 없음, zero
            a, b = z0i, z0t
        return model.classify(a, b)
    # 그 외 방법: 결손 모달 zero-fill
    a = pi if regime in ("full", "text") else z0i
    b = pt if regime in ("full", "image") else z0t
    if m == "ib":
        return model.ib_forward(a, b, sample=False)[0]
    return model.classify(a, b)


def eval_missing(model, loader, device):
    pi, pt, y = extract_proj(model, loader, device)
    pi, pt, y = pi.to(device), pt.to(device), y.to(device)
    return {r: f1_all(infer_logits(model, pi, pt, r, device), y)
            for r in ("full", "image", "text", "both")}


# ============================ 학습 ============================
def build_opt(model, args):
    head_mods = [model.pi, model.pt, model.cls, model.head_i, model.head_t]
    if model.method == "mmin":
        head_mods += [model.cra_t2i, model.cra_i2t]
    if model.method == "ib":
        head_mods += [model.ib_mu, model.ib_logvar, model.cls_ib]
    head_params = [p for mmod in head_mods for p in mmod.parameters()]
    if model.method == "ib":
        head_params.append(model.ib_gate)
    hid = set(id(p) for p in head_params)
    bb_params = [p for p in model.parameters() if p.requires_grad and id(p) not in hid]
    return torch.optim.Adam([{"params": bb_params, "lr": args.lr},
                             {"params": head_params, "lr": args.head_lr}], weight_decay=args.wd)


def base_loss(model, pi, pt, y, args):
    """모든 방법 공통 스캐폴드: fusion BCE + InfoNCE 정렬 + align_cos (v11 baseline과 동일)."""
    lc = info_nce(pi, pt, args.temp)
    return (F.binary_cross_entropy_with_logits(model.classify(pi, pt), y)
            + args.w_align * lc + args.beta * align_cos(pi, pt))


def train_step(model, px, txt, y, args, teacher=None):
    zi = model.encode_image(px)
    pi, pt = model.proj(zi, txt)
    m = model.method

    if m in ("zerofill", "gamma", "gamma_moddrop", "moddrop"):
        # moddrop: 분류 입력을 확률적으로 zero (정렬 항은 clean 유지)
        a, b = pi, pt
        if m in ("moddrop", "gamma_moddrop"):
            n = pi.size(0)
            di = (torch.rand(n, 1, device=pi.device) < args.moddrop_p).float()
            dt = (torch.rand(n, 1, device=pi.device) < args.moddrop_p).float()
            both = ((di + dt) > 1.5).float()           # 둘 다 drop 방지(한쪽은 살림)
            di, dt = di * (1 - both), dt * (1 - both)
            a, b = pi * (1 - di), pt * (1 - dt)
        lc = info_nce(pi, pt, args.temp)
        loss = (F.binary_cross_entropy_with_logits(model.classify(a, b), y)
                + args.w_align * lc + args.beta * align_cos(pi, pt))
        if m in ("gamma", "gamma_moddrop"):
            loss = loss + args.gamma * (F.binary_cross_entropy_with_logits(model.head_i(pi), y)
                                        + F.binary_cross_entropy_with_logits(model.head_t(pt), y))
        return loss

    if m == "mmin":
        # 결손패턴 학습: 샘플별 none/image/text (원 MMIN처럼 다양한 결손 패턴 노출)
        pi_hat, pt_hat = model.cra_t2i(pt), model.cra_i2t(pi)
        rec = (F.mse_loss(pi_hat, pi.detach()) + F.mse_loss(pt_hat, pt.detach()))
        patt = torch.randint(0, 3, (pi.size(0), 1), device=pi.device)   # 0 none / 1 image결손 / 2 text결손
        use_pi_hat = (patt == 1).float()                                # image 결손 → pi를 재구성으로 대체
        use_pt_hat = (patt == 2).float()                                # text 결손 → pt를 재구성으로 대체
        a = pi * (1 - use_pi_hat) + pi_hat * use_pi_hat
        b = pt * (1 - use_pt_hat) + pt_hat * use_pt_hat
        lc = info_nce(pi, pt, args.temp)
        return (F.binary_cross_entropy_with_logits(model.classify(a, b), y)
                + args.lam_rec * rec + args.w_align * lc + args.beta * align_cos(pi, pt))

    if m == "kd":
        # 학생: full 손실 + 결손 시뮬 입력을 교사 logit에 증류(교사 detach)
        loss = base_loss(model, pi, pt, y, args)
        with torch.no_grad():
            zi_t = teacher.encode_image(px)
            pit, ptt = teacher.proj(zi_t, txt)
            t_full = torch.sigmoid(teacher.classify(pit, ptt))         # 교사 full 확률
        z0i, z0t = torch.zeros_like(pi), torch.zeros_like(pt)
        s_img = torch.sigmoid(model.classify(z0i, pt))                 # 학생 image결손
        s_txt = torch.sigmoid(model.classify(pi, z0t))                 # 학생 text결손
        kd = bernoulli_kl(t_full.detach(), s_img) + bernoulli_kl(t_full.detach(), s_txt)
        return loss + args.kd_w * kd

    if m == "ib":
        logit_star, comp = model.ib_forward(pi, pt, sample=True)
        logit_full = model.classify(pi, pt)                            # 참조(충분) p(y|f)
        suff = bernoulli_kl(torch.sigmoid(logit_full).detach(), torch.sigmoid(logit_star))
        lc = info_nce(pi, pt, args.temp)
        return (F.binary_cross_entropy_with_logits(logit_star, y)
                + F.binary_cross_entropy_with_logits(logit_full, y)    # 참조 분류기 지도(빠지면 teacher 랜덤→붕괴)
                + args.ib_beta * comp + args.lam_suff * suff
                + args.gamma * (F.binary_cross_entropy_with_logits(model.head_i(pi), y)
                                + F.binary_cross_entropy_with_logits(model.head_t(pt), y))
                + args.w_align * lc + args.beta * align_cos(pi, pt))

    raise ValueError("unknown method " + m)


def run_training(args, method, tr, va, device, teacher=None, tag=""):
    model = V12Model(method, args.backbone, args.unfreeze_last, args.dim,
                     cra_blocks=args.cra_blocks, ib_bottleneck=args.ib_bottleneck,
                     siglip_name=args.siglip_name).to(device)
    opt = build_opt(model, args)
    nbb = sum(p.numel() for g in opt.param_groups[:1] for p in g["params"])
    print("=== v12 {} | method={} | backbone {} | dev={} ===".format(tag, method, args.backbone, device))
    best = {"val": -1.0, "state": None, "ep": args.epochs}
    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for px, txt, y in tr:
            px, txt, y = px.to(device), txt.to(device), y.to(device)
            loss = train_step(model, px, txt, y, args, teacher=teacher)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())
        print("  epoch {:>2d}  loss={:.4f}".format(ep + 1, tot / max(1, len(tr))))
        if args.early_stop and ((ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs):
            vs = eval_missing(model, va, device)["full"]["micro"]
            if vs > best["val"]:
                best.update(val=vs, ep=ep + 1,
                            state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            print("    [val] ep {} full_micro={:.3f} (best {:.3f}@{})".format(ep + 1, vs, best["val"], best["ep"]))
    if args.early_stop and best["state"] is not None:
        model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})
        print("  => 조기중단 선택: epoch {}".format(best["ep"]))
    return model


# ============================ 재현 체크 ============================
@torch.no_grad()
def recon_check(model, loader, device):
    """MMIN 재현 체크: CRA가 결손 모달을 그럴듯하게 재구성하는가 (cos 유사도)."""
    pi, pt, _ = extract_proj(model, loader, device)
    pi, pt = pi.to(device), pt.to(device)
    ci = F.cosine_similarity(model.cra_t2i(pt), pi, dim=1).mean().item()
    ct = F.cosine_similarity(model.cra_i2t(pi), pt, dim=1).mean().item()
    return {"recon_cos_image": ci, "recon_cos_text": ct}


@torch.no_grad()
def ib_check(model, loader, device):
    """IB 재현 체크: 마스크가 실제로 차원을 억제하는가 + 병목 압축량."""
    pi, pt, _ = extract_proj(model, loader, device)
    pi, pt = pi.to(device), pt.to(device)
    _, comp = model.ib_forward(pi, pt, sample=False)
    gate = torch.sigmoid(model.ib_gate)
    return {"mask_mean": gate.mean().item(), "mask_active_frac": (gate > 0.5).float().mean().item(),
            "compress_kl": float(comp)}


@torch.no_grad()
def usage_metrics(model, pi_te, pt_te, device):
    """사용 층위(H3) — fusion 분류기가 각 모달을 얼마나 쓰는가.
    가중치 norm(정적) + 활성 기여도 ||W_mod @ z||(동적). text결손 강건 = 이미지(약모달) 사용 비중↑.
    img_usage_share = 이미지 기여 / (이미지+텍스트 기여) — ModDrop이 올릴 것으로 예측(사용 재조정)."""
    W = model.cls.weight.detach()                       # (n_cls, 2*dim)
    d = model.dim
    Wi, Wt = W[:, :d], W[:, d:]
    pi, pt = pi_te.to(device), pt_te.to(device)
    img_contrib = (pi @ Wi.t()).norm(dim=1).mean().item()   # 결손 시 남은 이미지 기여 크기
    txt_contrib = (pt @ Wt.t()).norm(dim=1).mean().item()   # = text 자리 0으로 넣을 때 출력 변화량
    share = img_contrib / (img_contrib + txt_contrib + 1e-9)
    return {"img_w_norm": Wi.norm().item(), "txt_w_norm": Wt.norm().item(),
            "img_contrib": img_contrib, "txt_contrib": txt_contrib, "img_usage_share": share}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--feat_dir", default="feats")
    ap.add_argument("--raw_root", default=None)
    ap.add_argument("--backbone", choices=["clip", "siglip", "stub"], default="clip")
    ap.add_argument("--siglip_name", default=SIGLIP_NAME)
    ap.add_argument("--method", choices=["zerofill", "gamma", "moddrop", "mmin", "kd", "ib",
                                         "gamma_moddrop"], default="zerofill")
    ap.add_argument("--unfreeze_last", type=int, default=3)
    ap.add_argument("--dim", type=int, default=256)
    # 공통 정렬 스캐폴드
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--w_align", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=0.07)
    # 방법별
    ap.add_argument("--gamma", type=float, default=2.0, help="gamma/ib 브랜치 supervision 가중치")
    ap.add_argument("--moddrop_p", type=float, default=0.3, help="ModDrop 모달 zero 확률")
    ap.add_argument("--cra_blocks", type=int, default=3, help="MMIN CRA 잔차 블록 수")
    ap.add_argument("--lam_rec", type=float, default=1.0, help="MMIN 재구성 가중치")
    ap.add_argument("--kd_w", type=float, default=1.0, help="KD 증류 가중치")
    ap.add_argument("--ib_bottleneck", type=int, default=128, help="DMIB 병목 차원")
    ap.add_argument("--ib_beta", type=float, default=1e-3, help="DMIB 압축(KL) 가중치")
    ap.add_argument("--lam_suff", type=float, default=1.0, help="DMIB sufficiency KL 가중치")
    # 학습
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--early_stop", action="store_true")
    ap.add_argument("--eval_every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/v12.json")
    ap.add_argument("--save_preds", default=None, help="H2용 text결손 per-sample 이진예측+라벨 .pt 저장 경로")
    ap.add_argument("--limit", type=int, default=0, help="real 스모크: 각 split 앞 N개만")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 2
    if args.raw_root:                                  # ~ 확장(셸 따옴표 안에서 미확장 방지)
        args.raw_root = os.path.expanduser(args.raw_root)
    args.feat_dir = os.path.expanduser(args.feat_dir)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr = make_loader(args, "train", True)
    va = make_loader(args, "dev", False)
    te = make_loader(args, "test", False)

    teacher = None
    if args.method == "kd":                            # KD: full 교사 먼저 학습·freeze
        teacher = run_training(args, "zerofill", tr, va, device, tag="[teacher]")
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        t_ev = eval_missing(teacher, te, device)["full"]
        print("  [KD] 교사 full macro/micro = {:.3f}/{:.3f}".format(t_ev["macro"], t_ev["micro"]))

    model = run_training(args, args.method, tr, va, device, teacher=teacher, tag="[main]")

    ev = eval_missing(model, te, device)
    pi_tr, pt_tr, y_tr = extract_proj(model, tr, device)
    pi_te, pt_te, y_te = extract_proj(model, te, device)
    si = multilabel_probe(pi_tr, y_tr, pi_te, y_te, device=device, seed=args.seed)["f1_macro"]
    stx = multilabel_probe(pt_tr, y_tr, pt_te, y_te, device=device, seed=args.seed)["f1_macro"]

    checks = {}
    if args.method == "mmin":
        checks = recon_check(model, te, device)
    elif args.method == "ib":
        checks = ib_check(model, te, device)
    usage = usage_metrics(model, pi_te, pt_te, device)          # H3 사용 층위

    if args.save_preds:                                          # H2 per-sample text결손 예측
        lt = infer_logits(model, pi_te.to(device), pt_te.to(device), "text", device)
        torch.save({"pred_text": (torch.sigmoid(lt) > 0.5).int().cpu(), "y": y_te.int().cpu()}, args.save_preds)
        print(">>> preds saved", args.save_preds)

    print("\n--- 결손 성능 (F1 macro / micro / weighted) ---")
    for r in ("full", "image", "text", "both"):
        e = ev[r]
        print("  {:6s}: {:.3f} / {:.3f} / {:.3f}".format(r, e["macro"], e["micro"], e["weighted"]))
    print("--- suff --- z_image {:.3f}  z_text {:.3f}".format(si, stx))
    print("--- usage --- img_share {:.3f} (img_contrib {:.2f} / txt_contrib {:.2f})".format(
        usage["img_usage_share"], usage["img_contrib"], usage["txt_contrib"]))
    if checks:
        print("--- 재현 체크 ---", checks)

    res = {"method": args.method, "config": vars(args), "eval": ev,
           "suff": {"z_image": si, "z_text": stx}, "usage": usage, "checks": checks}
    json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    print(">>> saved", args.out)


if __name__ == "__main__":
    main()
