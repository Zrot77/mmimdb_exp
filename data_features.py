"""캐시된 CLIP 특징 로드(real) 또는 synthetic 생성 + 결손 η 유틸.
feats_{split}.pt = {"img":(N,512),"txt":(N,512),"y":(N,23),"ids":[...]}"""
import os

import torch

from utils import N_GENRE


def load_feats(feat_dir, split):
    d = torch.load(os.path.join(feat_dir, "feats_{}.pt".format(split)), map_location="cpu")
    return d["img"].float(), d["txt"].float(), d["y"].float()


def synth_feats(n, seed=0, dim=512, text_noise=0.8):
    """스모크용. 두 모달이 '상보적'(앞 절반 장르는 이미지가, 뒤 절반은 텍스트가 강하게)이라
    한쪽 결손 시 그 절반 정보가 빠져 성능이 하락 → missing_degrades 체크까지 검증됨.
    텍스트를 더 잡음 많게(약한 모달 흉내)."""
    g = torch.Generator().manual_seed(seed)
    Y = (torch.rand(n, N_GENRE, generator=g) < 0.15).float()
    Wi = torch.randn(N_GENRE, dim, generator=g)
    Wt = torch.randn(N_GENRE, dim, generator=g)
    mask_i = torch.zeros(N_GENRE); mask_i[: N_GENRE // 2] = 1.0     # 이미지 담당 장르
    mask_t = 1.0 - mask_i                                          # 텍스트 담당 장르
    img = (Y * mask_i) @ Wi + 0.3 * torch.randn(n, dim, generator=g)
    txt = (Y * mask_t) @ Wt + text_noise * torch.randn(n, dim, generator=g)
    return img.float(), txt.float(), Y.float()


def get_split(mode, feat_dir, split, seed=0):
    if mode == "synthetic":
        n = {"train": 1500, "dev": 400, "test": 600}[split]
        return synth_feats(n, seed={"train": 0, "dev": 1, "test": 2}[split])
    return load_feats(feat_dir, split)


def apply_missing(zi, zt, mode):
    """평가용 결손: 'full' | 'image'(이미지 결손→zero) | 'text'(텍스트 결손→zero)."""
    if mode == "image":
        zi = torch.zeros_like(zi)
    elif mode == "text":
        zt = torch.zeros_like(zt)
    return zi, zt


def train_missing_mask(zi, zt, eta, gen=None):
    """학습 시 결손 시뮬레이션: 확률 eta로 각 샘플에서 한 모달을 zero(둘 다 남기지 않도록 한쪽만)."""
    if eta <= 0:
        return zi, zt
    B = zi.size(0)
    r = torch.rand(B, device=zi.device, generator=gen)
    drop = r < eta
    drop_img = drop & (torch.rand(B, device=zi.device, generator=gen) < 0.5)
    drop_txt = drop & ~drop_img
    zi = zi.clone(); zt = zt.clone()
    zi[drop_img] = 0.0
    zt[drop_txt] = 0.0
    return zi, zt
