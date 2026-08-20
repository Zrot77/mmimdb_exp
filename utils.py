"""공용 유틸 — 손실(info_nce/supcon/align) + 다중라벨 지표(F1) + 다중라벨 frozen probe.
MM-IMDb는 23-way 다중라벨이라 정확도가 아니라 F1(macro/micro)로 측정, probe도 다중라벨(BCE)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

# MM-IMDb 표준 23 장르 (multihot 순서 고정)
GENRES = ["Drama", "Comedy", "Romance", "Thriller", "Crime", "Action", "Adventure",
          "Horror", "Documentary", "Mystery", "Sci-Fi", "Fantasy", "Family", "Biography",
          "War", "History", "Music", "Animation", "Musical", "Western", "Sport",
          "Short", "Film-Noir"]
N_GENRE = len(GENRES)


# ============================ contrastive / 정렬 손실 ============================
def info_nce(z1, z2, temp):
    z1 = F.normalize(z1, p=2, dim=1); z2 = F.normalize(z2, p=2, dim=1)
    scores = torch.mm(z1, z2.t()) / temp
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(scores, labels) + F.cross_entropy(scores.t(), labels))


def supcon_multilabel(z1, z2, temp, Y):
    """다중라벨용 SupCon: 라벨 벡터가 하나라도 겹치면 positive로 본다(교집합>0)."""
    z1 = F.normalize(z1, p=2, dim=1); z2 = F.normalize(z2, p=2, dim=1)
    scores = torch.mm(z1, z2.t()) / temp
    pos = (Y.float() @ Y.float().t() > 0).float()
    logprob = F.log_softmax(scores, dim=1)
    loss = -(pos * logprob).sum(1) / pos.sum(1).clamp(min=1)
    return loss.mean()


def align_cos(z1, z2):
    z1 = F.normalize(z1, p=2, dim=1); z2 = F.normalize(z2, p=2, dim=1)
    return 1.0 - torch.mean(torch.diagonal(torch.mm(z1, z2.t())))


# ============================ 다중라벨 지표 ============================
def f1_macro_micro(logits, Y, thresh=0.5):
    """logits/Y: (N,23). 시그모이드>thresh 예측으로 F1 macro/micro."""
    pred = (torch.sigmoid(logits) > thresh).int().cpu().numpy()
    Yt = Y.int().cpu().numpy()
    return (f1_score(Yt, pred, average="macro", zero_division=0),
            f1_score(Yt, pred, average="micro", zero_division=0))


# ============================ 다중라벨 frozen probe (suff) ============================
@torch.no_grad()
def _to_t(x, device):
    return x if torch.is_tensor(x) else torch.tensor(x, device=device)


def multilabel_probe(ztr, Ytr, zte, Yte, epochs=300, lr=1e-2, device="cpu", seed=0):
    """frozen 임베딩 z → 라벨 선형 probe(BCE). suff(z) = test F1-macro/micro.
    sklearn OneVsRest의 '단일 클래스 라벨' 에러를 피하려 torch Linear+BCE로 학습."""
    torch.manual_seed(seed)
    ztr = _to_t(ztr, device).float(); Ytr = _to_t(Ytr, device).float()
    zte = _to_t(zte, device).float(); Yte = _to_t(Yte, device).float()
    clf = nn.Linear(ztr.size(1), Ytr.size(1)).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(clf(ztr), Ytr)
        loss.backward(); opt.step()
    with torch.no_grad():
        fm, fmi = f1_macro_micro(clf(zte), Yte)
    return {"f1_macro": float(fm), "f1_micro": float(fmi)}
