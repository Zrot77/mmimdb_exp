"""파인튜닝용 데이터 — 원본 포스터(학습 대상) + 고정 텍스트 특징(정렬 앵커, 캐시 재활용) + 라벨.
feats_{split}.pt의 ids로 원본 이미지를 찾고, txt는 그 캐시(고정)를 그대로 앵커로 쓴다."""
import os

import torch
from torch.utils.data import Dataset


class ImageTextFT(Dataset):
    def __init__(self, feat_dir, raw_root, split, processor, max_pixels=25_000_000):
        d = torch.load(os.path.join(feat_dir, "feats_{}.pt".format(split)), map_location="cpu")
        self.ids, self.txt, self.y = d["ids"], d["txt"].float(), d["y"].float()
        self.dir = os.path.join(raw_root, "dataset")
        self.proc = processor
        self.maxp = max_pixels
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(os.path.join(self.dir, self.ids[i] + ".jpeg")).convert("RGB")
        px = self.proc(images=img, return_tensors="pt")["pixel_values"][0]   # (3,224,224)
        return px, self.txt[i], self.y[i]


def _parse_plot(obj):
    p = obj.get("plot") or obj.get("plot outline") or obj.get("synopsis") or ""
    if isinstance(p, list):
        p = max(p, key=len) if p else ""
    return str(p).strip() or " "


class ImageTextTokens(Dataset):
    """블록1(대칭 FT)용 — 원본 포스터 + 원문 플롯 토큰(둘 다 학습 대상). label은 캐시 재활용."""
    def __init__(self, feat_dir, raw_root, split, processor, max_len=77):
        import os as _os
        d = torch.load(_os.path.join(feat_dir, "feats_{}.pt".format(split)), map_location="cpu")
        self.ids, self.y = d["ids"], d["y"].float()
        self.dir = _os.path.join(raw_root, "dataset")
        self.proc = processor
        self.max_len = max_len
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        import json as _json
        import os as _os
        from PIL import Image
        cid = self.ids[i]
        img = Image.open(_os.path.join(self.dir, cid + ".jpeg")).convert("RGB")
        plot = _parse_plot(_json.load(open(_os.path.join(self.dir, cid + ".json"), encoding="utf-8")))
        enc = self.proc(images=img, text=plot, return_tensors="pt",
                        padding="max_length", truncation=True, max_length=self.max_len)
        return enc["pixel_values"][0], enc["input_ids"][0], enc["attention_mask"][0], self.y[i]


class SyntheticSym(Dataset):
    """대칭 FT 스모크용 — px + 랜덤 토큰 + mask + label."""
    def __init__(self, n=256, n_cls=23, vocab=1000, seq=16, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.y = (torch.rand(n, n_cls, generator=g) < 0.15).float()
        Wp = torch.randn(n_cls, 3 * 32 * 32, generator=g)
        self.px = (self.y @ Wp + 0.6 * torch.randn(n, 3 * 32 * 32, generator=g)).reshape(n, 3, 32, 32)
        # 토큰: 라벨에 약하게 의존(각 라벨이 특정 토큰 존재 확률↑)
        self.ids = torch.randint(1, vocab, (n, seq), generator=g)
        self.mask = torch.ones(n, seq, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.px[i], self.ids[i], self.mask[i], self.y[i]


class SyntheticFT(Dataset):
    """스모크용. 라벨이 이미지 픽셀·텍스트에서 약하게 복원 가능하게."""
    def __init__(self, n=256, n_cls=23, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.y = (torch.rand(n, n_cls, generator=g) < 0.15).float()
        Wp = torch.randn(n_cls, 3 * 32 * 32, generator=g)
        Wt = torch.randn(n_cls, 512, generator=g)
        self.px = (self.y @ Wp + 0.6 * torch.randn(n, 3 * 32 * 32, generator=g)).reshape(n, 3, 32, 32)
        self.txt = self.y @ Wt + 0.5 * torch.randn(n, 512, generator=g)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.px[i], self.txt[i], self.y[i]
