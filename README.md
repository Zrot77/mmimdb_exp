# mmimdb_exp — 1주차: MM-IMDb 환경 전환 (지시서 v6)

이미지+텍스트 결손 강건성. **사전학습 CLIP(freeze) + 정렬** 위에서 다중라벨(23 장르) 베이스라인 + sanity.
설계: backbone freeze니 **CLIP 특징을 한 번만 캐시**(extract_features) → 학습은 그 위 proj+정렬+BCE(빠름).

## 파일
- `utils.py` — info_nce/supcon(다중라벨)/align + F1(macro/micro) + 다중라벨 frozen probe(BCE) + 23 장르
- `extract_features.py` — raw MM-IMDb → frozen CLIP(get_image/text_features) → `feats/feats_{train,dev,test}.pt`
- `data_features.py` — 캐시/synthetic 로드 + 결손 η(평가: full/image/text, 학습: 확률 결손)
- `train_week1.py` — proj+정렬+BCE 학습 → full/결손 F1 + suff(frozen probe) + **S4 sanity 게이트**
- `slurm_week1.sbatch` — extract / train 스테이지

## 서버 셋업 (새 env)
```bash
conda create -n mmimdb python=3.10 -y && conda activate mmimdb
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-mmimdb.txt
python -c "import torch,transformers; print(torch.__version__, torch.cuda.is_available())"
```

## 데이터 (raw, archive.org 미러 8.1G)
```bash
mkdir -p ~/Yechan/mmimdb_data && cd ~/Yechan/mmimdb_data
wget -c https://archive.org/download/mmimdb/mmimdb.tar.gz
tar -xzf mmimdb.tar.gz         # → mmimdb/dataset/<id>.{json,jpeg}, split.json
ls mmimdb; ls mmimdb/dataset | head   # 구조 확인해 extract_features.py 파서와 맞추기
```

## 스모크 (데이터·CLIP 없이 로직 점검)
```bash
python train_week1.py --mode synthetic --smoke
```

## 실행 (서버, 데이터 준비 후)
```bash
# 1) CLIP 특징 캐시 — 스모크 먼저
python extract_features.py --raw_root ~/Yechan/mmimdb_data/mmimdb --out feats --limit 40
python extract_features.py --raw_root ~/Yechan/mmimdb_data/mmimdb --out feats      # 전체
# 2) 학습 + S4 sanity 게이트
python train_week1.py --mode real --feat_dir feats --epochs 30 --seed 1 --out results/week1_s1.json
```

## S4 sanity 게이트 (다음 주 넘어가기 전 필수)
- full F1-macro 붕괴 아님, 다중라벨 지표 정상, **결손 시 하락**(image/text 각각 full보다↓), frozen-probe 다중라벨 동작.
- 4개 다 PASS면 "STEP4 게이트 통과". 하나라도 FAIL이면 원인 수정.

## 기록해둘 것 (다음 주 "약한 모달 확정"에 영향)
- 텍스트 **77토큰 절단**(CLIP 한계) → 플롯 평균 92단어라 텍스트 모달이 인위적으로 약해질 수 있음. 절단 방식 명시.
- backbone freeze 기준. 파인튜닝은 성능 안 나오면 그때.

## ⚠️ 서버서 확정
`extract_features.py`의 raw 파서(`parse_plot`, `split.json`, `dataset/<id>.{json,jpeg}`)를 **실제 해제 구조로 확인**해 맞출 것. json 키(`plot`/`genres`)·split 파일명이 다르면 조정.
