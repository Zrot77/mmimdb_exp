# A0.1 저장소 지도 (repo_map)

> 저장소 루트(서버): ~/Yechan/mmimdb_exp (= 로컬 D:/MMN/multimodal_ib/MMIB/mmimdb_exp, git: github.com/Zrot77/mmimdb_exp)
> 아래는 코드에서 확인되는 사실. 미확인은 "확인 불가"로 표기

## 학습·평가·결과·config
- 메인 학습·평가 스크립트: `train_v12.py` (L0 파이프라인, 결손 평가 내장)
- 결과 로그 디렉터리: `results/` (run별 json). 정본표·thesis 산출물은 `results/thesis_round/<EXP_ID>/`
- config 디렉터리: 없음 — 모든 설정은 `train_v12.py`의 argparse 인자 + 각 run json의 `config` 필드에 기록됨
- 집계: `agg_v12.py`~`agg_v19.py`, `agg_v14b/14c.py`, `ceiling_probe.py`

## 각 방법 구현 위치 (전부 train_v12.py, 함수 `train_step` 내 `--method`로 분기)
- γ (sufficiency 앵커링): method `gamma` — `loss += args.gamma*(BCE(head_i(pi),y)+BCE(head_t(pt),y))`
- ModDrop: method `moddrop` — 분류 입력 pi/pt를 확률 `args.moddrop_p`로 zero
- γ+ModDrop: method `gamma_moddrop`
- KD: method `kd` — full 교사(2단 학습) → 결손 학생 bernoulli_kl 증류
- DMIB(IB): method `ib` — `V12Model.ib_forward`(ib_gate 마스크+ib_mu/ib_logvar 병목+cls_ib), 손실 BCE(logit_star)+BCE(logit_full)+ib_beta*comp+lam_suff*suff+gamma*branch
- MMIN: method `mmin` — `CRA` 클래스(cascade residual AE) cra_t2i/cra_i2t, 샘플별 결손패턴 + lam_rec 재구성
- B1(텍스트 essence 이전): method `b1`/`gamma_b1` — `pi_b1`(이미지→텍스트 특징공간), `lam_b*(1-cos(pi_b1(zi), txt.detach()))`
- TLP(추론 시점): 별도 `tlp_v19.py` — 3C logit prompt, L_u(MSE-to-0.5)+(1/C)L_c(nearest-complete Bernoulli-KL), `--lam_u`
- feature-noise(frozen판): `train_v9.py` method `noise`/`train_v11.py` presc `noise` — 정렬 직전 (pi,pt)에 `presc_sigma*randn`. ★L0(train_v12)엔 미구현(A3에서 이식 필요)

## probe
- suff probe: `utils.py` `multilabel_probe` (torch Linear+BCE, epochs 300 lr 1e-2, thresh 0.5, 정규화 없음), suff = test f1_macro. train_v12에서 si/stx로 호출
- SigLIP 계열 "표현강도" probe: `ceiling_probe.py`/`agg_v16.py` `frozen_strength` — 동일한 `multilabel_probe`를 캐시 이미지 특징(feats_*/img)에 적용. ★raw suff와 동일 구현(입력만 다름)
- IB 인과 probe: train_v12 `ib_causal`(suff_f/suff_fstar/var/effrank), MMIN `mmin_causal`
- 사용층위: train_v12 `usage_metrics`(img_usage_share)

## 백본 로딩 + 모델 id (V12Model.__init__)
- CLIP: `transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")`, L0=마지막 3블록+visual_projection+post_layernorm 학습, img_dim 512
- SigLIP: `transformers.AutoModel.from_pretrained(args.siglip_name)`, L0=마지막 3블록+head+post_layernorm, img_dim=vision_config.hidden_size. 기본 so400m="google/siglip-so400m-patch14-384"(1152)
- 계열 id: base-patch16-{224,256,384,512}(768), large-patch16-256(1024), siglip2-base-patch16-224(768). 이미지 프로세서는 `AutoImageProcessor`(텍스트 토크나이저 우회)
- 텍스트: CLIP 텍스트 앵커(frozen 캐시 feats_*/txt, 512). SigLIP 세팅도 텍스트는 CLIP 앵커 유지
- 특징 추출: `extract_features.py`(CLIP), `extract_siglip.py`(SigLIP 계열), `extract_b4.py`(DINOv2+BERT)

## 공통 L0 설정 (train_v12 기본)
- epochs 8, early_stop=val full-micro F1, 차등 lr(백본 1e-5 / head 1e-3), wd 1e-4, batch 64(CLIP)/16(SigLIP), proj dim 256, 23-genre 다중라벨 BCE, zero-fill 결손
- 손실 스캐폴드(전 method 공통): BCE(fusion) + w_align*InfoNCE + beta*align_cos (w_align 1.0, beta 0.3, temp 0.07)
- seed: 1,2,3 (A0.4j에서 로그 재확인)
