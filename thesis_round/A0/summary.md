# A0 summary — 저장소 감사 + 수치 정본표 + 재현 점검

## 1. 실행 조건
- repo_map.md: 코드에서 스크립트·방법 구현·probe·백본 id 정리(동봉)
- audit_a0.py: 기존 results/*.json을 config 기준 묶어 ddof=1 재계산(학습 없음)
- 재현 점검: CLIP L0 γ=8 seed1 신규 1런 학습 → 기존 로그와 text결손 macro 비교

## 2. 결과
- **재현 점검(A0.2): 통과.** 새 run text결손 macro 0.4398 vs 기존 v13_gamma_g8.0_s1 0.4398, |Δ|=0.0000 (게이트 ≤0.010). 완전 일치 → 파이프라인 결정적
- **정본표(A0.3): run-json에 있는 수치 전부 일치(mismatch 없음).** L0 zero-fill(CLIP·SigLIP), CLIP γ 스윕, SigLIP γ16, 처방표(IB γ8=0.550·γ2=0.517), CLIP 이미지 suff 4종, SigLIP 계열 6행 γ_text·IB_text, B1표 — canonical_numbers.csv 참조
- **부수 재현 확인**: v19 재학습 SigLIP γ8(0.582)이 v14b γ8(0.582)과 동일

## 3. 통계
- seed 목록(A0.4j): [1, 2, 3] 확정
- ddof(A0.4i): 기존 집계 혼재 — 초기 agg(v12~v16) ddof=0, 후기(agg_v17/18) ddof=1. 이번 재계산은 §3.4 폴백대로 ddof=1 사용. 이후 ddof=1 통일 권고
- paired 통계는 A1부터 적용(df=2, t0.975=4.303)

## 4. 사전 판정
- A0.2 재현 점검: **통과** → A1 진행 가능

## 5. 이상 징후 (사실만)
- config.backbone=None 1건, stub(합성 스모크) 다수 → 실험 run 아님, 집계서 무시
- `clip/-` 와 `clip/siglip-so400m` 중복행(ib/kd/mmin) = siglip_name 인자 도입 전/후 run이 같은 방법으로 중복 집계, 값 동일 → 정본엔 1개로 dedup
- CLIP γ16 재계산값 0.468(문서 0.469는 γ16~32 포화 대표값, 불일치 아님)

## 6. 사용자 확인 필요
- 정본표 미완 항목(별도 소스): raw suff·표현강도(= ceiling_probe 재실행), frozen 3차(= v9/v11 로그). 정본표 완성용으로 recompute 할지
- ddof=1 통일 승인 여부
