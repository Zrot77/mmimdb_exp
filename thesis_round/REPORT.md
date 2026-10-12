# 학위논문 추가 실험 — 전체 보고서 (thesis_round)

| 항목 | 상태 | 사전 판정 | 비고 |
|---|---|---|---|
| A0 저장소 감사·정본표·재현점검 | 완료 | 재현점검 통과(|Δ|=0.000) | 정본 run-json 전부 일치, seed[1,2,3], ddof 혼재→1 통일 권고. 미완: raw suff·frozen3차 별도소스 |
| A1 CLIP γ8+ModDrop factorial | 미실행 | - | A0 통과로 착수 가능 |
| A2 CLIP γ-suff gap (권장) | 미실행 | - | ★CLIP γ 체크포인트 없음 → γ16 재학습 or 스킵 결정 필요 |
| A3 feature-noise L0 (권장) | 미실행 | - | L0에 미구현 → 이식 필요. SigLIP쪽 무거움 |
| B0 PAD-UFES-20 준비 | 미실행 | - | 다운로드 방식(서버 wget vs 사용자 제공) 확인 필요 |
| B1·B2 PAD 모델·게이트 | 미실행 | - | B0 후 |
| B3·B4 PAD 본실험·천장 | 미실행 | - | B2 게이트 통과 시 |
| C ESBL | 미실행 | - | 사용자 명시 착수 지시 전 금지 |

## 링크
- A0: thesis_round/A0/summary.md, repo_map.md, canonical_numbers.csv, recompute.csv

## 미해결 결정 (사용자)
1. ddof=1 통일 승인
2. A2: CLIP γ16 재학습 vs 스킵
3. A3: feature-noise L0 이식 진행 여부(SigLIP 무거움)
4. B0: PAD-UFES-20 다운로드 방식
5. 정본표 미완 항목(raw suff·frozen 3차) recompute 여부
