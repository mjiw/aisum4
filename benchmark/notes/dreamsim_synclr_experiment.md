# DreamSim SynCLR ViT-B/16 평가 기록

담당: yeooonsu / 브랜치: `test/dreamsim-synclr` (`test/dino-v2`의 `8314c0f`에서 분기)
평가 실행·속도 측정·문서 작성: 2026-10-05

## 1. 모델

| 항목 | 값 |
|---|---|
| 모델 | DreamSim single-branch (SynCLR ViT-B/16 백본) |
| `_REGISTRY` 이름 | `dreamsim_synclr_vitb16` |
| `dreamsim_type` | `synclr_vitb16` |
| 임베딩 차원 | **768** (ensemble 1792의 절반 이하, DINOv2 variant와 같음) |
| 가중치 | dreamsim 패키지가 GitHub 릴리스에서 자동 다운로드 (`synclr_vit_b_16.pth` + `synclr_vitb16_single_lora`) |
| 출처 | Fu et al., *DreamSim: Learning New Dimensions of Human Visual Similarity from Synthetic Data*, NeurIPS 2023 ([github](https://github.com/ssundaram21/dreamsim)) |
| 백본 출처 | Tian et al., *Learning Vision from Models Rivals Learning Vision from Data* (SynCLR), CVPR 2024 |

SynCLR은 **실사 이미지 없이** 생성 모델이 만든 합성 이미지와 캡션만으로 학습한
백본입니다. DreamSim은 이 백본 위에 사람의 유사도 판단 데이터로 LoRA를 얹었습니다.

같은 DreamSim 미세조정을 거친 `dreamsim_dinov2_vitb14`와 비교하면,
**백본을 실사 데이터로 학습했는지(DINOv2) 합성 데이터로 학습했는지(SynCLR)**가
상품 검색 성능에 어떤 차이를 만드는지 볼 수 있습니다. 패치 크기도 다릅니다
(B/16 vs B/14). 224 입력에서 토큰이 196개 vs 256개라 SynCLR 쪽 연산이 적습니다.

래퍼는 기존 [`embedding/models/dreamsim.py`](../../embedding/models/dreamsim.py)를 그대로
재사용하고 `dreamsim_type`만 바꿨습니다. 새 모델 파일은 만들지 않았습니다.

### 등록할 때 건드린 파일

| 파일 | 내용 |
|---|---|
| `benchmark/config.json` | `models.dreamsim_synclr_vitb16` 추가 (`dreamsim_type`, `embed_dim: 768`, `batch_size: 16`, `cache_dir`) |
| `benchmark/models/__init__.py` | `_REGISTRY`에 `("dreamsim", "DreamSim")` 한 줄 |
| `benchmark/README.md` | 담당 표에 한 줄 |

`embed_dim`은 config에 **반드시 명시**해야 합니다. 래퍼의 `_KNOWN_DIMS`에는 `ensemble`만
있어서, 없으면 "출력 차원을 모릅니다" 예외가 납니다. 나머지 설정은
`dreamsim_dinov2_vitb14`와 같습니다.

## 2. 실행 환경

| 항목 | 값 |
|---|---|
| Python | 3.11.7 (`.venv`) |
| torch | 2.14.0+cu130 (CUDA 13.0) |
| dreamsim | 0.2.1 |
| transformers / datasets / numpy | 5.17.0 / 5.0.1 / 2.4.6 |
| GPU | NVIDIA GeForce RTX 4090 (24GB), VRAM 상한 35% |
| OS | Windows 11 |

`dreamsim_dinov2_vitb14` 평가와 같은 `.venv`입니다.

## 3. 환경 이슈와 해결법

이번에는 새로 생긴 이슈가 없습니다. dreamsim은 DINOv2 평가 때 이미 설치돼 있었고,
SynCLR 가중치는 첫 실행 때 자동으로 받아졌습니다.

처음 설치하는 환경이라면 DINOv2 문서에 적은 두 가지가 그대로 적용됩니다
([dreamsim_dinov2_experiment.md](dreamsim_dinov2_experiment.md#3-환경-이슈와-해결법)).

- Windows Smart App Control이 실행을 막으면 설정에서 끄고 재부팅
- dreamsim 설치 시 `$env:PYTHONUTF8 = "1"` (cp949 `UnicodeDecodeError` 회피)

## 4. 실행한 명령어

```bash
# 등록 확인 (데이터셋 없이 래퍼 동작만)
python -m benchmark.probe_model --model dreamsim_synclr_vitb16

# 스모크 테스트 (결과 JSON은 커밋하지 않고 삭제)
python -m benchmark.run_eval --model dreamsim_synclr_vitb16 --config real_studio_flat --limit 200
python -m benchmark.run_eval --model dreamsim_synclr_vitb16 --dataset sop --limit 200

# LookBench 4개 서브셋
python -m benchmark.run_eval --model dreamsim_synclr_vitb16 --config real_studio_flat
python -m benchmark.run_eval --model dreamsim_synclr_vitb16 --config real_streetlook
python -m benchmark.run_eval --model dreamsim_synclr_vitb16 --config aigen_studio
python -m benchmark.run_eval --model dreamsim_synclr_vitb16 --config aigen_streetlook

# SOP
python -m benchmark.run_eval --model dreamsim_synclr_vitb16 --dataset sop

# 속도 (dreamsim ensemble 대비, DINOv2 때와 같은 설정)
python -m benchmark.speed --models dreamsim dreamsim_synclr_vitb16 \
    --batch-size 16 --n-images 3200 --repeats 3 --warmup 10 --seed 0

# 결과 설정 일치 검사
python -m benchmark.check_settings --model dreamsim_synclr_vitb16
python -m benchmark.check_settings
```

평가 설정: `batch_size 16`, `device cuda`, `gpu_memory_fraction 0.35`, `fine_mode exact`,
`limit null`, LookBench noise 풀 포함, SOP leave-one-out(`exclude_self true`),
LookBench revision `151449aa`(v20251201), SOP revision `24a1b9b8`.

검증 결과:

- `probe_model`: 768차원, L2 norm 1.000000, batch/단건 차이 1.14e-04 → 통과
- `check_settings --model dreamsim_synclr_vitb16`: 5건 설정 일치, 경고 0건 → 통과
- `check_settings` 전체(25건): 서브셋 5개 모두 5개 모델의 설정이 같음, 오류 0건 → 통과.
  경고 16건은 이 모델과 무관한 기존 결과의 것입니다(7절 참고)

## 5. 결과

### LookBench 서브셋별

| 서브셋 | 쿼리 | coarse R@1 | coarse R@10 | coarse mAP@10 | fine R@1 | fine R@10 | fine mAP@10 |
|---|---|---|---|---|---|---|---|
| real_studio_flat | 1,011 | 0.7488 | 0.8714 | 0.4459 | 0.3274 | 0.5381 | 0.2341 |
| real_streetlook | 981 | 0.4332 | 0.6820 | 0.1767 | 0.1876 | 0.3384 | 0.1331 |
| aigen_studio | 193 | 0.5285 | 0.7306 | 0.2622 | 0.2228 | 0.4093 | 0.1338 |
| aigen_streetlook | 160 | 0.4500 | 0.6500 | 0.1947 | 0.2125 | 0.3375 | 0.1256 |
| **전체 (쿼리 수 가중)** | **2,345** | **0.5783** | **0.7655** | **0.3010** | **0.2525** | **0.4303** | **0.1762** |

### SOP (test, leave-one-out)

| 지표 | 값 |
|---|---|
| exact recall@1 | 0.6484 |
| exact recall@10 | 0.7893 |
| exact mAP@10 | 0.3596 |

### 속도 (batch 16, RTX 4090, fp32, ensemble = 1)

| 데이터셋 | forward 소요시간 | preprocess 포함 소요시간 |
|---|---|---|
| LookBench | **0.33배** | 0.50배 |
| SOP | **0.34배** | 0.51배 |

장당 시간은 forward 1.37ms(ensemble 4.09ms), preprocess 약 1.43ms입니다.

## 6. dreamsim_dinov2_vitb14와 비교

같은 DreamSim 미세조정에 백본만 다른 두 variant입니다. 차원(768)과 평가 설정이 같습니다.

### 정확도: LookBench 전체(쿼리 수 가중)와 SOP

| 지표 | dreamsim_dinov2_vitb14 | **dreamsim_synclr_vitb16** | 차이 (SynCLR − DINOv2) |
|---|---|---|---|
| LookBench coarse R@1 | **0.6964** | 0.5783 | −11.8pp |
| LookBench coarse R@10 | **0.8631** | 0.7655 | −9.8pp |
| LookBench coarse mAP@10 | **0.3871** | 0.3010 | −8.6pp |
| LookBench fine R@1 | **0.2964** | 0.2525 | −4.4pp |
| LookBench fine R@10 | **0.4840** | 0.4303 | −5.4pp |
| LookBench fine mAP@10 | **0.2110** | 0.1762 | −3.5pp |
| SOP exact R@1 | 0.6189 | **0.6484** | +3.0pp |
| SOP exact R@10 | 0.7694 | **0.7893** | +2.0pp |
| SOP exact mAP@10 | 0.3399 | **0.3596** | +2.0pp |

### LookBench 서브셋별 coarse R@1

| 서브셋 | dreamsim_dinov2_vitb14 | dreamsim_synclr_vitb16 | 차이 |
|---|---|---|---|
| real_studio_flat | **0.8220** | 0.7488 | −7.3pp |
| real_streetlook | **0.5973** | 0.4332 | −16.4pp |
| aigen_studio | **0.6114** | 0.5285 | −8.3pp |
| aigen_streetlook | **0.6125** | 0.4500 | −16.3pp |

### 속도 (ensemble = 1, 소요시간 배율)

| 데이터셋 | 기준 | dreamsim_dinov2_vitb14 | **dreamsim_synclr_vitb16** |
|---|---|---|---|
| LookBench | forward | 0.42 | **0.33** |
| LookBench | preprocess 포함 | 0.55 | **0.50** |
| SOP | forward | 0.42 | **0.34** |
| SOP | preprocess 포함 | 0.58 | **0.51** |

forward 장당 시간은 SynCLR 1.37ms, DINOv2 1.71ms로 **SynCLR이 약 20% 빠릅니다**.
토큰 수 비율(196/256 = 0.77)과 거의 같아서, 차이는 패치 크기에서 온 것으로 보입니다.

두 측정은 다른 날(DINOv2 2026-09-21, SynCLR 2026-10-05) 같은 장비에서 했습니다.
ensemble 기준 forward 시간이 4.093ms와 4.094ms로 거의 같아서, **forward 배율은 나란히
비교해도 됩니다.** 반면 preprocess는 CPU 상태에 따라 세션 간 차이가 있었습니다
(ensemble 기준 1.43~1.61ms). 그래서 preprocess 포함 배율은 0.05 안팎의 차이를
오차 범위로 보는 게 맞습니다.

### 읽은 내용

- **패션 검색(LookBench)에서는 DINOv2 백본이 확실히 낫습니다.** coarse R@1이 11.8pp
  차이 나고, 서브셋별 24개 지표 중 23개에서 DINOv2가 앞섭니다. 유일한 예외는
  aigen_studio fine R@10(SynCLR 0.4093 vs 0.3990, 쿼리 193개)입니다.
- **streetlook에서 격차가 특히 큽니다**(−16pp). 배경이 복잡한 착용샷에서 크롭된 상품을
  찾는 과제인데, 이 상황에서 SynCLR 백본이 약합니다. 단색 배경의 studio에서는
  격차가 절반 수준(−7~8pp)입니다.
- **일반 상품(SOP)에서는 SynCLR이 낫습니다**(R@1 +3.0pp, mAP +2.0pp). 도메인에 따라
  순위가 뒤집히는 것은 DINOv2 문서에서 dinov3_vitb16과 비교했을 때와 같은 양상입니다.
  SOP는 가구·생활용품이라 이 프로젝트의 목표(패션 검색)와는 거리가 있습니다.
- **속도는 SynCLR이 20% 빠르지만 정확도 손실이 더 큽니다.** 이 프로젝트 기준으로는
  `dreamsim_dinov2_vitb14`가 더 나은 선택입니다. 둘 다 ensemble보다 정확도는 낮습니다
  (ensemble LookBench coarse R@1 0.7569).

## 7. 알려진 이슈

1. **ensemble·dinov2_vitb14·dinov3_vitb16 결과에 mAP@10이 없습니다.** mAP는
   `478992b`에서 추가됐고 그 전에 돌린 결과들입니다. 지시에 따라 이번에 재실행하지
   않았습니다. 그래서 **mAP 비교는 `dreamsim_dinov2_vitb14`와 이 모델 둘 사이에서만**
   가능합니다.
2. **ensemble 결과의 `device`가 섞여 있습니다**(cuda 2건, cpu 2건, SOP cuda). 지표 비교에는
   영향이 없습니다.
3. README의 기준 환경(python 3.13.11)과 실제 실행 환경(3.11.7)이 다릅니다.
4. `requirements.txt`에 `dreamsim` 버전이 고정돼 있지 않습니다. 이 평가는 0.2.1에서
   했고, 0.2.1의 `config.py`에 `synclr_vitb16` 항목이 있는 것을 확인했습니다. 다른
   버전에서는 이 `dreamsim_type`이 없거나 가중치가 다를 수 있습니다.

1·2번은 `python -m benchmark.check_settings`로 다시 확인할 수 있습니다.

## 8. 관련 커밋

| 해시 | 내용 |
|---|---|
| `0b0c18e` | DreamSim SynCLR ViT-B/16 평가 대상 등록 (config, 레지스트리, README) |
| (이 문서와 함께) | 평가 결과 JSON 5개, 속도 결과 2개, 이 문서 |

비교 대상 `dreamsim_dinov2_vitb14` 커밋은 `4258d00`(등록), `b396dad`·`478992b`(결과)입니다.

결과 파일: `benchmark/results/dreamsim_synclr_vitb16__lookbench-{real_studio_flat,real_streetlook,aigen_studio,aigen_streetlook}.json`,
`benchmark/results/dreamsim_synclr_vitb16__sop-test.json`,
`benchmark/results/speed__dreamsim+dreamsim_synclr_vitb16__{lookbench,sop}__bs16.json`
