# DreamSim DINOv2 ViT-B/14 평가 기록

담당: yeooonsu / 브랜치: `test/dino-v2`
평가 실행: 2026-09-14 / 속도 측정: 2026-09-21 / 문서 작성: 2026-10-05

## 1. 모델

| 항목 | 값 |
|---|---|
| 모델 | DreamSim single-branch (DINOv2 ViT-B/14 백본) |
| `_REGISTRY` 이름 | `dreamsim_dinov2_vitb14` |
| `dreamsim_type` | `dinov2_vitb14` |
| 임베딩 차원 | **768** (ensemble 1792의 절반 이하) |
| 가중치 | dreamsim 패키지가 GitHub 릴리스에서 자동 다운로드 (`dinov2_vitb14_pretrain.pth` + `dinov2_vitb14_single_lora`) |
| 출처 | Fu et al., *DreamSim: Learning New Dimensions of Human Visual Similarity from Synthetic Data*, NeurIPS 2023 ([github](https://github.com/ssundaram21/dreamsim)) |

DreamSim은 ImageNet 사전학습 백본에 사람의 유사도 판단 데이터로 LoRA 미세조정을 얹은
perceptual metric입니다. 그래서 **raw DINOv2(`facebook/dinov2-base`, 레지스트리의
`dinov2_vitb14`)와는 가중치가 다른 별개 모델**입니다. 두 모델을 비교하면 perceptual
tuning이 상품 검색에 도움이 되는지를 볼 수 있어서 대조군으로 함께 돌렸습니다.

래퍼는 기존 [`embedding/models/dreamsim.py`](../../embedding/models/dreamsim.py)를 그대로
재사용하고 `dreamsim_type`만 바꿨습니다. 새 모델 파일은 만들지 않았습니다.

### 등록할 때 건드린 파일

| 파일 | 내용 |
|---|---|
| `benchmark/config.json` | `models.dreamsim_dinov2_vitb14` 추가 (`dreamsim_type`, `embed_dim: 768`, `batch_size: 16`, `cache_dir`) |
| `benchmark/models/__init__.py` | `_REGISTRY`에 `("dreamsim", "DreamSim")` 한 줄 |
| `benchmark/README.md` | 담당 표에 한 줄 |

`embed_dim`은 config에 **반드시 명시**해야 합니다. 래퍼의 `_KNOWN_DIMS`에는 `ensemble`만
있어서, 없으면 "출력 차원을 모릅니다" 예외가 납니다.

## 2. 실행 환경

| 항목 | 값 |
|---|---|
| Python | 3.11.7 (`.venv`) |
| torch | 2.14.0+cu130 (CUDA 13.0) |
| dreamsim | 0.2.1 |
| transformers / datasets / numpy | 5.17.0 / 5.0.1 / 2.4.6 |
| GPU | NVIDIA GeForce RTX 4090 (24GB), VRAM 상한 35% |
| OS | Windows 11 |

README의 "기준 실행 환경"에는 python 3.13.11로 적혀 있지만, 이 평가는 **3.11.7**에서
돌렸습니다. 둘 중 하나로 문서를 맞춰야 합니다(미해결).

## 3. 환경 이슈와 해결법

### Smart App Control 차단

Windows의 Smart App Control이 서명 없는 실행 파일을 막아 설치·실행이 중단됐습니다.
설정에서 끄고 재부팅한 뒤 진행했습니다. 한 번 끄면 되돌릴 수 없으니 팀원은 각자
판단해서 끄면 됩니다.

### dreamsim 설치 시 `PYTHONUTF8=1` 필요

dreamsim 설치 과정에서 UTF-8로 된 파일을 Windows 기본 코덱(cp949)으로 읽다가
`UnicodeDecodeError`가 납니다. UTF-8 모드를 켜고 설치하면 통과합니다.

```powershell
$env:PYTHONUTF8 = "1"
pip install dreamsim
```

평가 실행 자체에는 필요 없고 설치할 때만 필요합니다. 다만 콘솔 출력이 깨지는 걸 막으려면
실행할 때도 `PYTHONIOENCODING=utf-8`을 주는 편이 편합니다.

> 위 두 항목은 이전 세션에서 겪은 내용을 기록으로 남긴 것입니다. 이 문서를 쓰면서
> 재현해 확인한 것은 아닙니다.

## 4. 실행한 명령어

```bash
# 등록 확인 (데이터셋 없이 래퍼 동작만)
python -m benchmark.probe_model --model dreamsim_dinov2_vitb14

# LookBench 4개 서브셋
python -m benchmark.run_eval --model dreamsim_dinov2_vitb14 --config real_studio_flat
python -m benchmark.run_eval --model dreamsim_dinov2_vitb14 --config real_streetlook
python -m benchmark.run_eval --model dreamsim_dinov2_vitb14 --config aigen_studio
python -m benchmark.run_eval --model dreamsim_dinov2_vitb14 --config aigen_streetlook

# SOP
python -m benchmark.run_eval --model dreamsim_dinov2_vitb14 --dataset sop

# 결과 설정 일치 검사 → 비교표
python -m benchmark.check_settings --model dreamsim_dinov2_vitb14
python -m benchmark.compare

# 속도 (dreamsim ensemble 대비)
python -m benchmark.speed --models dreamsim dreamsim_dinov2_vitb14 --batch-size 16
```

평가 설정: `batch_size 16`, `device cuda`, `gpu_memory_fraction 0.35`, `fine_mode exact`,
`limit null`, LookBench noise 풀 포함, SOP leave-one-out(`exclude_self true`),
LookBench revision `151449aa`(v20251201), SOP revision `24a1b9b8`.

## 5. 결과

### LookBench 서브셋별

| 서브셋 | 쿼리 | coarse R@1 | coarse R@10 | coarse mAP@10 | fine R@1 | fine R@10 | fine mAP@10 |
|---|---|---|---|---|---|---|---|
| real_studio_flat | 1,011 | 0.8220 | 0.9327 | 0.5273 | 0.3798 | 0.5955 | 0.2725 |
| real_streetlook | 981 | 0.5973 | 0.8002 | 0.2749 | 0.2304 | 0.4037 | 0.1643 |
| aigen_studio | 193 | 0.6114 | 0.8653 | 0.3188 | 0.2435 | 0.3990 | 0.1581 |
| aigen_streetlook | 160 | 0.6125 | 0.8063 | 0.2721 | 0.2375 | 0.3750 | 0.1730 |
| **전체 (쿼리 수 가중)** | **2,345** | **0.6964** | **0.8631** | **0.3871** | **0.2964** | **0.4840** | **0.2110** |

studio가 streetlook보다 크게 높습니다. 배경이 단색인 제품컷끼리 매칭하는 쪽이
거리 착용샷보다 쉬운 구도라 예상에 맞는 방향입니다.

### SOP (test, leave-one-out)

| 지표 | 값 |
|---|---|
| exact recall@1 | 0.6189 |
| exact recall@10 | 0.7694 |
| exact mAP@10 | 0.3399 |

### 속도 (batch 16, RTX 4090, fp32, ensemble = 1)

| 데이터셋 | forward 소요시간 | preprocess 포함 소요시간 |
|---|---|---|
| LookBench | **0.42배** | 0.55배 |
| SOP | **0.42배** | 0.58배 |

장당 시간은 forward 1.71ms(ensemble 4.09ms), preprocess 약 1.5ms로 두 모델이 비슷합니다.
preprocess는 같은 전처리를 쓰므로 차이가 없고, 그만큼 전체 배율이 희석됩니다.
forward 배율이 두 데이터셋에서 같아서, 입력이 224로 고정되면 데이터셋은 영향이 없습니다.
측정 방법은 [`benchmark/speed.py`](../speed.py) 주석에 있습니다.

## 6. 다른 모델과의 비교

LookBench 전체(쿼리 수 가중 평균)와 SOP입니다. mAP는 지표 추가 이후에 돌린
`dreamsim_dinov2_vitb14`에만 있습니다.

| 모델 | 차원 | coarse R@1 | coarse R@10 | fine R@1 | fine R@10 | SOP R@1 | SOP R@10 |
|---|---|---|---|---|---|---|---|
| dreamsim (ensemble) | 1792 | **0.7569** | **0.9045** | **0.3207** | **0.5424** | **0.7014** | **0.8441** |
| **dreamsim_dinov2_vitb14** | 768 | 0.6964 | 0.8631 | 0.2964 | 0.4840 | 0.6189 | 0.7694 |
| dinov2_vitb14 (raw) | 768 | 0.6401 | 0.8273 | 0.2418 | 0.4328 | 0.5500 | 0.7109 |
| dinov3_vitb16 | 768 | 0.6857 | 0.8708 | 0.2657 | 0.4661 | 0.6532 | 0.8073 |

읽은 내용:

- **ensemble이 모든 지표에서 가장 좋습니다.** single-branch는 coarse R@1에서 6.1pp,
  fine R@1에서 2.4pp, SOP R@1에서 8.3pp 낮습니다. 대신 차원이 절반 이하(768 vs 1792)고
  forward가 2.4배 빠릅니다. 벡터 저장 용량과 검색 비용도 차원에 비례해 줄어듭니다.
- **perceptual tuning은 효과가 있습니다.** 같은 DINOv2 B/14 백본인데 raw 대비
  coarse R@1 +5.6pp, fine R@1 +5.5pp, SOP R@1 +6.9pp입니다. 백본이 같으니 차이는
  DreamSim의 미세조정에서 온 것입니다.
- **SOP에서는 dinov3_vitb16이 더 좋습니다**(R@1 0.6532 vs 0.6189). SOP는 가구·생활용품이고
  옷이 없어서, 패션 도메인(LookBench)과 순위가 뒤집힙니다. LookBench에서는
  dreamsim_dinov2가 coarse/fine R@1 모두 앞섭니다.

## 7. 알려진 이슈

1. **ensemble 결과에 mAP@10이 없습니다.** mAP는 `478992b`에서 추가됐고, ensemble은
   그 전(2026-09-10)에 돌린 결과입니다. `dinov2_vitb14`, `dinov3_vitb16`도 같은 상태라
   **mAP가 있는 결과는 `dreamsim_dinov2_vitb14` 하나뿐**입니다. mAP로 모델을 비교하려면
   나머지를 재실행해야 합니다.
2. **ensemble 결과의 `device`가 섞여 있습니다.** real_studio_flat과 aigen_streetlook은
   `cuda`, real_streetlook과 aigen_studio는 `cpu`입니다. 지표는 같지만 소요 시간은
   비교할 수 없습니다. `dinov2_vitb14`와 `dinov3_vitb16`은 5건 모두 `cpu`입니다.
3. README의 기준 환경(python 3.13.11)과 실제 실행 환경(3.11.7)이 다릅니다.
4. `requirements.txt`에 `dreamsim` 버전이 고정돼 있지 않습니다. 실제로 쓴 것은 0.2.1이고,
   버전이 바뀌면 가중치나 전처리가 달라질 수 있습니다.

위 1·2번은 `python -m benchmark.check_settings`로 다시 확인할 수 있습니다.

## 8. 관련 커밋

| 해시 | 내용 |
|---|---|
| `4258d00` | DreamSim DINOv2 ViT-B/14 평가 대상 등록 (config, 레지스트리, README) |
| `b396dad` | 평가 결과 JSON 5개 추가 |
| `478992b` | mAP@10 지표 추가 및 이 모델 결과 갱신 |
| `65720e1` | 속도 측정 스크립트 `benchmark/speed.py` + 속도 결과 2개 |
| `f3d3251` | `probe_model.py`, `check_settings.py` 추가 |

결과 파일: `benchmark/results/dreamsim_dinov2_vitb14__lookbench-{real_studio_flat,real_streetlook,aigen_studio,aigen_streetlook}.json`,
`benchmark/results/dreamsim_dinov2_vitb14__sop-test.json`,
`benchmark/results/speed__dreamsim+dreamsim_dinov2_vitb14__{lookbench,sop}__bs16.json`
