# 임베딩 속도 측정

DreamSim, DINOv3 ViT-B/16, OpenVision2 ViT-L/14 세 모델이 이미지를 임베딩하는 데
실제로 몇 초 걸리는지 같은 조건에서 측정합니다.

## 설치

```bash
pip install -r requirements.txt
```

DINOv3(`facebook/dinov3-vitb16-pretrain-lvd1689m`)는 Hugging Face 승인이 필요한 모델입니다.
모델 페이지에서 접근 승인을 받은 뒤 로그인해 주세요.

```bash
huggingface-cli login
```

## 실행 (repo 루트에서)

```bash
# 세 모델 모두 측정
python -m benchmark.speed --models dreamsim dinov3_vitb16 openvision2_vitl14 --datasets sop_test lookbench

# 결과 표 출력
python -m benchmark.compare --speed
```

- 데이터셋(SOP test, LookBench)은 처음 실행할 때 Hugging Face에서 자동으로 받습니다
  (`benchmark/.hf_cache/`). 각 데이터셋에서 3,000장을 고정 시드로 뽑아 씁니다.
- 결과는 `benchmark/results/speed/<모델>__<데이터셋>.json`에 저장됩니다.
  이미 결과 파일이 있는 조합은 건너뛰므로, 다시 재려면 `--force`를 붙이세요.
- `--datasets`를 빼면 로컬 이미지 폴더(`folder158`)도 측정하려 하므로 위처럼 지정해 주세요.

## 결과 읽는 법

| 항목 | 의미 |
|---|---|
| `e2e_s` | 표본 전체(3,000장)를 임베딩하는 데 걸린 실제 시간(초). 디코딩 + 전처리 + 모델 forward |
| `fwd_s` | 위 시간 중 모델 forward만 |
| `e2e_s_per_1000`, `fwd_s_per_1000` | 1,000장 기준으로 환산한 시간 |
| `peak_vram_gb` | 최대 VRAM 사용량 |

모든 시간은 3회 측정한 값의 중앙값입니다. GPU 과부하를 막으려고 배치 사이에 쉬는데,
쉬는 시간은 측정에 포함되지 않습니다.

GPU·배치 크기가 다르면 숫자를 직접 비교할 수 없으니, 비교할 결과는 같은 머신에서 측정해 주세요.
`results/speed/`에 있는 DreamSim, OpenVision2 결과는 NVIDIA RTX PRO 5000 Blackwell,
batch 16, fp32 조건에서 측정했습니다.

## 주요 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--sample` | 3000 | 데이터셋에서 뽑을 이미지 수 |
| `--batch-size` | config 값(16) | 배치 크기 |
| `--runs` | 3 | 반복 측정 횟수 |
| `--device` | 자동 | `cpu` 또는 `cuda` |
| `--gpu-mem-fraction` | 0.35 | 프로세스가 쓸 수 있는 VRAM 비율 상한 |
