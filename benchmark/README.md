# benchmark — 임베딩 모델 비교 평가

기존 `search/` · `vectordb/` 파이프라인과 **분리된** 평가 전용 폴더입니다.
여러 임베딩 모델을 LookBench로 돌려 Recall / Precision / nDCG를 비교하고,
가장 좋은 모델을 골라 본 파이프라인에 이식하는 것이 목적입니다.

## 설치

```bash
pip install -r requirements.txt -r benchmark/requirements.txt
```

DINOv3는 HuggingFace **gated 저장소**라 사전 승인이 필요합니다.

1. https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m 에서 라이선스 동의
2. `hf auth login` (구버전은 `huggingface-cli login`)

## 실행

```bash
# 스모크 테스트 (200장만, CPU)
python -m benchmark.run_eval --model dinov3_vitb16 --config real_studio_flat --limit 200 --device cpu

# 본 평가
python -m benchmark.run_eval --model dinov3_vitb16 --config real_studio_flat
python -m benchmark.run_eval --model dinov3_vitb16 --config real_streetlook
```

사용 가능한 서브셋: `real_studio_flat`, `real_streetlook`, `aigen_studio`, `aigen_streetlook`, `noise`

DINOv3는 **수동 승인(gated=manual)** 저장소입니다. 모델 페이지에서 Request access 후
승인을 받아야 다운로드됩니다.

결과는 `benchmark/results/{model}__{config}.json`에 저장됩니다.
팀원 결과를 이 폴더에 모은 뒤 비교표를 뽑습니다:

```bash
python -m benchmark.compare
python -m benchmark.compare --config real_studio_flat --sort-by fine_recall@1 --csv compare.csv
```

## 구조

| 파일 | 역할 |
|---|---|
| `models/dinov3.py` | DINOv3 ViT-B/16 래퍼. `embedding/base.py` 인터페이스 준수 |
| `models/__init__.py` | 모델 레지스트리. 팀원은 여기에 한 줄 추가 |
| `data/lookbench.py` | HF 원본 → 공통 스키마(image, item_id, category, attrs) |
| `encode.py` | 임베딩 → `.npy` 캐시. 갤러리 60K장은 모델당 1회만 |
| `retrieve.py` | 코사인 top-K (numpy 행렬곱) |
| `relevance.py` | **정답 판정. 팀 공통 고정** |
| `metrics.py` | Recall@K / Precision@K / nDCG@K 집계 |
| `run_eval.py` | CLI 진입점 |
| `compare.py` | `results/*.json`을 모아 모델 비교표 출력 |
| `probe_schema.py` | 데이터셋 실제 스키마 확인용 일회성 스크립트 |

인코딩 → 검색 → 판정 → 집계 4단계이고, 임베딩이 캐시되므로
metric을 고쳐 다시 돌릴 때는 인코딩을 건너뜁니다.

## 지표

**LookBench — 4개**

| 지표 | 정답 조건 |
|---|---|
| `coarse_recall@1`, `@10` | category 일치 |
| `fine_recall@1`, `@10` | category + 속성 전부 일치 |

**SOP — 2개**

| 지표 | 정답 조건 |
|---|---|
| `exact_recall@1`, `@10` | `item_id` 일치 (같은 상품) |

Recall@K는 검색 벤치마크 관례대로 **top-K 안에 정답이 하나라도 있으면 1**(hit rate)이며
쿼리 전체 평균입니다. LookBench 논문과 SOP 표준이 모두 이 정의를 씁니다.

SOP에 `fine`을 쓰지 않는 이유는 속성 라벨이 없어 `coarse`와 같은 값이 되기 때문입니다.
그 값은 카테고리 12종 기준이라 어떤 모델이든 0.94~0.97이 나와 판별력이 없습니다.
LookBench에 `exact`를 쓰지 않는 이유는 공식 지표가 아니기 때문입니다
(LookBench는 Coarse Recall / Fine Recall / nDCG / MRR / MAP를 보고합니다).

`fine_mode`는 `config.json`에서 바꿉니다.
- `exact` (기본): 쿼리와 갤러리의 속성 집합이 완전히 같아야 정답
- `subset`: 쿼리 속성이 갤러리 아이템에 모두 포함되면 정답

> 논문 문구가 "exact category and all attributes to match"라 집합 상등인지 포함인지
> 모호하지만, 실측상 차이는 최대 1.6pp입니다. 속성 개수가 대부분 정확히 4개
> (main_attribute 1 + other_attributes 3)라 두 정의가 거의 같아지기 때문입니다.

## 데이터셋은 git에 올리지 않습니다

LookBench 원본은 **2.1GB**이고 개별 파케이 파일이 최대 **350MB**입니다.
GitHub은 파일당 100MB가 하드 제한이라 푸시가 거부되고, Git LFS 무료 용량(1GB)에도
들어가지 않습니다. HuggingFace가 이미 무료로 호스팅하므로 옮길 이유도 없습니다.

git으로 공유하는 것은 **코드와 결과 JSON(36KB)뿐**입니다. 데이터는 코드를 받은
사람이 `load_dataset`으로 자동으로 내려받습니다.

**대신 revision을 고정합니다.** LookBench는 반기마다 갱신되는 live 벤치마크라,
고정하지 않으면 나중에 받은 팀원이 다른 데이터로 평가해 숫자 비교가 깨집니다.

```json
"revision": "151449aa3a906899f29fd3e0a81a21e83a48c569",
"dataset_version": "v20251201"
```

`config.json`의 이 값이 모든 실행에 적용되고 결과 JSON에도 기록됩니다.
벤치마크 새 버전으로 옮길 때는 이 값을 바꾸고 **전원이 다시 실행**해야 합니다.

## 실험 환경 맞추기 (팀 공유)

### 팀원 진행 순서

```bash
git fetch origin
git checkout -b test/{자기모델명} origin/test/dino-v3
pip install -r requirements.txt -r benchmark/requirements.txt
```

그다음 클로드 코드에 이렇게 요청하면 됩니다:

> benchmark 폴더 코드 참고해서 동일한 환경으로 **{자기 모델}** 성능 평가 진행해줘.
> LookBench 4개 서브셋이랑 SOP 데이터셋 전부.

끝나면 `benchmark/results/*.json`을 커밋해서 push하세요.
모으면 `python -m benchmark.compare`로 자동 비교표가 나옵니다.

### HuggingFace 토큰 (대부분 불필요)

데이터셋(LookBench, SOP)은 **gated가 아니라 토큰 없이 받아집니다.** 토큰이 필요한 건
**gated 모델**을 쓸 때뿐입니다 — 예를 들어 `facebook/dinov3-*`는 Meta 수동 승인이 필요합니다.

해당되는 경우에만 `hf auth login` 하시면 되고, 환경변수 `HF_TOKEN`이나
저장소 루트의 `token.txt`(gitignore 처리됨, 각자 본인 토큰)도 인식합니다.

### 반드시 같아야 하는 것 — 다르면 비교 무효

| 항목 | 값 | 왜 |
|---|---|---|
| 판정 로직 | `relevance.py` | 정답 기준이 다르면 숫자의 의미가 달라짐 |
| 집계 | `metrics.py` | Recall@K 정의 |
| LookBench revision | `151449aa3a906899f29fd3e0a81a21e83a48c569` (`v20251201`) | **반기마다 갱신되는 live 벤치마크** |
| SOP revision | `24a1b9b8ec6c0b1fc4dd324f24b2d829413a6c69` | |
| noise 풀 | **포함** (LookBench) | 빼면 갤러리가 1/15로 줄어 최대 11pp 부풀려짐 |
| 임베딩 정규화 | **L2 정규화 필수** | 안 하면 코사인이 아니라 내적이 되어 결과가 달라짐 |
| SOP 프로토콜 | leave-one-out, 자기 자신 제외 | 빼지 않으면 Recall@1이 1.0으로 나옴 |

`config.json`에 다 들어 있으므로 이 저장소를 그대로 쓰면 자동으로 맞습니다.

### 모델마다 달라도 되는 것 — 단, 결과에 기록할 것

| 항목 | 예 |
|---|---|
| 모델 / 가중치 | `hf_id` + `revision` (결과 JSON의 `model_metadata`에 자동 기록) |
| 임베딩 차원 | DINOv3 ViT-B/16 = 768 |
| pooling | `cls` / `mean` — 모델마다 최적이 다르므로 어느 쪽을 썼는지 반드시 남길 것 |
| batch_size | 결과에 영향 없음 (메모리 사정에 맞게) |

### 평가 설정

| | LookBench | SOP |
|---|---|---|
| 판정 기준 | `coarse`, `fine` | `exact` |
| K | 1, 10 | 1, 10 |
| 지표 개수 | **4개** | **2개** |
| fine_mode | `exact` | — |
| exclude_self | false | **true** |
| 쿼리 / 갤러리 | 서브셋별 상이 (아래 표) | 60,502 / 60,502 |

서브셋 통합 점수는 논문 Table 3과 같이 **쿼리 수 가중 평균**을 씁니다.
서브셋 쿼리 수가 160~1,011로 6배 차이나 단순 평균과 값이 다릅니다.

### 참고: 기준 실행 환경

숫자가 안 맞을 때 대조용입니다. 버전이 달라도 대체로 재현되지만,
전처리(`AutoImageProcessor`) 동작이 바뀌면 임베딩이 달라질 수 있습니다.

```
python 3.13.11 / Windows 11
torch 2.13.0+cu130 (CUDA 13.0, RTX 4090)
transformers 5.4.0 / datasets 3.6.0 / numpy 2.4.1
```

Franca 결과는 아래 환경에서 나왔습니다. Franca는 transformers를 거치지 않고
전처리를 `models/franca.py`에서 torchvision으로 직접 정의하므로
`AutoImageProcessor` 버전 차이의 영향을 받지 않습니다.

```
python 3.14.5 / Windows 11
torch 2.13.0+cu130 (CUDA 13.1, RTX 5070 Ti)
torchvision 0.28.0+cu130 / datasets 5.0.1 / numpy 2.5.2
```

### 새 모델 추가하는 법

1. `benchmark/models/<이름>.py`에 `ImageEmbeddingModel` 상속 클래스 작성
   (`embed()`가 L2 정규화를 해주므로 `_embed_raw`만 구현하면 됨)
2. `benchmark/models/__init__.py`의 `_REGISTRY`에 한 줄 추가
3. `config.json`의 `models`에 `hf_id`, `revision`, `embed_dim`, `pooling`, `batch_size` 추가

HuggingFace가 아니라 **torch.hub로 배포되는 모델**(Franca 등)은 `hf_id`/`revision` 대신
`hub_repo`(태그까지 고정), `hub_entry`, `weights`를 쓰고 전처리를 직접 정의합니다.
`models/franca.py`를 참고하세요. 이때 `torch.hub.load(..., trust_repo=True)`가 필요합니다 —
저장소 코드를 실행해도 되는지 묻는 프롬프트가 뜨는데, 비대화형 실행에서는 EOFError로 죽습니다.

```bash
python -m benchmark.run_eval --model <이름> --config real_studio_flat
python -m benchmark.run_eval --model <이름> --config real_streetlook
python -m benchmark.run_eval --model <이름> --config aigen_studio
python -m benchmark.run_eval --model <이름> --config aigen_streetlook
python -m benchmark.run_eval --model <이름> --dataset sop
python -m benchmark.compare
```

소요 시간은 갤러리 6만 장 인코딩이 대부분입니다. RTX 4090 기준 데이터셋당 10분 내외이고,
LookBench는 noise 임베딩이 서브셋 간 재사용되어 첫 서브셋만 오래 걸립니다.

## 팀 협업 규칙

1. **`relevance.py`와 `metrics.py`는 공통입니다.** 고치면 반드시 공유하고 전원 재실행.
   판정 기준이 다르면 모델 간 숫자 비교가 무의미해집니다.
2. 같은 `--config`, 같은 `k_values`로 돌립니다.
3. 결과 JSON을 그대로 모아 비교합니다. `model_metadata`에 모델 식별 정보가 들어갑니다.
4. `dataset_revision`이 서로 다른 결과는 비교하지 마세요. 데이터가 다릅니다.
5. `.embed_cache/`(수백 MB)는 커밋하지 않습니다. 각자 로컬에서 재생성됩니다.

## 담당

| 모델 | 차원 | 담당 |
|---|---|---|
| DINOv3 ViT-B/16 | 768 | (이 폴더) |
| DINOv2 ViT-B/14 | 768 | 파이프라인 검증 대역 겸 비교 baseline |
| DreamSim ensemble | 1792 | 현재 파이프라인 baseline |
| Franca ViT-B/14 | 768 | `models/franca.py` (In21K 224, RASA 미사용) |

## 데이터셋 구조 (확인 완료)

컬럼: `image`, `raw_image`, `category`, `main_attribute`, `other_attributes`,
`bbox`, `item_ID`, `task`, `difficulty` — 이미지 2종만 `Image` 타입이고 나머지는 전부 문자열입니다.

| config | query | gallery | + noise = 실제 갤러리 |
|---|---|---|---|
| `real_studio_flat` | 1,011 | 3,951 | **62,226** |
| `real_streetlook` | 981 | 3,278 | **61,553** |
| `noise` | — | 58,275 | (공용 distractor 풀) |

> ★ **평가 갤러리 = 서브셋 gallery + `noise` 풀**입니다. 합계가 논문 표의 수치와
> 정확히 일치합니다. noise를 빼면 갤러리가 1/15로 줄어 점수가 크게 부풀려집니다.
> `run_eval.py`가 자동으로 합치며 `--no-noise`로 끌 수 있습니다(빠른 확인용).
> noise 임베딩은 서브셋 간 캐시가 재사용되므로 모델당 한 번만 계산됩니다.

## 데이터 처리 시 주의점 (실측으로 확인)

**`image`는 이미 `bbox`로 잘려 있습니다.** row 0의 bbox `[195, 131, 1068, 1394]`는
폭 873 × 높이 1263이고 `image.size`가 정확히 (873, 1263)입니다. 따라서 `image`에
bbox를 또 적용하면 **이중 crop**이 되어 엉뚱한 영역이 잘립니다.
`load_split(use_bbox=...)`의 기본값은 `False`이며, `raw_image`에서 직접 자를 때만 True로 줍니다.
bbox 형식은 `[x1, y1, x2, y2]` 절대 픽셀입니다.

**`noise` 풀은 순수 distractor입니다.** 58,275장 전부 `category`가 `'noise data'`
한 종류이고 속성은 비어 있습니다. 따라서 어떤 쿼리와도 카테고리가 일치하지 않아
`coarse`/`fine`/`graded` 모두 0이 됩니다. 정답으로 잡히는 일 없이 진짜 정답을
top-K 밖으로 밀어내는 역할만 합니다.

**`item_ID`**는 서브셋에서는 split 내 순번(`'0'`, `'1'`, ...), noise 풀에서는
`'noise_1'` 형식입니다. `load_split`이 `"{config}:{item_ID}"`로 네임스페이스를
붙여 서브셋 간 혼동을 막습니다(값 자체는 이미 구분되지만 서브셋을 여러 개 합칠 때
안전합니다). 쿼리와 같은 서브셋의 갤러리는 접두사가 같으므로 `exact` 매칭이 유지됩니다.

`other_attributes`는 `'applique, printed, bead'` 같은 쉼표 구분 문자열입니다.
`main_attribute`와 합쳐 소문자·중복제거·정렬한 것이 `attrs`가 됩니다.

## 확인이 필요한 항목

- [ ] `fine_mode` exact / subset 중 공식 리더보드와 맞는 쪽
- [ ] `task` / `difficulty` 컬럼 활용 여부 (현재 미사용)
- [ ] gallery 쪽도 `image`(crop)를 쓰는 게 맞는지, `raw_image`(전체 상품 사진)가 나은지 비교

## 승자 이식 방법

`models/dinov3.py`는 `embedding/base.py`의 `ImageEmbeddingModel`을 그대로 상속합니다.
DINOv3가 선택되면:

1. `benchmark/models/dinov3.py` → `embedding/models/dinov3.py`로 이동
2. `embedding/config.json`의 `models`에 `dinov3` 항목 추가

`embedding/model_loader.py`의 `_REGISTRY`에는 `"dinov3"`가 이미 등록돼 있어 수정이 필요 없습니다.
