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

## 속도 · 메모리 측정

현업 투입 판단에는 정확도만으로 부족합니다. 1,000장을 몇 초에 처리하는지,
VRAM을 얼마나 먹는지를 같은 조건에서 재서 `results/perf__{model}.json`에 남깁니다.

```bash
python -m benchmark.profile_perf --model dreamsim
python -m benchmark.profile_perf --model unicom_vitl14
```

같은 이미지(기본: real_studio_flat 쿼리 200장), 같은 배치로 재야 비교가 됩니다.
기본값을 바꿨다면 모든 모델을 같은 옵션으로 다시 재세요. 뷰어가 조건이 다르면
경고를 띄웁니다.

> **절대 시간은 머신 상태에 따라 흔들립니다.** 같은 모델이 다른 날 2배 차이로
> 나오기도 합니다. 시트에는 DreamSim=1 기준 **배수**를 적는 편이 안전합니다.
> 배수는 같은 실행 안에서 재면 재현성이 높습니다(GR-Lite GPU 기준 2.37~2.44).

## 모델 비교 사이트 (Web GUI)

LookBench 테스트 이미지로 **실제 검색 결과**를 보고 DreamSim(현행 파이프라인)과
비교하는 페이지입니다. 링크만 있으면 GPU·설치 없이 열립니다.

- **검색 결과**: DreamSim을 맨 위에 고정하고 고른 모델을 아래에 나란히 놓습니다.
  같은 순위가 세로로 맞물려 바로 비교되고, 정답은 초록 테두리, DreamSim도 찾은
  이미지는 `D3`(DreamSim 3위)처럼 표시됩니다. 모델 순서는 서브셋 성능순(R@1)·
  이 쿼리 성적순(AP@10)·기본 중에서 고르고, 각 줄에 `3위/10`처럼 순위가 붙습니다.
  화면 상태가 주소에 담겨 링크로 공유됩니다.
- **성능 순위**: 정확도(R@1 / R@10 / mAP@10, DreamSim 대비 %p와 쿼리별 승·무·패),
  속도·자원(1000장 처리 시간, 쿼리 1장 지연, 이미지 토큰 수, VRAM, 파라미터,
  모델 크기, 차원, 6만 장 인덱스 크기, 로딩 시간). 처음엔 R@1·처리 시간 순위로
  나오고, 열 제목을 누르면 그 지표로 순위가 다시 매겨집니다.
- **실험 설정**: 공통 평가 조건(데이터셋 revision, 서브셋 크기, 정답 기준, 검색 방식,
  실행 환경, 속도 측정 방법)과 모델별 실제 실행값(입력 해상도, 평가·속도 배치,
  임베딩 차원, 풀링, 가중치 revision, 브랜치@커밋, 비고). 결과 JSON에 기록된 값을
  그대로 보여줍니다.

비교 대상은 `benchmark/compute/lineup.json`에 있습니다(모델 키·이름·브랜치·링크,
순서 = 화면 순서). 모델마다 실행 인자가 달라야 하면 `run_args`(예: VRAM이 모자란
WeMM의 `["--batch-size", "2"]`)와 이유를 적는 `note`를 넣습니다. 둘 다 사이트의
실험 설정 화면에 그대로 나옵니다. 실패한 서브셋·속도 측정은 30초 뒤 한 번 더 시도합니다. 만드는 과정은 세 단계이고 GPU는 1단계에만 필요합니다.

```bash
python -m benchmark.compute status                  # 무엇이 준비됐는지
python -m benchmark.compute run --models all        # 1) GPU: 빠진 것만 돌린다
python -m benchmark.viewer.export_static            # 2) site/ 굽기 (GPU 불필요)
python -m benchmark.viewer.deploy                   # 3) GitHub Pages로 올리기
```

**1단계(워커)가 하는 일.** 모델마다 그 브랜치를 임시 워크트리로 펼쳐 **브랜치 코드
그대로** 임베딩을 만들고(`run_eval`), 결과를 쿼리별 top-10·정답 여부·AP로 압축해
`results/retrieval/{model}__lookbench-{subset}.npz`(수십 KB)에 남깁니다. 속도는
이쪽 `profile_perf.py`를 덮어씌워 모든 모델을 같은 방식으로 잽니다.
정답 판정·집계는 **이 브랜치의 `relevance.py`/`metrics.py`로 모든 모델에 똑같이**
합니다(브랜치마다 metrics.py 버전이 달라 그대로 모으면 잣대가 섞입니다). 그래서
사이트 숫자는 각 브랜치가 남긴 JSON과 소수점 아래가 조금 다를 수 있습니다.
TIPSv2처럼 `benchmark/` 틀을 쓰지 않는 브랜치는 어댑터(`compute/adapters.py`)가
모델 코드만 가져와 같은 틀에 얹습니다.

### GPU 바꾸기

워커를 어디서 돌릴지는 `benchmark/compute/gpu.json`의 백엔드가 정합니다.
머신마다 다른 값은 `gpu.local.json`(git 제외)에 덮어씁니다.

```jsonc
// benchmark/compute/gpu.local.json — 예시
{
  "default": "lab",                       // 기본 백엔드를 바꾸고 싶을 때
  "backends": {
    "local": { "python": "C:/Users/me/venvs/aisum4-bench/Scripts/python.exe" },
    "lab":   { "type": "ssh", "host": "me@gpu.lab.ac.kr", "port": 22,
               "repo_dir": "~/aisum4", "python": "~/venvs/bench/bin/python" }
  }
}
```

```bash
python -m benchmark.compute gpu --gpu lab           # 연결·GPU 확인
python -m benchmark.compute run --models all --gpu lab
AISUM_GPU=lab python -m benchmark.compute run       # 환경변수로 골라도 된다
```

- `local`: 이 PC의 GPU에서 워커를 하위 프로세스로 돌립니다.
- `ssh`: 원격 머신에서 `git fetch` → `ref` 체크아웃 → 워커 실행 후, 결과 파일
  (`retrieval/*.npz`, `perf__*.json`, 결과 JSON)만 `scp`로 받아옵니다. 원격에는
  이 저장소 clone과 파이썬 환경이 있어야 하고, **이쪽 변경은 먼저 push** 해야 반영됩니다.
  원격 clone은 워커 전용으로 두세요(매번 체크아웃합니다).
- 다른 방식(작업 큐, 클라우드 API 등)은 `compute/backends.py`의 `Backend`를
  상속해 `run_model`·`check`를 구현하고 `TYPES`에 등록하면 됩니다. 결과 파일
  규칙만 지키면 사이트 쪽은 그대로입니다.

### 실행 환경

10개 모델을 한 환경에서 돌리려면 transformers 5.x가 필요합니다(WeMM은 5.2 이상).
팀 기준 환경(`transformers 5.4.0`)에 맞춘 venv를 따로 두는 걸 권합니다.

```bash
python -m venv ~/venvs/aisum4-bench
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
pip install transformers==5.4.0 datasets==5.0.1 "huggingface_hub>=1.5,<2" dreamsim==0.2.1     open_clip_torch==3.3.0 timm==1.0.29 peft accelerate qwen-vl-utils sentencepiece scipy
```

### 사이트 굽기 · 배포

- 정확도 요약은 **서브셋 전체 쿼리**로 계산합니다('전체' = 네 서브셋 쿼리 수 가중).
  이미지까지 보여주는 쿼리는 서브셋당 `--queries`개(기본 50)이며, 절반은 모델 간
  결과가 갈리는 쿼리, 절반은 무작위입니다.
- 빌드할 때 결과물의 갤러리 순서를 로컬 데이터셋과 대조합니다. 어긋나면 썸네일이
  엉뚱한 상품을 가리키므로 멈춥니다.
- `site/index.html`을 더블클릭해도 열립니다(서버 불필요). `site/`는 git 제외입니다.
- 배포 대상은 `viewer/deploy.json`의 전용 저장소입니다. 매번 커밋 하나로 덮어써
  썸네일 이력이 쌓이지 않습니다. 무료 Pages는 공개 페이지(링크를 알면 누구나 열람)라
  검색엔진 노출만 `noindex`로 막아 두었습니다.

## 로컬 탐색 뷰어 (Streamlit)

라인업에 없는 모델을 잠깐 시험하거나 전체 쿼리를 훑어볼 때 쓰는 로컬 도구입니다.
`.embed_cache/`의 임베딩을 직접 읽으므로 GPU 없이 즉시 검색합니다.

```bash
streamlit run benchmark/viewer/app.py
python -m benchmark.viewer.remote --branch test/zooclaw-fashionsiglip2     --model zooclaw_fashionsiglip2 --configs real_studio_flat   # 브랜치 모델 돌리기
```

사이드바의 「브랜치에서 모델 가져오기」는 위 `remote` 명령과 같습니다(체크아웃
불필요, 남의 브랜치 코드를 그대로 실행하므로 같은 팀 저장소 전제).
다른 기기에서 열려면 `VIEWER_PASSWORD`로 잠그고, 방화벽은 허용할 대역만 여세요.
**포트를 인터넷 전체에 열지 마세요** — 버튼 하나로 GPU 작업을 돌리는 앱입니다.

```bash
VIEWER_PASSWORD='긴-비밀번호' streamlit run benchmark/viewer/app.py
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
| `metrics.py` | Recall@K / mAP@K 집계 |
| `run_eval.py` | CLI 진입점 |
| `compare.py` | `results/*.json`을 모아 모델 비교표 출력 |
| `profile_perf.py` | 모델별 속도·메모리 측정 → `results/perf__{model}.json` |
| `compute/lineup.json` | 비교 사이트의 모델 라인업 (키·이름·브랜치·링크, 화면 순서) |
| `compute/gpu.json` | GPU 백엔드 정의 (`gpu.local.json`으로 머신별 덮어쓰기) |
| `compute/backends.py` | 워커를 어디서 돌릴지 — local / ssh |
| `compute/worker.py` | GPU 머신에서 브랜치 모델 실행 → 결과물·속도 측정 |
| `compute/artifacts.py` | 임베딩 캐시 → 쿼리별 top-10·정답·AP 압축 (`results/retrieval/`) |
| `compute/adapters.py` | `benchmark/` 틀이 아닌 브랜치 모델(TIPSv2)을 같은 틀에 얹기 |
| `compute/worktree.py` | 브랜치를 임시 워크트리로 펼치기 (체크아웃 불필요) |
| `viewer/export_static.py` | 결과물 + 테스트 이미지 → 정적 사이트 `site/` |
| `viewer/web/` | 사이트 화면 (HTML/CSS/JS, 서버 없음) |
| `viewer/deploy.py` | `site/`를 GitHub Pages 전용 저장소로 배포 |
| `viewer/app.py` | 로컬 탐색 뷰어 (Streamlit). 캐시된 임베딩만 사용 |
| `viewer/core.py` | Streamlit 뷰어용 검색·판정·이미지 조회 |
| `viewer/remote.py` | 라인업 밖 브랜치 모델을 임시 워크트리에서 실행 |
| `probe_schema.py` | 데이터셋 실제 스키마 확인용 일회성 스크립트 |

인코딩 → 검색 → 판정 → 집계 4단계이고, 임베딩이 캐시되므로
metric을 고쳐 다시 돌릴 때는 인코딩을 건너뜁니다.

## 지표

**LookBench — 6개**

| 지표 | 정답 조건 |
|---|---|
| `coarse_recall@1`, `@10`, `coarse_map@10` | category 일치 |
| `fine_recall@1`, `@10`, `fine_map@10` | category + 속성 전부 일치 |

**SOP — 3개**

| 지표 | 정답 조건 |
|---|---|
| `exact_recall@1`, `@10`, `exact_map@10` | `item_id` 일치 (같은 상품) |

Recall@K는 검색 벤치마크 관례대로 **top-K 안에 정답이 하나라도 있으면 1**(hit rate)이며
쿼리 전체 평균입니다. LookBench 논문과 SOP 표준이 모두 이 정의를 씁니다.

mAP@10은 쿼리별 AP@10의 평균입니다.
`AP@K = (1 / min(R, K)) * Σ_{k≤K} P@k · rel(k)`, R은 갤러리 전체 정답 수
(SOP는 자기 자신 제외). Recall@K와 달리 top-K 안에서 정답이 **몇 개, 몇 위에**
들어왔는지까지 반영합니다. 분모가 min(R, K)라 LookBench coarse처럼 정답이 수천 개면
top-10을 전부 정답으로 채워야 1이 되고, SOP처럼 상품당 이미지가 적으면 R이 분모가 됩니다.

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
| 집계 | `metrics.py` | Recall@K, mAP@K 정의 |
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

### 새 모델 추가하는 법

1. `benchmark/models/<이름>.py`에 `ImageEmbeddingModel` 상속 클래스 작성
   (`embed()`가 L2 정규화를 해주므로 `_embed_raw`만 구현하면 됨)
2. `benchmark/models/__init__.py`의 `_REGISTRY`에 한 줄 추가
3. `config.json`의 `models`에 `hf_id`, `revision`, `embed_dim`, `pooling`, `batch_size` 추가

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
| GR-Lite (ViT-L/16) | 1024 | `models/gr_lite.py` |
| DreamSim ensemble | 1792 | 현재 파이프라인 baseline |

### GR-Lite 참고사항

`srpone/gr-lite`는 **LookBench를 만든 팀이 낸 패션 검색 전용 모델**입니다.
DINOv3 ViT-L/16을 파인튜닝했지만 가중치가 저장소에 통째로 들어 있어
`facebook/dinov3-*` 승인 없이 받아집니다(apache-2.0, gated 아님).

다른 모델과 다르게 처리해야 하는 점이 세 가지 있습니다.

- **transformers 5.x에서는 못 돌립니다.** 저장소의 모델링 코드가 4.49 기준으로
  작성돼 5.x의 `PreTrainedModel` 계약을 만족하지 않습니다. 5.17에서
  `AttributeError: 'GRLiteModel' object has no attribute 'all_tied_weights_keys'`로
  죽습니다. 그래서 `requirements.txt`에 `transformers>=4.56,<5`로 상한을 걸었습니다.
  (기존 DINOv2/v3 결과는 5.4.0에서 나온 것이지만 이미 커밋돼 있어 재실행이 필요 없습니다.
  재실행하게 되면 전처리가 달라질 수 있으니 그때 확인하세요.)
- **전처리를 직접 만듭니다.** 저장소에 `preprocessor_config.json`이 없어
  `AutoImageProcessor`가 뜨지 않습니다. 모델 카드가 명시한 336x336 리사이즈 +
  ImageNet 정규화를 `models/gr_lite.py`에 그대로 옮겼습니다. 이 때문에
  `torchvision` 의존성이 추가됐습니다.
- **커스텀 코드를 실행합니다.** `model_type`이 `gr_lite`라 transformers에
  내장돼 있지 않고 저장소의 `modeling_gr_lite.py`를 받아 돌립니다
  (`trust_remote_code=True`). revision을 고정해 실행 코드도 함께 고정했습니다.

`pooler_output`이 이미 L2 정규화된 CLS라 `base.embed()`의 정규화는 멱등입니다.
pooling은 모델 내부에 고정돼 있어 `cls`/`mean` 선택이 불가능합니다.

**실행 (CUDA 머신 권장)**

```bash
python -m benchmark.run_eval --model gr_lite --config real_studio_flat
python -m benchmark.run_eval --model gr_lite --config aigen_studio
python -m benchmark.run_eval --model gr_lite --config real_streetlook
python -m benchmark.run_eval --model gr_lite --config aigen_streetlook
python -m benchmark.run_eval --model gr_lite --dataset sop
python -m benchmark.compare
```

첫 명령이 noise 풀 58,275장을 인코딩하므로 가장 오래 걸리고, 이후 서브셋은
그 캐시를 재사용합니다. 인코딩 총량은 약 13만 장입니다.

> **GR-Lite는 무겁습니다.** DINOv3 ViT-B/16(224px)에 비해 ViT-L/16 + 336px라
> 이미지당 연산량이 약 10배입니다. 배치 크기를 키워도 나아지지 않습니다
> (M4 실측 16/32/64 → 3.9/3.9/4.1 img/s, 연산 병목).
> VRAM이 부족하면 `--batch-size`를 낮추세요. 결과에는 영향이 없습니다.

> **Apple Silicon에서는 비권장**: `embedding/base.py`는 `cuda` 아니면 `cpu`만
> 고르므로 `models/gr_lite.py`에서만 MPS로 승격합니다(공유 파일은 건드리지 않음).
> 그래도 M4 실측 약 4 img/s라 13만 장에 **9시간 이상** 걸립니다. CUDA 머신을 쓰세요.
> `--device cpu`를 주면 MPS 승격 없이 그대로 CPU로 갑니다.

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
