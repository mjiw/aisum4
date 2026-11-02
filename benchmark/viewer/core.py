"""뷰어용 데이터 접근 — 캐시된 임베딩으로 검색하고, 그 결과를 이미지로 되돌린다.

평가(run_eval)가 숫자만 남기는 데 반해, 뷰어는 "실제로 무엇이 검색됐는지"를 본다.
새로 인코딩하지 않고 `.embed_cache/`에 이미 있는 임베딩을 그대로 쓰므로 GPU가 필요 없다.

갤러리 인덱스 ↔ 이미지 대응이 이 모듈의 핵심이다. run_eval은 갤러리를

    [서브셋 gallery 0..n-1] + [noise gallery 0..m-1]

순서로 이어붙인다(run_eval.py의 encode 호출 순서). 뷰어도 같은 순서를 재구성해야
검색 결과 인덱스가 올바른 이미지를 가리킨다. 순서가 어긋나면 틀린 이미지가 조용히
표시되므로, 캐시 메타의 item_id로 대조해 확인한다(verify_alignment).
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from benchmark.data import lookbench
from benchmark.encode import cache_paths
from benchmark.metrics import average_precision_at_k, evaluate
from benchmark.relevance import RelevanceIndex
from benchmark.retrieve import search

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"

# 보기 좋은 이름. 여기 없는 모델도 캐시만 있으면 자동으로 잡히며,
# 그때는 모델 키를 그대로 라벨로 쓴다(새 모델을 붙일 때 이 파일을 안 고쳐도 된다).
KNOWN_LABELS = {
    "dreamsim": "DreamSim",
    "unicom_vitl14": "UNICOM ViT-L/14",
    "gr_lite": "GR-Lite",
    "dinov3_vitb16": "DINOv3 ViT-B/16",
    "dinov2_vitb14": "DINOv2 ViT-B/14",
    # 팀원 브랜치의 후보 모델들 (viewer가 브랜치에서 바로 가져올 수 있다)
    "tianmu_mere": "Tianmu-MERE",
    "zooclaw_fashionsiglip2": "ZooClaw-FashionSigLIP2",
    "pe_core_l14_336": "PE-Core-L14-336",
    "moda_fashion_crossdomain": "MODA-Fashion-CrossDomain",
    "qwen3_vl_embedding_2b": "Qwen3-VL-Embedding-2B",
    "wemm_embedding_2b": "WeMM-Embedding-2B",
    "mcip_so400m_14_384": "MCIP ViT-SO400M-14-384",
    "franca_vitb14": "Franca ViT-B/14",
    "bge_vl_large": "BGE-VL-Large",
    "eva02_large": "EVA02-Large",
    "fgclip2_large": "FG-CLIP2-Large",
    "openvision_vitl14": "OpenVision ViT-L/14",
    "openvision2_vitl14": "OpenVision2 ViT-L/14",
    "dreamsim_dinov2_vitb14": "DreamSim-DINOv2-B/14",
    "dreamsim_synclr_vitb16": "DreamSim-SynCLR-B/16",
}
BASELINE = "dreamsim"

# 모델 줄 순서: 기준이 맨 위, 그다음은 이름이 알려진 순서, 새 모델은 뒤에 붙는다.
_ORDER = list(KNOWN_LABELS)


def label_of(model: str) -> str:
    return KNOWN_LABELS.get(model, model)


def _sort_key(model: str):
    known = model in _ORDER
    return (model != BASELINE, not known, _ORDER.index(model) if known else 0, model)


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _cache_dir() -> Path:
    return Path(load_config()["data"]["embed_cache_dir"])


def subsets() -> list:
    return list(load_config()["datasets"]["lookbench"]["configs"])


def _cache_key(config: str) -> str:
    return f"lookbench-{config}"


def _has_cache(model: str, config: str, split: str) -> bool:
    vec, meta = cache_paths(_cache_dir(), model, _cache_key(config), split)
    return vec.exists() and meta.exists()


def known_models() -> list:
    """이 저장소에서 다룰 수 있는 모델.

    캐시 폴더에 임베딩이 있는 모델(브랜치에서 가져와 돌린 것 포함)과
    로컬 config.json에 설정이 있는 모델을 합친다. KNOWN_LABELS는 이름표일 뿐
    목록의 근거가 아니다 — 안 그러면 아직 돌리지도 않은 후보 모델이 전부
    '캐시 없음'으로 뜬다.
    """
    found = set()
    for path in _cache_dir().glob("*__lookbench-*__*.npy"):
        model = path.name.split("__", 1)[0]
        if not model.endswith("-limit"):          # 스모크 테스트용 부분 캐시 제외
            found.add(model)
    return sorted(found | set(load_config().get("models", {})), key=_sort_key)


def available_models(config: str) -> list:
    """이 서브셋에서 바로 볼 수 있는 모델. 기준 모델을 맨 앞에 둔다.

    쿼리·갤러리·noise 캐시가 모두 있어야 한다. 하나라도 없으면 run_eval을
    돌려 캐시를 만들어야 하므로 목록에서 뺀다.
    """
    found = [m for m in known_models()
             if _has_cache(m, config, "query") and _has_cache(m, config, "gallery")
             and _has_cache(m, "noise", "gallery")]
    return sorted(found, key=_sort_key)


def missing_models(config: str) -> list:
    return [m for m in known_models() if m not in available_models(config)]


def _load_cache(model: str, config: str, split: str):
    vec_path, meta_path = cache_paths(_cache_dir(), model, _cache_key(config), split)
    vectors = np.load(vec_path)
    with open(meta_path, encoding="utf-8") as f:
        metas = json.load(f)
    return vectors, metas


@lru_cache(maxsize=16)
def load_queries(model: str, config: str):
    """(쿼리 임베딩, 쿼리 메타)."""
    return _load_cache(model, config, "query")


@lru_cache(maxsize=16)
def load_gallery(model: str, config: str):
    """(갤러리 임베딩, 갤러리 메타, 서브셋 갤러리 크기).

    run_eval과 같은 순서로 서브셋 갤러리 뒤에 noise 풀을 이어붙인다.
    """
    sub_vecs, sub_metas = _load_cache(model, config, "gallery")
    noise_vecs, noise_metas = _load_cache(model, "noise", "gallery")
    vectors = np.concatenate([sub_vecs, noise_vecs], axis=0)
    return vectors, sub_metas + noise_metas, len(sub_metas)


def _meta_source(config: str) -> str:
    """갤러리 메타를 읽어올 모델. 메타는 모델과 무관하게 같으므로 아무거나 된다.

    기준 모델 캐시가 아직 없을 수 있어(평가를 안 돌린 상태) 있는 것 중에 고른다.
    """
    models = available_models(config)
    if not models:
        raise FileNotFoundError(
            f"'{config}'에 임베딩 캐시가 하나도 없습니다. "
            f"python -m benchmark.run_eval --model <이름> --config {config} 를 먼저 돌리세요."
        )
    return models[0]


@lru_cache(maxsize=4)
def relevance_index(config: str, model: str = None) -> RelevanceIndex:
    """정답 판정기. 갤러리 메타는 모델과 무관하게 같으므로 한 번만 만든다."""
    _, metas, _ = load_gallery(model or _meta_source(config), config)
    ds_cfg = load_config()["datasets"]["lookbench"]
    return RelevanceIndex(metas, fine_mode=ds_cfg["fine_mode"])


def verify_alignment(config: str, models: list) -> list:
    """모델 간 갤러리 순서가 같은지 확인한다. 어긋난 모델 이름 목록을 돌려준다.

    캐시를 서로 다른 시점에 만들었거나 데이터셋 revision이 달라지면 순서가
    틀어질 수 있다. 그러면 같은 인덱스가 다른 상품을 가리켜 비교가 무의미해진다.
    """
    if not models:
        return []
    _, base_metas, _ = load_gallery(models[0], config)
    base_ids = [m["item_id"] for m in base_metas]
    bad = []
    for model in models[1:]:
        _, metas, _ = load_gallery(model, config)
        if [m["item_id"] for m in metas] != base_ids:
            bad.append(model)
    return bad


def query_rows(config: str, model: str = None) -> list:
    """쿼리 목록. (인덱스, 라벨) 튜플."""
    _, metas = load_queries(model or _meta_source(config), config)
    return [(i, f"#{i}  {m['category']}  [{', '.join(m['attrs'][:3]) or '속성 없음'}]")
            for i, m in enumerate(metas)]


def search_one(model: str, config: str, query_idx: int, topk: int):
    """한 쿼리의 top-K. (인덱스 배열, 점수 배열)."""
    q_vecs, _ = load_queries(model, config)
    g_vecs, _, _ = load_gallery(model, config)
    sims = g_vecs @ q_vecs[query_idx]
    topk = min(topk, sims.shape[0])
    part = np.argpartition(-sims, topk - 1)[:topk]
    order = part[np.argsort(-sims[part])]
    return order, sims[order]


def judge_hits(config: str, query_meta: dict, rows, criterion: str) -> np.ndarray:
    return relevance_index(config).judge_at(query_meta, rows)[criterion]


def query_scores(config: str, query_meta: dict, hits: np.ndarray, criterion: str) -> dict:
    """이 쿼리 한 건의 지표. 평가 전체 평균과 같은 정의를 쓴다."""
    n_rel = relevance_index(config).count_relevant(query_meta, criterion)
    return {
        "recall@1": float(hits[:1].any()),
        "recall@10": float(hits[:10].any()),
        "map@10": average_precision_at_k(hits, 10, n_rel),
        "hits@10": int(hits[:10].sum()),
        "n_relevant": n_rel,
    }


def _results_dir() -> Path:
    return Path(load_config()["data"]["output_dir"])


@lru_cache(maxsize=32)
def eval_metrics(model: str, config: str) -> tuple:
    """서브셋 전체 지표를 캐시된 임베딩에서 직접 계산한다.

    results/*.json을 읽지 않는 이유: 팀이 모델마다 브랜치를 따로 쓰기 때문에
    한 브랜치에는 그 모델의 결과 JSON만 있다. 반면 `.embed_cache/`는 git이
    추적하지 않아 브랜치와 무관하게 로컬에 쌓인다. 캐시에서 계산하면 어느
    브랜치에서 뷰어를 켜도 돌려본 모델이 전부 보인다.

    lru_cache를 위해 dict가 아닌 (키, 값) 튜플로 돌려준다.
    """
    q_vecs, q_metas = load_queries(model, config)
    g_vecs, _, _ = load_gallery(model, config)
    ds_cfg = load_config()["datasets"]["lookbench"]

    top_idx, _ = search(q_vecs, g_vecs, topk=max(ds_cfg["k_values"]))
    scores = evaluate(q_metas, top_idx, relevance_index(config),
                      k_values=ds_cfg["k_values"], criteria=ds_cfg["criteria"],
                      map_k_values=ds_cfg.get("map_k_values", ()), progress=False)
    return tuple(scores.items())


def results_metrics(model: str, config: str) -> dict:
    """run_eval이 남긴 결과 JSON. 현재 브랜치에 있는 것만 읽힌다."""
    path = _results_dir() / f"{model}__lookbench-{config}.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("metrics", {})


def perf(model: str) -> dict:
    """profile_perf가 남긴 속도·메모리. 없으면 빈 dict."""
    path = _results_dir() / f"perf__{model}.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def perf_table(models: list) -> list:
    """모델별 속도·메모리 요약. 기준 모델 대비 배수를 함께 넣는다.

    측정 파일이 없는 모델은 measured=False로 돌려준다. 새 모델이 추가돼도
    profile_perf만 돌리면 이 표에 자동으로 들어온다.
    """
    # perf 파일에도 "model" 키가 있으므로 덮어쓰는 순서로 합친다
    rows = [{**perf(m), "model": m, "label": label_of(m)} for m in models]
    base = next((r for r in rows if r["model"] == BASELINE and r.get("sec_per_1000")), None)
    for row in rows:
        row["measured"] = bool(row.get("sec_per_1000"))
        row["is_baseline"] = row["model"] == BASELINE
        if base and row["measured"]:
            row["speed_ratio"] = row["sec_per_1000"] / base["sec_per_1000"]
            row["gpu_ratio"] = (row["gpu_sec_per_1000"] / base["gpu_sec_per_1000"]
                                if base.get("gpu_sec_per_1000") else None)
        else:
            row["speed_ratio"] = row["gpu_ratio"] = None
    return rows


def perf_mismatch(rows: list) -> bool:
    """측정 조건(이미지 수·배치)이 서로 다르면 비교가 성립하지 않는다."""
    seen = {(r.get("config"), r.get("n_images"), r.get("batch_size"), r.get("gpu"))
            for r in rows if r.get("measured")}
    return len(seen) > 1


@lru_cache(maxsize=8)
def _dataset(config: str, split: str):
    ds_cfg = load_config()["datasets"]["lookbench"]
    data_cfg = load_config()["data"]
    ds, to_record, _ = lookbench.load_split(
        ds_cfg["repo_id"], config, split,
        cache_dir=data_cfg["hf_cache_dir"], revision=ds_cfg.get("revision"),
    )
    return ds, to_record


def _gallery_source(config: str, row: int, n_subset: int):
    """갤러리 인덱스를 (config, split, 행 번호)로 되돌린다."""
    if row < n_subset:
        return config, "gallery", row
    return "noise", "gallery", row - n_subset


def gallery_image(config: str, row: int, n_subset: int):
    src_config, split, src_row = _gallery_source(config, row, n_subset)
    ds, to_record = _dataset(src_config, split)
    return to_record(ds[src_row]).image


def query_image(config: str, row: int):
    ds, to_record = _dataset(config, "query")
    return to_record(ds[row]).image
