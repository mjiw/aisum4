"""검색 결과물 — 임베딩 캐시를 쿼리별 top-K로 압축해 저장하고 읽는다.

임베딩 캐시는 모델 하나에 수백 MB라 옮기기도, 커밋하기도 어렵다. 사이트에
필요한 건 "쿼리마다 무엇을 몇 위로 찾았고 그게 정답인지"뿐이므로, GPU 머신에서
그것만 뽑아 모델·서브셋당 수십 KB짜리 npz로 남긴다.

    results/retrieval/{model}__lookbench-{subset}.npz

정답 판정은 **이 브랜치의** relevance.py로 한다. 모델 브랜치마다 metrics.py
버전이 달라(mAP 유무 등) 브랜치가 남긴 숫자를 그대로 모으면 잣대가 섞인다.
임베딩만 브랜치 코드로 만들고, 판정·집계는 모든 모델에 같은 코드를 쓴다.

갤러리 순서는 [서브셋 gallery] + [noise gallery]다(run_eval과 같다). 행 번호가
n_subset 이상이면 noise 풀의 (행 - n_subset)번째 이미지다.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from benchmark.compute import BENCH, REPO, retrieval_dir

CRITERIA = ("coarse", "fine")
TOPK = 10
MAP_K = 10


def _bench_config() -> dict:
    with open(BENCH / "config.json", encoding="utf-8") as f:
        return json.load(f)


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO / p).resolve()


def path_of(model: str, subset: str) -> Path:
    return retrieval_dir() / f"{model}__lookbench-{subset}.npz"


def exists(model: str, subset: str) -> bool:
    return path_of(model, subset).exists()


def ids_digest(metas: list) -> str:
    """item_id 순서의 지문. 모델 간 갤러리 순서가 같은지 비교하는 데 쓴다."""
    h = hashlib.sha1()
    for m in metas:
        h.update(m["item_id"].encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _load_cache(cache_dir: Path, model: str, subset: str, split: str):
    from benchmark.encode import cache_paths

    vec_path, meta_path = cache_paths(cache_dir, model, f"lookbench-{subset}", split)
    if not (vec_path.exists() and meta_path.exists()):
        raise FileNotFoundError(f"임베딩 캐시가 없습니다: {vec_path.name}")
    with open(meta_path, encoding="utf-8") as f:
        metas = json.load(f)
    return np.load(vec_path), metas


def build(model: str, subset: str, source: dict = None) -> Path:
    """캐시된 임베딩으로 전체 쿼리를 검색해 결과물을 만든다 (CPU 작업)."""
    from benchmark.metrics import average_precision_at_k
    from benchmark.relevance import RelevanceIndex
    from benchmark.retrieve import search

    cfg = _bench_config()
    ds_cfg = cfg["datasets"]["lookbench"]
    cache_dir = _abs(cfg["data"]["embed_cache_dir"])

    q_vecs, q_metas = _load_cache(cache_dir, model, subset, "query")
    g_vecs, g_metas = _load_cache(cache_dir, model, subset, "gallery")
    n_vecs, n_metas = _load_cache(cache_dir, model, ds_cfg["noise_config"], "gallery")
    gallery = np.concatenate([g_vecs, n_vecs], axis=0)
    metas = g_metas + n_metas

    rows, scores = search(q_vecs, gallery, TOPK)
    index = RelevanceIndex(metas, fine_mode=ds_cfg["fine_mode"])

    n_q = len(q_metas)
    hits = {c: np.zeros((n_q, TOPK), dtype=bool) for c in CRITERIA}
    n_rel = {c: np.zeros(n_q, dtype=np.int32) for c in CRITERIA}
    ap = {c: np.zeros(n_q, dtype=np.float32) for c in CRITERIA}
    for qi, qmeta in enumerate(q_metas):
        judged = index.judge_at(qmeta, rows[qi])
        for c in CRITERIA:
            hits[c][qi] = judged[c]
            n_rel[c][qi] = index.count_relevant(qmeta, c)
            ap[c][qi] = average_precision_at_k(judged[c], MAP_K, int(n_rel[c][qi]))

    info = {
        "model": model,
        "subset": subset,
        "topk": TOPK,
        "map_k": MAP_K,
        "n_queries": n_q,
        "n_subset_gallery": len(g_metas),
        "n_gallery": len(metas),
        "embed_dim": int(q_vecs.shape[1]),
        "gallery_digest": ids_digest(metas),
        "query_digest": ids_digest(q_metas),
        "dataset_revision": ds_cfg.get("revision"),
        "fine_mode": ds_cfg["fine_mode"],
        "source": source or {},
        "created": datetime.now().isoformat(timespec="seconds"),
    }

    out = path_of(model, subset)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        info=np.array(json.dumps(info, ensure_ascii=False)),
        rows=rows.astype(np.int32),
        scores=scores.astype(np.float16),
        **{f"hits_{c}": hits[c] for c in CRITERIA},
        **{f"n_rel_{c}": n_rel[c] for c in CRITERIA},
        **{f"ap_{c}": ap[c] for c in CRITERIA},
    )
    tmp.replace(out)                   # 중간에 죽어도 반쪽짜리 파일이 남지 않게
    return out


def load(model: str, subset: str) -> dict:
    with np.load(path_of(model, subset)) as z:
        data = {k: z[k] for k in z.files if k != "info"}
        data["info"] = json.loads(str(z["info"]))
    return data


def metrics(data: dict) -> dict:
    """서브셋 전체 지표. run_eval과 같은 정의다 (R@1, R@10, mAP@10)."""
    out = {}
    for c in CRITERIA:
        h = data[f"hits_{c}"]
        out[f"{c}_recall@1"] = float(h[:, :1].any(axis=1).mean())
        out[f"{c}_recall@10"] = float(h[:, :10].any(axis=1).mean())
        out[f"{c}_map@10"] = float(data[f"ap_{c}"].mean())
    return out
