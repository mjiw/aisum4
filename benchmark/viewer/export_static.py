"""비교 뷰어를 정적 사이트로 굽는다 (서버 없이 링크만으로 열리는 페이지).

    python -m benchmark.viewer.export_static                  # site/ 생성
    python -m benchmark.viewer.export_static --queries 80     # 서브셋당 쿼리 수
    python -m benchmark.viewer.deploy                         # 만든 site/를 배포

입력은 GPU 워커가 남긴 결과물뿐이다 (GPU·임베딩 캐시 불필요).

    results/retrieval/{model}__lookbench-{subset}.npz   쿼리별 top-10, 정답 여부, AP
    results/perf__{model}.json                          속도·메모리·토큰
    LookBench 데이터셋 (로컬 HF 캐시)                     썸네일과 쿼리 정보

모델 목록·순서·링크는 benchmark/compute/lineup.json을 따른다. 결과물이 없는
모델은 표에 '미측정'으로 남고 검색 결과 화면에서는 빠진다.

정확도 요약은 **서브셋의 전체 쿼리**로 계산한다. 이미지까지 보여주는 쿼리는
서브셋당 --queries개만 고른다(전부 넣으면 썸네일이 수만 장이 된다). 고르는 기준은
기본이 mixed — 절반은 모델끼리 결과가 크게 갈리는 쿼리, 절반은 무작위다.
"""

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from benchmark.compute import REPO, artifacts, load_lineup, results_dir

WEB = Path(__file__).resolve().parent / "web"
CRITERIA = artifacts.CRITERIA


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="site", help="출력 폴더 (저장소 기준)")
    p.add_argument("--queries", type=int, default=50, help="서브셋당 이미지로 보여줄 쿼리 수")
    p.add_argument("--select", choices=["mixed", "disagree", "random"], default="mixed")
    p.add_argument("--thumb", type=int, default=220, help="갤러리 썸네일 긴 변 픽셀")
    p.add_argument("--query-px", type=int, default=440, help="쿼리 이미지 긴 변 픽셀")
    p.add_argument("--quality", type=int, default=74, help="WebP 품질")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ------------------------------------------------------------------ 데이터셋

_SPLITS = {}


def _split(config: str, split: str):
    """(hf_dataset, to_record, cols). 같은 split을 여러 번 열지 않게 캐시한다."""
    if (config, split) not in _SPLITS:
        from benchmark.data import lookbench

        cfg = json.loads((REPO / "benchmark" / "config.json").read_text(encoding="utf-8"))
        ds_cfg = cfg["datasets"]["lookbench"]
        cache = Path(cfg["data"]["hf_cache_dir"])
        cache = cache if cache.is_absolute() else REPO / cache
        _SPLITS[(config, split)] = lookbench.load_split(
            ds_cfg["repo_id"], config, split, cache_dir=str(cache),
            revision=ds_cfg.get("revision"))
    return _SPLITS[(config, split)]


def dataset_gallery_digest(subset: str) -> str:
    """데이터셋에서 바로 계산한 갤러리 item_id 순서의 지문 (이미지는 읽지 않는다).

    결과물의 행 번호가 정말 이 데이터셋의 같은 이미지를 가리키는지 확인하는 용도다.
    어긋나면 썸네일이 조용히 엉뚱한 상품을 보여주므로 빌드를 멈춘다.
    item_id 규칙은 lookbench.load_split과 같다 ('{config}:{item_ID}').
    """
    h = hashlib.sha1()
    for config in (subset, "noise"):
        ds, _, cols = _split(config, "gallery")
        for value in ds.select_columns([cols["item_id"]])[cols["item_id"]]:
            h.update(f"{config}:{value}".encode("utf-8"))
            h.update(b"\n")
    return h.hexdigest()


def gallery_source(row: int, subset: str, n_subset: int):
    """갤러리 행 번호 → (config, 그 config 안의 행 번호)."""
    return (subset, row) if row < n_subset else ("noise", row - n_subset)


def save_image(img, path: Path, size: int, quality: int):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    img = img.convert("RGB")
    img.thumbnail((size, size))
    img.save(path, format="WEBP", quality=quality, method=5)


# ------------------------------------------------------------------ 결과물

def load_results(lineup: dict) -> dict:
    """{subset: {model: artifact}}. 갤러리·쿼리 순서가 모델끼리 같은지 확인한다."""
    out = {}
    for subset in lineup["subsets"]:
        found = {}
        for spec in lineup["models"]:
            if artifacts.exists(spec["key"], subset):
                found[spec["key"]] = artifacts.load(spec["key"], subset)
        if not found:
            continue
        digests = {(a["info"]["gallery_digest"], a["info"]["query_digest"]) for a in found.values()}
        if len(digests) != 1:
            detail = {k: a["info"]["gallery_digest"][:8] for k, a in found.items()}
            raise SystemExit(f"[{subset}] 모델마다 갤러리 순서가 다릅니다: {detail}")
        out[subset] = found
    return out


def load_perf(key: str) -> dict:
    path = results_dir() / f"perf__{key}.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        perf = json.load(f)
    # 지연·토큰이 없는 예전 측정은 조건이 달라 섞지 않는다 (worker.perf_ready와 같은 기준)
    if "latency_ms_b1" not in perf:
        return {}
    keep = ("sec_per_1000", "images_per_sec", "latency_ms_b1", "preprocess_sec_per_1000",
            "gpu_sec_per_1000", "peak_vram_mb", "model_params", "model_size_mb",
            "embed_dim", "gallery_60k_mb", "vision_tokens_per_image",
            "llm_tokens_per_image", "load_sec", "batch_size", "input_shape", "gpu",
            "n_images", "config", "torch", "transformers", "timestamp")
    return {k: perf.get(k) for k in keep if k in perf}


def _stats(per_query: list) -> dict:
    """여러 서브셋의 쿼리별 값을 이어붙여 전체 지표를 낸다 (= 쿼리 수 가중 평균)."""
    out = {"n_queries": int(sum(len(p["ap_coarse"]) for p in per_query))}
    for c in CRITERIA:
        hits = np.concatenate([p[f"hits_{c}"] for p in per_query])
        ap = np.concatenate([p[f"ap_{c}"] for p in per_query])
        out[c] = {"recall@1": float(hits[:, :1].any(axis=1).mean()),
                  "recall@10": float(hits[:, :10].any(axis=1).mean()),
                  "map@10": float(ap.mean())}
    return out


def _versus(per_query: list, base: list) -> dict:
    """기준 모델 대비 쿼리별 승·무·패 (AP@10 기준)."""
    out = {}
    for c in CRITERIA:
        ap = np.concatenate([p[f"ap_{c}"] for p in per_query]).astype(np.float64)
        ap_b = np.concatenate([b[f"ap_{c}"] for b in base]).astype(np.float64)
        diff = ap - ap_b
        out[c] = {"win": int((diff > 1e-6).sum()), "tie": int((np.abs(diff) <= 1e-6).sum()),
                  "loss": int((diff < -1e-6).sum())}
    return out


def build_summary(lineup: dict, results: dict) -> dict:
    """{'all' | subset: {model: {n_queries, coarse:{...}, fine:{...}, vs:{...}}}}."""
    baseline = lineup["baseline"]
    subsets = list(results)
    summary = {}
    for scope in ["all", *subsets]:
        scope_subsets = subsets if scope == "all" else [scope]
        rows = {}
        for spec in lineup["models"]:
            key = spec["key"]
            # '전체'는 네 서브셋이 다 있을 때만 낸다. 일부만 있으면 쿼리 구성이 달라 비교가 안 된다.
            if not all(key in results[s] for s in scope_subsets):
                continue
            per_query = [results[s][key] for s in scope_subsets]
            rows[key] = _stats(per_query)
            if key != baseline and all(baseline in results[s] for s in scope_subsets):
                rows[key]["vs"] = _versus(per_query, [results[s][baseline] for s in scope_subsets])
        summary[scope] = rows
    return summary


# ------------------------------------------------------------------ 쿼리 고르기

def pick_queries(found: dict, baseline: str, n: int, how: str, rng) -> list:
    """이미지로 보여줄 쿼리 인덱스.

    disagree는 기준 모델의 top-10과 덜 겹치고, 모델 간 정답 여부가 갈리는 쿼리다.
    그런 쿼리만 모으면 어려운 쿼리 위주라 전체 인상이 나빠 보이므로 기본값
    mixed는 절반만 그렇게 고르고 나머지는 무작위로 채운다.
    """
    total = next(iter(found.values()))["rows"].shape[0]
    if n >= total:
        return list(range(total))
    if how == "random":
        return sorted(rng.choice(total, size=n, replace=False).tolist())

    base = found.get(baseline) or next(iter(found.values()))
    others = [a for k, a in found.items() if a is not base] or [base]
    overlap = np.zeros(total)
    for a in others:
        same = (a["rows"][:, :, None] == base["rows"][:, None, :]).any(axis=2)
        overlap += same.mean(axis=1)
    top1 = np.stack([a["hits_coarse"][:, 0] for a in found.values()] +
                    [a["hits_fine"][:, 0] for a in found.values()], axis=1)
    spread = top1.std(axis=1)                       # 모델끼리 1위 정답 여부가 갈리면 큼
    score = overlap / len(others) - spread          # 작을수록 비교할 거리가 많다
    order = np.argsort(score, kind="stable")

    if how == "disagree":
        return sorted(order[:n].tolist())
    hard = order[: n // 2].tolist()
    rest = np.setdiff1d(np.arange(total), hard)
    extra = rng.choice(rest, size=n - len(hard), replace=False).tolist()
    return sorted(hard + extra)


# ------------------------------------------------------------------ 굽기

def export_subset(subset: str, found: dict, lineup: dict, args, out: Path, rng) -> dict:
    baseline = lineup["baseline"]
    info = next(iter(found.values()))["info"]
    n_subset = info["n_subset_gallery"]

    digest = dataset_gallery_digest(subset)
    if digest != info["gallery_digest"]:
        raise SystemExit(f"[{subset}] 결과물의 갤러리 순서가 로컬 데이터셋과 다릅니다. "
                         "데이터셋 revision을 확인하세요.")

    picked = pick_queries(found, baseline, args.queries, args.select, rng)
    q_ds, q_rec, _ = _split(subset, "query")
    if len(q_ds) != info["n_queries"]:
        raise SystemExit(f"[{subset}] 쿼리 수가 다릅니다: 결과물 {info['n_queries']}, "
                         f"데이터셋 {len(q_ds)}")

    queries, used = [], set()
    for qi in picked:
        rec = q_rec(q_ds[qi])
        rel = f"img/{subset}/q{qi}.webp"
        save_image(rec.image, out / rel, args.query_px, args.quality)
        entry = {"idx": qi, "img": rel, "cat": rec.category, "attrs": rec.attrs,
                 "n_rel": {c: int(next(iter(found.values()))[f"n_rel_{c}"][qi]) for c in CRITERIA},
                 "res": {}}
        for key, a in found.items():
            rows = a["rows"][qi].tolist()
            entry["res"][key] = {
                "rows": rows,
                "scores": [round(float(s), 3) for s in a["scores"][qi]],
                "hits": {c: "".join("1" if h else "0" for h in a[f"hits_{c}"][qi]) for c in CRITERIA},
                "ap": {c: round(float(a[f"ap_{c}"][qi]), 4) for c in CRITERIA},
            }
            used.update(rows)
        queries.append(entry)

    gallery = {}
    for row in sorted(used):
        config, local = gallery_source(row, subset, n_subset)
        ds, rec_fn, _ = _split(config, "gallery")
        rec = rec_fn(ds[local])
        rel = f"img/{config}/g{local}.webp"            # noise 이미지는 서브셋끼리 공유
        save_image(rec.image, out / rel, args.thumb, args.quality)
        gallery[str(row)] = {"img": rel, "cat": "noise" if config == "noise" else rec.category,
                             "attrs": rec.attrs[:6]}

    print(f"[{subset}] 쿼리 {len(queries)}개 / 갤러리 썸네일 {len(gallery)}장 / 모델 {len(found)}개")
    return {"key": subset, "n_queries": info["n_queries"], "n_gallery": info["n_gallery"],
            "models": list(found), "queries": queries, "gallery": gallery}


def main():
    args = parse_args()
    out = (REPO / args.out).resolve()
    if REPO not in out.parents:
        raise SystemExit(f"출력 폴더는 저장소 안이어야 합니다: {out}")
    lineup = load_lineup()
    results = load_results(lineup)
    if not results:
        raise SystemExit("결과물이 하나도 없습니다. python -m benchmark.compute run 을 먼저 돌리세요.")

    # 빌드 결과물 폴더라 매번 새로 만든다. 지난 빌드의 썸네일(다른 크기, 이제 안 쓰는
    # 쿼리)이 섞여 배포 용량만 늘어나는 걸 막는다.
    shutil.rmtree(out, ignore_errors=True)

    rng = np.random.default_rng(args.seed)
    subsets = {s: export_subset(s, found, lineup, args, out, rng) for s, found in results.items()}

    first = next(iter(next(iter(results.values())).values()))["info"]
    models = []
    for spec in lineup["models"]:
        key = spec["key"]
        src = next((results[s][key]["info"].get("source", {})
                    for s in results if key in results[s]), {})
        models.append({
            "key": key, "label": spec["label"], "note": spec.get("note", ""),
            "links": spec.get("links", {}), "branch": spec["branch"],
            "commit": (src.get("commit") or "")[:7],
            "baseline": key == lineup["baseline"],
            "subsets": [s for s in results if key in results[s]],
            "perf": load_perf(key),
        })

    bundle = {
        "meta": {
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "baseline": lineup["baseline"],
            "dataset": {"name": "LookBench", "revision": first.get("dataset_revision"),
                        "url": "https://huggingface.co/datasets/srpone/look-bench"},
            "topk": first["topk"],
            "queries_per_subset": args.queries,
            "select": args.select,
        },
        "models": models,
        "summary": build_summary(lineup, results),
        "subsets": subsets,
    }

    # JSON이 아니라 JS로 싣는다. fetch()는 file://에서 막혀, 폴더째 받아 index.html을
    # 더블클릭해도 열리게 하려면 <script>로 읽히는 형태여야 한다.
    (out / "data").mkdir(parents=True, exist_ok=True)
    payload = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    (out / "data" / "bundle.js").write_text(f"window.VIEWER_DATA={payload};\n", encoding="utf-8")
    for name in ("index.html", "app.js", "style.css"):
        shutil.copy(WEB / name, out / name)
    (out / ".nojekyll").write_text("", encoding="utf-8")

    size_mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1024**2
    print(f"\n[done] {out} ({size_mb:.1f}MB)")
    print(f"로컬 확인: {out / 'index.html'} 를 브라우저로 열기 (서버 불필요)")


if __name__ == "__main__":
    main()
