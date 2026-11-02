"""뷰어를 정적 사이트로 내보낸다 (GitHub Pages용).

Streamlit 뷰어는 서버가 필요하다. 연구실 밖에서 보여주려면 PC를 켜 두고 방화벽·
네트워크까지 맞춰야 하는데, 교내망에서는 그게 잘 안 된다. 그래서 **미리 뽑은
쿼리들의 검색 결과를 통째로 파일로 구워** 정적 페이지로 띄운다. 서버가 없으니
링크만 주면 누구나 본다.

    python -m benchmark.viewer.export_static                 # 기본값으로 docs/ 생성
    python -m benchmark.viewer.export_static --queries 80    # 서브셋당 쿼리 수

쿼리 선택은 기본이 `disagree`다. 모델끼리 결과가 갈리는 쿼리를 우선 뽑는다 —
다 같은 걸 찾아오는 쿼리를 200개 넣어봐야 비교에 쓸모가 없다.

썸네일은 (출처 config, 행 번호)로 이름을 지어 모델·쿼리·서브셋 간에 공유한다.
noise 풀 이미지는 어느 서브셋에서든 같은 파일을 쓴다.
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from benchmark.viewer import core

ROOT = Path(__file__).resolve().parents[2]
WEB = Path(__file__).resolve().parent / "web"       # index.html, app.js, style.css
CRITERIA = ("coarse", "fine")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs", help="출력 폴더 (저장소 기준 상대경로)")
    p.add_argument("--queries", type=int, default=50, help="서브셋당 쿼리 수")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--thumb", type=int, default=160, help="썸네일 한 변 픽셀")
    p.add_argument("--quality", type=int, default=72, help="JPEG 품질")
    p.add_argument("--select", choices=["mixed", "disagree", "random", "first"],
                   default="mixed",
                   help="쿼리 고르는 기준 (mixed: 절반은 모델 간 결과가 갈리는 쿼리, "
                        "절반은 무작위)")
    p.add_argument("--subsets", default=None, help="쉼표로 구분. 기본은 전체")
    p.add_argument("--models", default=None, help="쉼표로 구분. 기본은 캐시가 있는 전체")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def pick_queries(config: str, models: list, n: int, topk: int, how: str, seed: int) -> list:
    """보여줄 쿼리 인덱스.

    disagree는 모델 간 top-K가 덜 겹치는 쿼리를 고른다. 비교할 거리가 많지만
    어려운 쿼리만 모여 전반적인 성능이 실제보다 나빠 보인다. 그래서 기본값
    mixed는 절반만 그렇게 고르고 나머지는 무작위로 채워 균형을 맞춘다.
    """
    _, metas = core.load_queries(models[0], config)
    total = len(metas)
    if n >= total:
        return list(range(total))
    if how == "first":
        return list(range(n))
    rng = np.random.default_rng(seed)
    if how == "random":
        return sorted(rng.choice(total, size=n, replace=False).tolist())
    if how == "mixed":
        hard = pick_queries(config, models, n // 2, topk, "disagree", seed)
        rest = [i for i in range(total) if i not in set(hard)]
        extra = rng.choice(len(rest), size=min(n - len(hard), len(rest)), replace=False)
        return sorted(hard + [rest[i] for i in extra])

    # 모델별 top-K를 한 번에 구해 겹침을 센다. 겹침이 적을수록 비교할 거리가 많다.
    tops = {}
    for model in models:
        q_vecs, _ = core.load_queries(model, config)
        g_vecs, _, _ = core.load_gallery(model, config)
        sims = q_vecs @ g_vecs.T                        # (Q, N)
        part = np.argpartition(-sims, topk - 1, axis=1)[:, :topk]
        tops[model] = part

    scores = []
    for qi in range(total):
        counts = Counter()
        for model in models:
            counts.update(tops[model][qi].tolist())
        # 모든 모델이 같은 걸 찾으면 len(models)짜리 항목만 나온다 → 겹침 점수 최대
        overlap = sum(c for c in counts.values() if c > 1) / (topk * len(models))
        scores.append((overlap, qi))
    scores.sort()                                       # 겹침 적은 순
    return sorted(qi for _, qi in scores[:n])


def collect(config: str, models: list, q_indices: list, topk: int) -> tuple:
    """쿼리별 결과와, 그 결과에 등장한 갤러리 행 목록을 만든다."""
    index = core.relevance_index(config)
    _, g_metas, n_subset = core.load_gallery(models[0], config)
    _, q_metas = core.load_queries(models[0], config)

    queries, used_rows = [], set()
    for qi in q_indices:
        q_meta = q_metas[qi]
        entry = {
            "idx": qi,
            "category": q_meta["category"],
            "attrs": q_meta["attrs"],
            "n_relevant": {c: index.count_relevant(q_meta, c) for c in CRITERIA},
            "results": {},
        }
        for model in models:
            rows, scores = core.search_one(model, config, qi, topk)
            hits = {c: core.judge_hits(config, q_meta, rows, c) for c in CRITERIA}
            entry["results"][model] = {
                "rows": [int(r) for r in rows],
                "scores": [round(float(s), 4) for s in scores],
                "hits": {c: [bool(h) for h in hits[c]] for c in CRITERIA},
                "metrics": {c: {k: round(v, 4) for k, v in
                                core.query_scores(config, q_meta, hits[c], c).items()}
                            for c in CRITERIA},
            }
            used_rows.update(int(r) for r in rows)
        queries.append(entry)

    gallery = {str(r): {"label": "noise" if r >= n_subset else g_metas[r]["category"]}
               for r in sorted(used_rows)}
    return queries, gallery, n_subset, len(g_metas)


def save_thumb(img, path: Path, size: int, quality: int):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    img = img.convert("RGB")
    img.thumbnail((size, size))
    img.save(path, format="JPEG", quality=quality, optimize=True)


def export_images(config: str, queries: list, gallery: dict, n_subset: int,
                  out: Path, size: int, quality: int):
    """쿼리·갤러리 썸네일을 굽는다. 갤러리 파일명은 출처 기준이라 서로 공유된다."""
    for entry in queries:
        save_thumb(core.query_image(config, entry["idx"]),
                   out / "img" / config / f"q{entry['idx']}.jpg", size, quality)
        entry["img"] = f"img/{config}/q{entry['idx']}.jpg"

    for row_s in gallery:
        row = int(row_s)
        # noise 풀 이미지는 서브셋이 달라도 같은 파일 하나만 있으면 된다
        src, name = ((config, f"g{row}") if row < n_subset
                     else ("noise", f"g{row - n_subset}"))
        rel = f"img/{src}/{name}.jpg"
        save_thumb(core.gallery_image(config, row, n_subset),
                   out / rel, size, quality)
        gallery[row_s]["img"] = rel


def build_summary(models: list, subsets: list) -> list:
    """모델별 정확도(서브셋 전체 평균) + 속도 + 메모리."""
    rows = core.perf_table(models)
    for row in rows:
        # 모델마다 돌린 서브셋이 다르다(브랜치에서 가져와 한 서브셋만 돌린 경우 등).
        # 캐시가 있는 서브셋만 계산하고 나머지는 비워 둔다 — 사이트에서 '—'로 나온다.
        row["accuracy"] = {
            config: dict(core.eval_metrics(row["model"], config))
            for config in subsets
            if row["model"] in core.available_models(config)
        }
        row.pop("runs_sec", None)            # 사이트에서 안 쓰는 큰 값은 뺀다
    return rows


def main():
    args = parse_args()
    out = (ROOT / args.out).resolve()
    subsets = ([s.strip() for s in args.subsets.split(",")] if args.subsets
               else core.subsets())

    manifest = {"subsets": [], "generated_with": {"queries": args.queries,
                                                  "topk": args.topk,
                                                  "select": args.select}}
    all_models = []
    for config in subsets:
        models = ([m.strip() for m in args.models.split(",")] if args.models
                  else core.available_models(config))
        if not models:
            print(f"[skip] {config}: 임베딩 캐시가 없습니다")
            continue

        print(f"[{config}] 모델 {len(models)}개, 쿼리 고르는 중 ({args.select})")
        q_indices = pick_queries(config, models, args.queries, args.topk,
                                 args.select, args.seed)
        queries, gallery, n_subset, gallery_size = collect(
            config, models, q_indices, args.topk)
        print(f"[{config}] 쿼리 {len(queries)}개, 썸네일 {len(gallery) + len(queries)}장 굽는 중")
        export_images(config, queries, gallery, n_subset, out, args.thumb, args.quality)

        data = {"config": config, "models": models, "topk": args.topk,
                "gallery_size": gallery_size, "queries": queries, "gallery": gallery}
        (out / "data").mkdir(parents=True, exist_ok=True)
        with open(out / "data" / f"{config}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        manifest["subsets"].append({"config": config, "n_queries": len(queries),
                                    "models": models})
        all_models += [m for m in models if m not in all_models]

    if not all_models:
        raise SystemExit("내보낼 모델이 없습니다. run_eval을 먼저 돌리세요.")

    manifest["models"] = [{"key": m, "label": core.label_of(m),
                           "baseline": m == core.BASELINE} for m in all_models]
    manifest["summary"] = build_summary(all_models, [s["config"] for s in manifest["subsets"]])
    with open(out / "data" / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    for name in ("index.html", "app.js", "style.css"):
        shutil.copy(WEB / name, out / name)
    (out / ".nojekyll").write_text("", encoding="utf-8")     # _ 로 시작하는 파일 보호

    size_mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1024**2
    print(f"\n[done] {out} ({size_mb:.1f}MB)")
    print("로컬 확인:  python -m http.server -d docs 8000")


if __name__ == "__main__":
    main()
