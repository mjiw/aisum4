"""
검색 성능 지표 계산 (LookBench / SOP 공용)

지표 (상위 k개 기준, 쿼리별로 계산해 평균). R = gallery 전체에서 그 쿼리의 정답 수
    Recall@k    : 상위 k개 안에 정답이 하나라도 있으면 1 (DreamSim 논문 방식)
    Precision@k : 상위 k개 중 정답 비율
    mAP@k       : AP@k = sum(P(i) * rel(i)) / min(R, k)

정답 기준 (같은 공식에 기준만 바꿔 넣는다)
    item_ID - accumulate()
        라벨이 같은 gallery 항목. LookBench는 item_ID, SOP는 class_id.
        LookBench 공식 코드와 같은 기준이다. 정답이 gallery에 없는 쿼리는 제외한다.
    Coarse  - accumulate_attribute_metrics() (LookBench 전용)
        category가 같은 gallery 항목.
    Fine    - accumulate_attribute_metrics() (LookBench 전용)
        category, main_attribute, other_attributes가 모두 같은 gallery 항목.

    Coarse/Fine은 LookBench 논문과 공식 README의 정의를 따른다 (논문은 Recall@1만 보고한다).
    공식 레포 코드에는 구현이 없어서 문구대로 옮겼고, Precision/mAP는 같은 기준으로 확장한 자체 지표다.
    item_ID는 보지 않는다 (다른 상품이어도 속성이 같으면 정답).
    정답이 gallery에 하나도 없는 쿼리도 0점으로 포함한다. 빼면 어려운 쿼리가 빠져 점수가 부풀려진다.

    속성 비교 규칙 (논문에 세부 규칙이 없어서 정한 것)
      - 앞뒤 공백을 지우고 소문자로 맞춘 뒤 문자열로 비교한다. 동의어(bag / handbag)는 다른 값이다.
      - other_attributes는 쉼표로 나눈 집합으로 비교한다 (순서 무시).
      - fine_match="exact"  : main_attribute가 같고 other_attributes 집합이 같다 (기본값, README의 "all attributes to match")
      - fine_match="subset" : main_attribute가 같고 쿼리의 other_attributes가 gallery 항목에 모두 들어 있다

임베딩은 build_index.py가 L2 정규화해 두므로 내적이 곧 코사인 유사도다.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

QUERY_CHUNK = 256  # 유사도 행렬을 한 번에 만들지 않고 쿼리를 나눠 계산한다
FINE_MATCH_MODES = ("exact", "subset")
METRIC_KEYS = ("recall", "precision", "ap")
ATTRIBUTE_LEVELS = ("coarse", "fine")
# (metrics dict 키 접두사, 표에 찍을 이름, CSV 열 접두사)
LEVELS = (("", "item_ID", ""), ("Coarse ", "Coarse", "coarse_"), ("Fine ", "Fine", "fine_"))


# ---------------------------------------------------------------------------
# 입력 로드
# ---------------------------------------------------------------------------
def load_embeddings(artifact_dir: Path):
    """build_index.py 출력 폴더 -> (embeddings, ids)."""
    embeddings_path = artifact_dir / "embeddings.npy"
    ids_path = artifact_dir / "ids.json"
    for path in (embeddings_path, ids_path):
        if not path.is_file():
            raise FileNotFoundError(f"{path}가 없습니다 (build_index.py를 먼저 실행하세요)")

    embeddings = np.load(embeddings_path).astype(np.float32, copy=False)
    with open(ids_path, encoding="utf-8") as f:
        ids = json.load(f)
    if embeddings.shape[0] != len(ids):
        raise ValueError(f"{artifact_dir}: 임베딩 {embeddings.shape[0]}개 != id {len(ids)}개")
    return embeddings, ids


def load_meta_entries(meta_path: Path, ids: list) -> list:
    """meta.json에서 id 순서대로 항목(dict)을 뽑는다."""
    if not meta_path.is_file():
        raise FileNotFoundError(f"{meta_path}가 없습니다 (준비 스크립트를 먼저 실행하세요)")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    missing = [i for i in ids if i not in meta]
    if missing:
        raise ValueError(f"{meta_path}에 없는 id가 있습니다: {missing[:5]} (총 {len(missing)}개)")
    return [meta[i] for i in ids]


def load_labels(meta_path: Path, ids: list, field: str) -> list:
    """meta.json에서 id 순서대로 라벨을 뽑는다."""
    return [entry.get(field) for entry in load_meta_entries(meta_path, ids)]


# ---------------------------------------------------------------------------
# 속성 비교
# ---------------------------------------------------------------------------
def _normalize(value: str) -> str:
    return value.strip().lower()


def parse_attributes(entry):
    """meta.json 항목 -> (category, main_attribute, other_attributes 집합). category가 없으면 None."""
    if not entry or not entry.get("category"):
        return None
    main = entry.get("main_attribute")
    others = entry.get("other_attributes") or ""
    return (
        _normalize(entry["category"]),
        _normalize(main) if main else None,
        frozenset(_normalize(a) for a in others.split(",") if a.strip()),
    )


def is_fine_match(query_attr, gallery_attr, mode: str = "exact") -> bool:
    """category와 모든 속성이 일치하는지. 규칙은 모듈 docstring 참고."""
    q_category, q_main, q_others = query_attr
    g_category, g_main, g_others = gallery_attr
    if q_category != g_category or q_main != g_main:
        return False
    if mode == "exact":
        return q_others == g_others
    return q_others <= g_others


def _fine_relevant_counter(gallery_attrs: list, mode: str):
    """쿼리 속성 -> gallery 전체의 Fine 기준 정답 수를 돌려주는 함수. is_fine_match와 같은 규칙이다."""
    valid = [g for g in gallery_attrs if g is not None]
    if mode == "exact":
        # (category, main, others)가 통째로 같아야 하므로 튜플 자체를 키로 센다
        counts = Counter(valid)
        return lambda query_attr: counts.get(query_attr, 0)

    # subset은 포함 관계라 키로 셀 수 없다. category+main이 같은 후보끼리만 비교한다
    groups = defaultdict(list)
    for category, main, others in valid:
        groups[(category, main)].append(others)
    return lambda query_attr: sum(
        1 for others in groups.get((query_attr[0], query_attr[1]), ()) if query_attr[2] <= others
    )


# ---------------------------------------------------------------------------
# 검색 (상위 k개)
# ---------------------------------------------------------------------------
def iter_topk(query_emb, gallery_emb, max_k: int, self_index=None):
    """(쿼리 번호, 유사도 내림차순 상위 max_k개의 gallery 행 번호)를 하나씩 흘려보낸다.

    self_index를 주면 그 쿼리의 gallery 행을 결과에서 뺀다 (SOP처럼 query와 gallery가
    같은 집합일 때 자기 자신이 1위로 잡히는 것을 막는다). 값이 음수면 제외하지 않는다.
    """
    n_gallery = gallery_emb.shape[0]
    available = n_gallery - 1 if self_index is not None else n_gallery
    if max_k > available:
        raise ValueError(f"k={max_k}가 검색 가능한 gallery 크기({available})보다 큽니다")

    for start in range(0, query_emb.shape[0], QUERY_CHUNK):
        chunk = query_emb[start : start + QUERY_CHUNK]
        scores = chunk @ gallery_emb.T  # L2 정규화되어 있으므로 내적 = 코사인 유사도

        if self_index is not None:
            own = np.asarray(self_index[start : start + chunk.shape[0]])
            rows = np.arange(chunk.shape[0])
            valid = own >= 0
            scores[rows[valid], own[valid]] = -np.inf

        # 상위 max_k개만 필요하므로 전체 정렬 대신 부분 정렬
        top = np.argpartition(-scores, max_k - 1, axis=1)[:, :max_k]
        row_index = np.arange(top.shape[0])[:, None]
        top = top[row_index, np.argsort(-scores[row_index, top], axis=1)]

        for offset in range(chunk.shape[0]):
            yield start + offset, top[offset]


# ---------------------------------------------------------------------------
# 지표 합계
# ---------------------------------------------------------------------------
def _empty_totals(ks: list) -> dict:
    return {k: {key: 0.0 for key in METRIC_KEYS} for k in ks}


def _add_ranked_hits(totals: dict, hits, n_relevant: int, ks: list) -> None:
    """쿼리 하나의 상위 k개 정답 여부(hits)로 Recall/Precision/AP를 totals에 더한다.

    n_relevant(R)가 0이면 맞힐 정답이 없으므로 세 지표 모두 0이 더해진다.
    """
    for k in ks:
        hits_k = hits[:k]
        n_hit = int(hits_k.sum())
        totals[k]["recall"] += 1.0 if n_hit else 0.0
        totals[k]["precision"] += n_hit / k
        if n_hit and n_relevant > 0:
            # AP@k: 정답을 맞춘 위치마다 그 시점의 precision을 더해 min(R, k)로 나눈다
            ranks = np.flatnonzero(hits_k) + 1
            precisions = (np.arange(ranks.size) + 1) / ranks
            totals[k]["ap"] += precisions.sum() / min(n_relevant, k)


def merge_totals(totals_list: list, ks: list) -> dict:
    """여러 config의 합계를 더한다 (쿼리 수 가중평균용)."""
    return {k: {key: sum(t[k][key] for t in totals_list) for key in METRIC_KEYS} for k in ks}


def accumulate(query_emb, query_labels, gallery_emb, gallery_labels, ks: list, self_index=None):
    """item_ID(라벨) 기준 Recall/Precision/mAP의 쿼리별 합계.
    반환: (totals, 채점한 쿼리 수, 정답 없던 쿼리 수)

    평균이 아니라 합계를 돌려주는 이유는, 여러 벤치마크를 합칠 때 쿼리 단위로
    가중평균하기 위해서다.
    """
    gallery_labels = np.asarray(gallery_labels, dtype=object)
    # 쿼리마다 gallery 전체를 훑으면 60,000장 규모에서 너무 느리다. 라벨별 개수를 미리 센다.
    label_counts = Counter(label for label in gallery_labels.tolist() if label is not None)

    totals = _empty_totals(ks)
    n_scored = 0
    n_no_answer = 0

    for query_index, top in iter_topk(query_emb, gallery_emb, max(ks), self_index):
        label = query_labels[query_index]
        n_relevant = label_counts.get(label, 0) if label is not None else 0
        if self_index is not None and self_index[query_index] >= 0:
            n_relevant -= 1  # 자기 자신은 정답에서 뺀다
        if n_relevant <= 0:
            n_no_answer += 1
            continue

        n_scored += 1
        _add_ranked_hits(totals, gallery_labels[top] == label, n_relevant, ks)

    return totals, n_scored, n_no_answer


def accumulate_attribute_metrics(query_emb, query_attrs, gallery_emb, gallery_attrs, ks: list,
                                 fine_match: str = "exact"):
    """Coarse/Fine 기준 Recall/Precision/mAP의 쿼리별 합계.
    반환: (totals {"coarse": ..., "fine": ...}, 채점한 쿼리 수, 정답 없던 쿼리 수 {"coarse": n, "fine": n})

    query_attrs / gallery_attrs는 parse_attributes() 결과 리스트다. gallery 쪽 값이 None이면
    (distractor처럼 속성 정보가 없는 항목) 어떤 쿼리와도 일치하지 않는다.
    정답 수 R은 기준별로 gallery 전체에서 따로 세고, R=0인 쿼리도 0점으로 포함한다.
    """
    if fine_match not in FINE_MATCH_MODES:
        raise ValueError(f"fine_match는 {FINE_MATCH_MODES} 중 하나여야 합니다: {fine_match}")

    coarse_counts = Counter(g[0] for g in gallery_attrs if g is not None)
    count_fine = _fine_relevant_counter(gallery_attrs, fine_match)

    totals = {level: _empty_totals(ks) for level in ATTRIBUTE_LEVELS}
    no_answer = {level: 0 for level in ATTRIBUTE_LEVELS}
    n_scored = 0

    for query_index, top in iter_topk(query_emb, gallery_emb, max(ks)):
        query_attr = query_attrs[query_index]
        if query_attr is None:
            continue
        n_scored += 1

        top_attrs = [gallery_attrs[i] for i in top]
        coarse_hits = np.array([g is not None and g[0] == query_attr[0] for g in top_attrs], dtype=bool)
        fine_hits = np.array(
            [g is not None and is_fine_match(query_attr, g, fine_match) for g in top_attrs], dtype=bool
        )

        n_coarse = coarse_counts.get(query_attr[0], 0)
        n_fine = count_fine(query_attr)
        if n_coarse == 0:
            no_answer["coarse"] += 1
        if n_fine == 0:
            no_answer["fine"] += 1

        _add_ranked_hits(totals["coarse"], coarse_hits, n_coarse, ks)
        _add_ranked_hits(totals["fine"], fine_hits, n_fine, ks)

    return totals, n_scored, no_answer


# ---------------------------------------------------------------------------
# 평균 / 출력
# ---------------------------------------------------------------------------
def to_metrics(totals: dict, n_scored: int, ks: list, prefix: str = "") -> dict:
    """합계 -> 쿼리 평균. prefix는 "" (item_ID), "Coarse ", "Fine " 중 하나."""
    metrics = {}
    for k in ks:
        metrics[f"{prefix}Recall@{k}"] = totals[k]["recall"] / n_scored
        metrics[f"{prefix}Precision@{k}"] = totals[k]["precision"] / n_scored
        metrics[f"{prefix}mAP@{k}"] = totals[k]["ap"] / n_scored
    return metrics


def print_table(title: str, metrics: dict, ks: list) -> None:
    """정답 기준별로 k마다 한 줄씩 출력한다. 계산하지 않은 기준(SOP의 Coarse/Fine)은 건너뛴다."""
    print(f"\n{title}")
    print(f"  {'basis':<8}{'k':>4}  {'Recall@k':>10}  {'Precision@k':>12}  {'mAP@k':>8}")
    for prefix, name, _ in LEVELS:
        if f"{prefix}Recall@{ks[0]}" not in metrics:
            continue
        for index, k in enumerate(ks):
            label = name if index == 0 else ""
            print(f"  {label:<8}{k:>4}  {metrics[f'{prefix}Recall@{k}']:>10.4f}  "
                  f"{metrics[f'{prefix}Precision@{k}']:>12.4f}  {metrics[f'{prefix}mAP@{k}']:>8.4f}")


def write_csv(out_csv: Path, results: list, model: str, ks: list) -> None:
    """results: {"config", "dim", "metrics", "n_query", "n_gallery", "n_distractor"} 리스트.
    Coarse/Fine 관련 값이 없는 결과(SOP)는 해당 열을 비워 둔다."""
    def fmt(value):
        return f"{value:.6f}" if value is not None else ""

    def blank_if_none(value):
        return "" if value is None else value

    metric_columns = [f"{column}{name}_at_k" for _, _, column in LEVELS for name in ("recall", "precision", "map")]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "model", "dim", "n_query", "n_query_attr", "n_gallery", "n_distractor",
                         "n_coarse_no_answer", "n_fine_no_answer", "k"] + metric_columns)
        for result in results:
            metrics = result["metrics"]
            no_answer = result.get("attr_no_answer") or {}
            for k in ks:
                row = [
                    result["config"], model, result["dim"], result["n_query"],
                    blank_if_none(result.get("n_attr")), result["n_gallery"], result.get("n_distractor", 0),
                    blank_if_none(no_answer.get("coarse")), blank_if_none(no_answer.get("fine")), k,
                ]
                for prefix, _, _ in LEVELS:
                    row += [fmt(metrics.get(f"{prefix}Recall@{k}")),
                            fmt(metrics.get(f"{prefix}Precision@{k}")),
                            fmt(metrics.get(f"{prefix}mAP@{k}"))]
                writer.writerow(row)
    print(f"\n저장: {out_csv}")
