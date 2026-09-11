"""이미지 → 임베딩. 모델·config·split별로 .npy에 캐시한다.

갤러리 6만 장 인코딩이 가장 비싼 단계라 한 번 돌리면 재사용한다.
metric을 고치고 다시 돌릴 때는 이 단계를 건너뛴다(run_eval --skip-encode).

이 머신은 GPU 드라이버 크래시가 반복되는 이력이 있어, 긴 인코딩을 통째로 잃지
않도록 checkpoint_every 장마다 부분 결과를 디스크에 떨군다. 다시 실행하면 남아
있는 부분 결과 뒤부터 이어서 인코딩한다.
"""

import json
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm


def cache_paths(cache_dir, model_name: str, config: str, split: str):
    base = Path(cache_dir) / f"{model_name}__{config}__{split}"
    return base.with_suffix(".npy"), base.with_suffix(".meta.json")


def _validate(vectors, metas, expect_dim=None, expect_rows=None):
    """캐시가 실제로 쓸 수 있는 값인지 확인한다.

    크래시로 재부팅되면 파일 크기만 할당된 채 내용이 전부 0인 캐시가 남을 수 있다
    (실제로 겪었다). 그대로 읽으면 조용히 엉터리 지표가 나오므로 여기서 걸러낸다.
    임베딩은 base.embed()에서 L2 정규화되므로 norm이 1이 아니면 깨진 것이다.
    """
    if vectors.shape[0] != len(metas):
        return f"벡터 {vectors.shape[0]}행 != 메타 {len(metas)}건"
    # --limit으로 만든 일부 캐시가 전체 실행에 재사용되면 조용히 잘못된 결과가 나온다
    if expect_rows is not None and vectors.shape[0] != expect_rows:
        return f"캐시 {vectors.shape[0]}행 != 데이터셋 {expect_rows}행"
    if expect_dim is not None and vectors.shape[0] and vectors.shape[1] != expect_dim:
        return f"차원 {vectors.shape[1]} != 모델 차원 {expect_dim}"
    if vectors.shape[0] == 0:
        return None
    if not np.isfinite(vectors).all():
        return "NaN/Inf 포함"
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-2):
        return f"L2 norm이 1이 아님 (min={norms.min():.4f}, max={norms.max():.4f})"
    return None


def _read_meta(path):
    """깨진 JSON은 예외 대신 None으로 돌려준다."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None


def _part_dir(cache_dir, model_name, config, split):
    return Path(cache_dir) / f"{model_name}__{config}__{split}.parts"


def _load_parts(part_dir):
    """저장된 부분 결과를 순서대로 이어붙여 돌려준다."""
    if not part_dir.exists():
        return [], []
    vectors, metas = [], []
    for vec_path in sorted(part_dir.glob("part_*.npy"), key=lambda p: int(p.stem.split("_")[1])):
        meta_path = vec_path.with_suffix(".json")
        if not meta_path.exists():
            break  # 벡터만 있고 메타가 없으면 저장 중 끊긴 것이므로 버린다
        try:
            chunk = np.load(vec_path)
        except (ValueError, OSError):
            break
        chunk_meta = _read_meta(meta_path)
        if chunk_meta is None or _validate(chunk, chunk_meta):
            break  # 크래시로 내용이 0으로 남은 부분 결과는 여기서 끊는다
        vectors.append(chunk)
        metas.extend(chunk_meta)
    return vectors, metas


def _save_part(part_dir, index, vectors, metas):
    part_dir.mkdir(parents=True, exist_ok=True)
    base = part_dir / f"part_{index:05d}"
    np.save(base.with_suffix(".npy"), vectors)
    # 메타를 나중에 쓰는 게 중요하다. _load_parts가 메타 유무로 완결성을 판단한다.
    with open(base.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False)


def encode_split(model, ds, to_record, batch_size, desc, part_dir=None,
                 checkpoint_every=4096, start=0, part_index=0):
    """(N, D) float32 임베딩과 길이 N의 메타 리스트. 순서는 ds와 같다."""
    vectors, metas = [], []
    pending_vecs, pending_metas = [], []
    buffer = []

    def flush_batch():
        if buffer:
            pending_vecs.append(model.embed(buffer))
            buffer.clear()

    def checkpoint():
        nonlocal part_index
        if part_dir is None or not pending_vecs:
            return
        block = np.concatenate(pending_vecs, axis=0)
        _save_part(part_dir, part_index, block, pending_metas)
        vectors.append(block)
        metas.extend(pending_metas)
        pending_vecs.clear()
        pending_metas.clear()
        part_index += 1

    total = len(ds) - start
    for row in tqdm(ds.select(range(start, len(ds))), total=total, desc=desc):
        rec = to_record(row)
        buffer.append(rec.image)
        pending_metas.append(
            {"item_id": rec.item_id, "category": rec.category, "attrs": rec.attrs}
        )
        if len(buffer) == batch_size:
            flush_batch()
        if len(pending_metas) >= checkpoint_every and not buffer:
            checkpoint()

    flush_batch()
    if part_dir is not None:
        checkpoint()
    else:
        vectors, metas = pending_vecs, pending_metas

    if not vectors:
        return np.empty((0, model.embed_dim), dtype=np.float32), []
    return np.concatenate(vectors, axis=0), metas


def encode_or_load(model, ds, to_record, batch_size, cache_dir, config, split,
                   force=False, checkpoint_every=4096):
    """캐시가 있으면 읽고, 없으면 인코딩해서 저장한다. 중단된 작업은 이어서 한다."""
    vec_path, meta_path = cache_paths(cache_dir, model.name, config, split)
    part_dir = _part_dir(cache_dir, model.name, config, split)

    if force:
        shutil.rmtree(part_dir, ignore_errors=True)
    elif vec_path.exists() and meta_path.exists():
        try:
            cached_vecs = np.load(vec_path)
        except (ValueError, OSError) as exc:
            cached_vecs, problem = None, f"npy 로드 실패 ({exc})"
        else:
            cached_metas = _read_meta(meta_path)
            problem = ("메타 JSON 손상" if cached_metas is None
                       else _validate(cached_vecs, cached_metas, model.embed_dim,
                                      expect_rows=len(ds)))
        if problem is None:
            print(f"[encode] 캐시 사용: {vec_path}")
            return cached_vecs, cached_metas
        print(f"[encode] 캐시가 손상돼 폐기하고 다시 인코딩합니다: {problem}")
        vec_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        shutil.rmtree(part_dir, ignore_errors=True)

    done_vecs, done_metas = ([], []) if force else _load_parts(part_dir)
    if len(done_metas) > len(ds):
        print(f"[encode] 부분 결과({len(done_metas)}행)가 데이터셋({len(ds)}행)보다 커서 폐기합니다")
        shutil.rmtree(part_dir, ignore_errors=True)
        done_vecs, done_metas = [], []
    start = len(done_metas)
    if start:
        print(f"[encode] 이어서 진행: {start}/{len(ds)}장 완료됨 ({part_dir.name})")

    vec_path.parent.mkdir(parents=True, exist_ok=True)
    new_vecs, new_metas = encode_split(
        model, ds, to_record, batch_size, f"{config}/{split}",
        part_dir=part_dir, checkpoint_every=checkpoint_every,
        start=start, part_index=len(done_vecs),
    )

    vectors = np.concatenate(done_vecs + [new_vecs], axis=0) if done_vecs else new_vecs
    metas = done_metas + new_metas

    np.save(vec_path, vectors)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False)
    shutil.rmtree(part_dir, ignore_errors=True)   # 최종본이 생겼으니 부분 결과는 정리
    print(f"[encode] 저장: {vec_path} {vectors.shape}")
    return vectors, metas
