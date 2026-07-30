"""
이미지 폴더 -> 임베딩 (embeddings.npy / ids.json / meta.json) 생성.
사용법 (repo 루트에서):
    python -m embedding.build_index --model dreamsim
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .image_loader import CroppedImageLoader
from .model_loader import available_models, create_model


def build_index(cfg: dict, model_name: str, files: list | None = None) -> Path:
    loader = CroppedImageLoader(cfg["data"]["image_folder"], files=files)
    if len(loader) == 0:
        raise RuntimeError(f"이미지가 한 장도 없습니다: {cfg['data']['image_folder']}")

    model_cfg = cfg.get("models", {}).get(model_name)
    if model_cfg is None:
        raise ValueError(f"config의 models 섹션에 '{model_name}' 설정이 없습니다")
    batch_size = model_cfg["batch_size"]
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"{model_name}.batch_size는 양의 정수여야 합니다: {batch_size}")

    model = create_model(model_name, cfg)

    # 전체 크기로 선할당하고 커서로 채운 뒤, 깨진 파일 수만큼 끝에서 잘라낸다
    embeddings = np.empty((len(loader), model.embed_dim), dtype=np.float32)
    ids = []
    cursor = 0

    n_batches = (len(loader) + batch_size - 1) // batch_size
    for images, batch_ids in tqdm(
        loader.iter_batches(batch_size), total=n_batches, desc=f"Embedding ({model_name})"
    ):
        embeddings[cursor : cursor + len(batch_ids)] = model.embed(images)
        ids.extend(batch_ids)
        cursor += len(batch_ids)

    if cursor == 0:
        raise RuntimeError(
            "정상적으로 임베딩한 이미지가 한 장도 없습니다 "
            f"(깨진 파일 {len(loader.broken_files)}개)"
        )

    embeddings = embeddings[:cursor]
    # 행-id 대응은 아티팩트 계약의 핵심이므로 -O 실행에서도 항상 검사한다
    if embeddings.shape[0] != len(ids):
        raise RuntimeError(
            "embeddings 행 수와 ids 길이가 일치하지 않습니다: "
            f"{embeddings.shape[0]} != {len(ids)}"
        )
    if len(set(ids)) != len(ids):
        dups = [i for i, c in Counter(ids).items() if c > 1]
        raise RuntimeError(f"id가 중복됩니다: {dups[:10]}")

    out_dir = Path(cfg["data"]["output_dir"]) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "embeddings.npy", embeddings)
    with open(out_dir / "ids.json", "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)
    meta = {
        "model": model_name,
        "embed_dim": model.embed_dim,
        "model_config": cfg["models"][model_name],
        "resolved_model": model.model_metadata,
        "count": len(ids),
        "dtype": "float32",
        "l2_normalized": True,
        "id_convention": "이미지 폴더 기준 상대 경로에서 확장자 제거",
        "image_folder": str(Path(cfg["data"]["image_folder"]).resolve()),
        "n_broken_files": len(loader.broken_files),
        "broken_files": loader.broken_files,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"완료: 임베딩 {embeddings.shape[0]}개 ({model.embed_dim}차원) -> {out_dir}/")
    if loader.broken_files:
        print(f"경고: 깨진 이미지 {len(loader.broken_files)}개 제외됨 (meta.json 참고)")
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="이미지 폴더를 임베딩해서 npy/json 저장"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.json")),
        help="설정 파일 경로 (기본: build_index.py 옆 config.json)",
    )
    parser.add_argument("--model", required=True, help=f"모델 이름 {available_models()}")
    parser.add_argument("--image-folder", default=None, help="config의 image_folder 덮어쓰기")
    parser.add_argument("--output-dir", default=None, help="config의 output_dir 덮어쓰기")
    parser.add_argument(
        "--image-list",
        default=None,
        help="이 목록 파일의 이미지만 임베딩 (한 줄에 하나, image_folder 기준 상대 또는 절대 경로)",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    if args.image_folder:
        cfg["data"]["image_folder"] = args.image_folder
    if args.output_dir:
        cfg["data"]["output_dir"] = args.output_dir

    files = None
    if args.image_list:
        with open(args.image_list, encoding="utf-8") as f:
            files = [line.strip() for line in f if line.strip()]

    build_index(cfg, args.model, files=files)


if __name__ == "__main__":
    main()
