from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class CroppedImageLoader:
    def __init__(self, image_folder, files=None):
        self.image_folder = Path(image_folder).resolve()
        if not self.image_folder.is_dir():
            raise FileNotFoundError(f"이미지 폴더가 없습니다: {self.image_folder}")
        if files is None:
            self.files = sorted(
                p
                for p in self.image_folder.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
        else:
            self.files = sorted(set(self._resolve(p) for p in files))

        # id 중복 방지 
        ids = [self._image_id(p) for p in self.files]
        if len(set(ids)) != len(ids):
            dups = [i for i, c in Counter(ids).items() if c > 1]
            raise ValueError(f"id가 중복되는 파일이 있습니다: {dups[:10]}")

        self.broken_files = []

    def _resolve(self, path) -> Path:
        path = Path(path)
        path = (path if path.is_absolute() else self.image_folder / path).resolve()
        if not path.is_relative_to(self.image_folder):
            raise ValueError(f"이미지 폴더 밖의 경로는 id를 만들 수 없습니다: {path}")
        return path

    def __len__(self) -> int:
        return len(self.files)

    def _image_id(self, path: Path) -> str:
        return path.relative_to(self.image_folder).with_suffix("").as_posix()

    def iter_batches(self, batch_size: int):
        for start in range(0, len(self.files), batch_size):
            images, ids = [], []
            for path in self.files[start : start + batch_size]:
                try:
                    img = Image.open(path).convert("RGB")
                except (FileNotFoundError, UnidentifiedImageError, OSError, SyntaxError) as e:
                    print(f"WARNING: 이미지 열기 실패, 건너뜁니다: {path} ({e})")
                    self.broken_files.append(self._image_id(path))
                    continue
                images.append(img)
                ids.append(self._image_id(path))
            if images:
                yield images, ids
