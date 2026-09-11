"""(쿼리, 갤러리 아이템) 정답 판정 — 평가의 심장부.

판정 기준 3종:

    exact  : item_id 일치            — 완전히 같은 상품
    coarse : category 일치           — 같은 카테고리면 정답
    fine   : category + 속성 전부 일치

어떤 기준을 쓸지는 데이터셋마다 다르다 (config.json의 criteria):
    LookBench — coarse, fine   (속성 라벨이 있음)
    SOP       — exact          (속성 라벨이 없어 fine이 coarse와 같아짐)

★ 팀원 전원이 이 파일을 공유해야 모델 간 숫자 비교가 성립한다. 수정하면 공유할 것.

성능: SOP는 쿼리 6만 x 갤러리 6만이라 쿼리마다 갤러리 전체를 판정하면 못 쓴다.
judge_at()으로 검색된 top-K 위치만 판정한다. 문자열 id/category는 정수 코드로
바꿔 두어 numpy 비교가 C 레벨에서 돌게 한다.
"""

import numpy as np

CRITERIA = ("exact", "coarse", "fine")


class RelevanceIndex:
    """갤러리 메타데이터를 정수 코드로 올려두고 쿼리별 판정을 돌려준다."""

    def __init__(self, gallery_metas: list, fine_mode: str = "exact"):
        if fine_mode not in ("exact", "subset"):
            raise ValueError(f"fine_mode는 'exact' 또는 'subset': {fine_mode}")
        self.fine_mode = fine_mode
        self.size = len(gallery_metas)

        # 문자열 비교는 object 배열에서 파이썬 루프가 돌아 매우 느리다. 정수 코드로 변환.
        self._item_to_id = {}
        self.item_ids = np.empty(self.size, dtype=np.int64)
        self._cat_to_id = {}
        self.cat_ids = np.empty(self.size, dtype=np.int32)
        for row, meta in enumerate(gallery_metas):
            self.item_ids[row] = self._item_to_id.setdefault(
                meta["item_id"], len(self._item_to_id))
            self.cat_ids[row] = self._cat_to_id.setdefault(
                meta["category"], len(self._cat_to_id))

        vocab = sorted({a for m in gallery_metas for a in m["attrs"]})
        self._attr_to_id = {a: i for i, a in enumerate(vocab)}
        self.attr_mat = np.zeros((self.size, len(vocab)), dtype=np.float32)
        for row, meta in enumerate(gallery_metas):
            for attr in meta["attrs"]:
                self.attr_mat[row, self._attr_to_id[attr]] = 1.0
        self.attr_cnt = self.attr_mat.sum(axis=1)

    def _judge_rows(self, query_meta: dict, rows) -> dict:
        cat_id = self._cat_to_id.get(query_meta["category"], -1)
        item_id = self._item_to_id.get(query_meta["item_id"], -1)

        same_cat = self.cat_ids[rows] == cat_id
        exact = self.item_ids[rows] == item_id

        q_ids = [self._attr_to_id[a] for a in query_meta["attrs"] if a in self._attr_to_id]
        q_cnt = float(len(query_meta["attrs"]))

        if q_ids:
            q_vec = np.zeros(self.attr_mat.shape[1], dtype=np.float32)
            q_vec[q_ids] = 1.0
            inter = self.attr_mat[rows] @ q_vec
        else:
            # rows가 슬라이스일 수 있으므로 이미 인덱싱한 배열의 크기를 쓴다
            inter = np.zeros(same_cat.shape[0], dtype=np.float32)

        if q_cnt == 0:
            # 속성이 없는 데이터셋(SOP)에서는 fine이 coarse로 퇴화한다
            fine = same_cat.copy()
        elif self.fine_mode == "exact":
            fine = same_cat & (inter == q_cnt) & (inter == self.attr_cnt[rows])
        else:  # subset: 쿼리 속성이 갤러리 아이템에 모두 포함되면 정답
            fine = same_cat & (inter == q_cnt)

        return {"exact": exact, "coarse": same_cat, "fine": fine}

    def judge_at(self, query_meta: dict, rows) -> dict:
        """검색된 top-K 위치에서만 판정한다. 각 값은 길이 len(rows) 불리언 배열."""
        return self._judge_rows(query_meta, np.asarray(rows))

    def judge(self, query_meta: dict) -> dict:
        """갤러리 전체 판정. 길이 N 배열. (작은 데이터셋 확인과 테스트용)"""
        return self._judge_rows(query_meta, slice(None))
