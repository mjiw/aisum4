"""(쿼리, 갤러리 아이템) 정답 판정 — 평가의 심장부.

LookBench는 정답이 이진(같은 상품)이 아니라 등급형이다. 판정 기준 4종:

    exact  : item_id 일치            — 완전히 같은 상품 (instance retrieval)
    coarse : category 일치           — 같은 카테고리면 정답
    fine   : category + 속성 전부 일치
    graded : nDCG용 0~1 점수. graded_from에 따라 계산 방식이 다르다.
             "attrs" (LookBench) — category 일치 시 속성 Jaccard
             "exact" (SOP)       — 상품 동일성. 속성 라벨이 없는 데이터셋용.

★ 팀원 전원이 이 파일을 공유해야 모델 간 숫자 비교가 성립한다. 수정하면 공유할 것.

성능: SOP는 쿼리 6만 x 갤러리 6만이라 쿼리마다 갤러리 전체를 판정하면 못 쓴다.
judge_at()으로 top-K 위치만 판정하고, IDCG는 ideal_graded()로 따로 구한다.
문자열 id/category는 정수 코드로 바꿔 두어 numpy 비교가 C 레벨에서 돌게 한다.
"""

import numpy as np


class RelevanceIndex:
    """갤러리 메타데이터를 정수 코드로 올려두고 쿼리별 판정을 돌려준다."""

    def __init__(self, gallery_metas: list, fine_mode: str = "exact",
                 graded_from: str = "attrs"):
        if fine_mode not in ("exact", "subset"):
            raise ValueError(f"fine_mode는 'exact' 또는 'subset': {fine_mode}")
        if graded_from not in ("attrs", "exact"):
            raise ValueError(f"graded_from은 'attrs' 또는 'exact': {graded_from}")
        self.fine_mode = fine_mode
        self.graded_from = graded_from
        self.size = len(gallery_metas)

        # 문자열 비교는 object 배열에서 파이썬 루프가 돌아 매우 느리다. 정수 코드로 변환.
        self._item_to_id = {}
        self.item_ids = np.empty(self.size, dtype=np.int64)
        for row, meta in enumerate(gallery_metas):
            self.item_ids[row] = self._item_to_id.setdefault(
                meta["item_id"], len(self._item_to_id))
        # item_id별 갤러리 개수 — exact 기준 IDCG를 정렬 없이 구하는 데 쓴다
        self._item_counts = np.bincount(self.item_ids, minlength=len(self._item_to_id))

        self._cat_to_id = {}
        self.cat_ids = np.empty(self.size, dtype=np.int32)
        for row, meta in enumerate(gallery_metas):
            self.cat_ids[row] = self._cat_to_id.setdefault(
                meta["category"], len(self._cat_to_id))

        vocab = sorted({a for m in gallery_metas for a in m["attrs"]})
        self._attr_to_id = {a: i for i, a in enumerate(vocab)}
        self.attr_mat = np.zeros((self.size, len(vocab)), dtype=np.float32)
        for row, meta in enumerate(gallery_metas):
            for attr in meta["attrs"]:
                self.attr_mat[row, self._attr_to_id[attr]] = 1.0
        self.attr_cnt = self.attr_mat.sum(axis=1)

    # --- 내부 계산: 주어진 행들에 대해서만 판정한다 ---
    def _judge_rows(self, query_meta: dict, rows):
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

        g_cnt = self.attr_cnt[rows]
        union = q_cnt + g_cnt - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            jaccard = np.where(union > 0, inter / union, 1.0)

        if q_cnt == 0:
            fine = same_cat.copy()
        elif self.fine_mode == "exact":
            fine = same_cat & (inter == q_cnt) & (inter == g_cnt)
        else:  # subset: 쿼리 속성이 갤러리 아이템에 모두 포함되면 정답
            fine = same_cat & (inter == q_cnt)

        if self.graded_from == "exact":
            graded = exact.astype(np.float32)
        else:
            graded = np.where(same_cat, jaccard, 0.0).astype(np.float32)

        return {"exact": exact, "coarse": same_cat, "fine": fine, "graded": graded}

    def judge_at(self, query_meta: dict, rows) -> dict:
        """검색된 top-K 위치에서만 판정한다. 각 값은 길이 len(rows) 배열."""
        return self._judge_rows(query_meta, np.asarray(rows))

    def judge(self, query_meta: dict) -> dict:
        """갤러리 전체 판정. 길이 N 배열. (작은 데이터셋과 테스트용)"""
        return self._judge_rows(query_meta, slice(None))

    def ideal_graded(self, query_meta: dict, max_k: int) -> np.ndarray:
        """nDCG의 IDCG용 — 갤러리 전체에서 가능한 최고 graded 점수 상위 max_k개(내림차순).

        graded_from="exact"면 점수가 0/1뿐이라 같은 상품 개수만 세면 되고,
        갤러리 전체를 정렬할 필요가 없다 (SOP 규모에서 결정적인 차이).
        """
        if self.graded_from == "exact":
            item_id = self._item_to_id.get(query_meta["item_id"], -1)
            n_rel = int(self._item_counts[item_id]) if item_id >= 0 else 0
            ideal = np.zeros(max_k, dtype=np.float32)
            ideal[: min(n_rel, max_k)] = 1.0
            return ideal

        graded_all = self._judge_rows(query_meta, slice(None))["graded"]
        k = min(max_k, graded_all.size)
        top = -np.partition(-graded_all, k - 1)[:k]
        top.sort()
        return top[::-1]
