"""DreamSim — 현재 search/vectordb 파이프라인이 쓰는 모델. 비교 기준선.

구현은 기존 파이프라인의 embedding/models/dreamsim.py를 그대로 재사용한다.
같은 코드를 두 벌 두지 않기 위해 여기서는 import만 한다
(기존 파이프라인은 수정하지 않는다).
"""

from embedding.models.dreamsim import DreamSim  # noqa: F401
