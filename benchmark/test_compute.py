"""GPU 실행 계층·결과물·사이트 집계 테스트 (GPU 불필요)."""

import json

import numpy as np
import pytest

from benchmark.compute import adapters, artifacts, backends, load_lineup
from benchmark.encode import cache_paths


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ------------------------------------------------------------------ lineup

def test_lineup_has_baseline_first_and_unique_keys():
    lineup = load_lineup()
    keys = [m["key"] for m in lineup["models"]]
    assert keys[0] == lineup["baseline"] == "dreamsim"
    assert len(keys) == len(set(keys)) == 10
    for spec in lineup["models"]:
        assert spec["branch"] and spec["label"]
        if spec.get("adapter"):
            assert spec["adapter"] in adapters.ADAPTERS


# ------------------------------------------------------------------ 결과물

def _write_cache(cache_dir, model, config, split, vecs, metas):
    vec_path, meta_path = cache_paths(cache_dir, model, f"lookbench-{config}", split)
    np.save(vec_path, unit(vecs))
    meta_path.write_text(json.dumps(metas), encoding="utf-8")


@pytest.fixture
def fake_caches(tmp_path, monkeypatch):
    """서브셋 갤러리 2장 + noise 2장, 쿼리 2개짜리 가짜 캐시."""
    cache = tmp_path / "cache"
    cache.mkdir()
    g_metas = [{"item_id": "s:0", "category": "bag", "attrs": ["red"]},
               {"item_id": "s:1", "category": "shoe", "attrs": []}]
    n_metas = [{"item_id": "noise:0", "category": "noise data", "attrs": []},
               {"item_id": "noise:1", "category": "noise data", "attrs": []}]
    q_metas = [{"item_id": "q:0", "category": "bag", "attrs": ["red"]},
               {"item_id": "q:1", "category": "shoe", "attrs": ["blue"]}]
    _write_cache(cache, "m", "s", "gallery", [[1, 0, 0, 0], [0, 1, 0, 0]], g_metas)
    _write_cache(cache, "m", "noise", "gallery", [[0, 0, 1, 0], [0, 0, 0, 1]], n_metas)
    # 쿼리 0은 bag(0번)에 가깝고, 쿼리 1은 noise(2번)에 가장 가깝다 → 1위 오답
    _write_cache(cache, "m", "s", "query", [[1, 0.1, 0, 0], [0, 0.5, 1, 0]], q_metas)

    cfg = {"data": {"embed_cache_dir": str(cache)},
           "datasets": {"lookbench": {"noise_config": "noise", "fine_mode": "exact",
                                      "revision": "abc"}}}
    monkeypatch.setattr(artifacts, "_bench_config", lambda: cfg)
    monkeypatch.setattr(artifacts, "retrieval_dir", lambda: tmp_path / "retrieval")
    monkeypatch.setattr(artifacts, "TOPK", 3)
    return tmp_path


def test_build_and_load_roundtrip(fake_caches):
    path = artifacts.build("m", "s", source={"branch": "b"})
    assert path.exists()
    data = artifacts.load("m", "s")

    assert data["rows"].shape == (2, 3)
    assert data["rows"][0, 0] == 0                       # 쿼리 0 → bag
    assert data["rows"][1, 0] == 2                       # 쿼리 1 → noise 0번 (전체 행 2)
    assert data["hits_coarse"][0].tolist() == [True, False, False]
    assert data["hits_fine"][0, 0]                       # 속성까지 같다
    assert not data["hits_coarse"][1, 0] and data["hits_coarse"][1, 1]
    assert not data["hits_fine"][1].any()                # 속성이 달라 fine 정답 없음
    assert data["n_rel_coarse"].tolist() == [1, 1]
    assert data["info"]["n_subset_gallery"] == 2
    assert data["info"]["n_gallery"] == 4
    assert data["info"]["source"] == {"branch": "b"}

    m = artifacts.metrics(data)
    assert m["coarse_recall@1"] == pytest.approx(0.5)
    assert m["coarse_recall@10"] == pytest.approx(1.0)
    # 쿼리 1: 정답이 2위 → AP = (1/2)/min(1, 10) = 0.5
    assert m["coarse_map@10"] == pytest.approx((1.0 + 0.5) / 2)


def test_digest_depends_on_order():
    a = [{"item_id": "x"}, {"item_id": "y"}]
    assert artifacts.ids_digest(a) != artifacts.ids_digest(a[::-1])
    assert artifacts.ids_digest(a) == artifacts.ids_digest([dict(r) for r in a])


# ------------------------------------------------------------------ 백엔드

def test_local_override_is_merged(tmp_path, monkeypatch):
    base = {"default": "local", "backends": {"local": {"type": "local", "python": None},
                                             "remote": {"type": "ssh", "host": "a", "repo_dir": "~/r"}}}
    over = {"default": "remote", "backends": {"remote": {"host": "me@gpu"}}}
    (tmp_path / "gpu.json").write_text(json.dumps(base), encoding="utf-8")
    (tmp_path / "gpu.local.json").write_text(json.dumps(over), encoding="utf-8")
    monkeypatch.setattr(backends, "GPU_PATH", tmp_path / "gpu.json")
    monkeypatch.setattr(backends, "GPU_LOCAL_PATH", tmp_path / "gpu.local.json")
    monkeypatch.delenv("AISUM_GPU", raising=False)

    be = backends.load_backend()
    assert isinstance(be, backends.SSHBackend)
    assert be.cfg["host"] == "me@gpu" and be.cfg["repo_dir"] == "~/r"

    monkeypatch.setenv("AISUM_GPU", "local")
    assert isinstance(backends.load_backend(), backends.LocalBackend)
    with pytest.raises(KeyError):
        backends.load_backend("nope")


def test_ssh_command_quotes_and_pull_paths():
    be = backends.SSHBackend("r", {"host": "u@h", "repo_dir": "~/my repo", "python": "python3",
                                   "ref": "origin/x"})
    cmd = be.remote_command("dreamsim", ["a", "b"], profile=False, force=True, force_encode=False)
    assert cmd.startswith("cd ~/'my repo' && git fetch")
    assert "--detach origin/x" in cmd
    assert "--model dreamsim --subsets a,b --skip-perf --force" in cmd
    assert be._remote_path("benchmark/results/x.json") == "my repo/benchmark/results/x.json"
    with pytest.raises(ValueError):
        backends.SSHBackend("r", {"host": "u@h"})


# ------------------------------------------------------------------ 어댑터

def test_register_and_config_patch(tmp_path):
    models = tmp_path / "benchmark" / "models"
    models.mkdir(parents=True)
    (models / "__init__.py").write_text(
        '_REGISTRY = {\n    "dreamsim": ("dreamsim", "DreamSim"),\n}\n', encoding="utf-8")
    (tmp_path / "benchmark" / "config.json").write_text('{"models": {}}', encoding="utf-8")

    adapters._register(tmp_path, "tipsv2_l14", "tipsv2", "TipsV2")
    adapters._register(tmp_path, "tipsv2_l14", "tipsv2", "TipsV2")    # 두 번 해도 한 줄
    adapters._add_config(tmp_path, "tipsv2_l14", {"revision": "r"})

    text = (models / "__init__.py").read_text(encoding="utf-8")
    assert text.count('"tipsv2_l14": ("tipsv2", "TipsV2")') == 1
    ns = {}
    exec(text, ns)
    assert set(ns["_REGISTRY"]) == {"dreamsim", "tipsv2_l14"}
    cfg = json.loads((tmp_path / "benchmark" / "config.json").read_text(encoding="utf-8"))
    assert cfg["models"]["tipsv2_l14"] == {"revision": "r"}


# ------------------------------------------------------------------ 사이트 집계

def _art(hits_c, ap_c, rows=None):
    hits_c = np.asarray(hits_c, dtype=bool)
    n = hits_c.shape[0]
    return {"rows": np.asarray(rows if rows is not None else np.arange(n * 10).reshape(n, 10)),
            "hits_coarse": hits_c, "hits_fine": hits_c,
            "ap_coarse": np.asarray(ap_c, dtype=np.float32),
            "ap_fine": np.asarray(ap_c, dtype=np.float32)}


def test_summary_weights_by_query_count_and_counts_wins():
    from benchmark.viewer import export_static as ex

    lineup = {"baseline": "base", "models": [{"key": "base"}, {"key": "m"}, {"key": "half"}]}
    z = np.zeros((1, 10), dtype=bool)
    hit = z.copy(); hit[0, 0] = True
    results = {
        "big": {"base": _art(np.vstack([hit, z, z]), [1, 0, 0]),
                "m": _art(np.vstack([hit, hit, z]), [1, 0.5, 0]),
                "half": _art(np.vstack([z, z, z]), [0, 0, 0])},
        "small": {"base": _art(hit, [1]), "m": _art(z, [0])},
    }
    s = ex.build_summary(lineup, results)

    assert s["all"]["base"]["n_queries"] == 4
    assert s["all"]["base"]["coarse"]["recall@1"] == pytest.approx(2 / 4)
    assert s["all"]["m"]["coarse"]["recall@1"] == pytest.approx(2 / 4)
    assert s["all"]["m"]["vs"]["coarse"] == {"win": 1, "tie": 2, "loss": 1}
    assert "half" not in s["all"]                # 서브셋 하나가 빠지면 '전체'를 내지 않는다
    assert "half" in s["big"] and "vs" not in s["big"]["base"]


def test_pick_queries_mixed_is_unique_and_sized():
    from benchmark.viewer import export_static as ex

    rng = np.random.default_rng(0)
    n = 40
    base_rows = np.arange(n * 10).reshape(n, 10)
    other_rows = base_rows.copy()
    other_rows[:5] += 100000                     # 앞 5개 쿼리만 결과가 완전히 다르다
    hits = np.zeros((n, 10), dtype=bool)
    found = {"base": _art(hits, np.zeros(n), base_rows), "m": _art(hits, np.zeros(n), other_rows)}

    picked = ex.pick_queries(found, "base", 10, "mixed", rng)
    assert len(picked) == len(set(picked)) == 10
    assert set(range(5)) <= set(picked)          # 갈리는 쿼리는 반드시 들어간다
    assert ex.pick_queries(found, "base", 100, "mixed", rng) == list(range(n))


# ------------------------------------------------------------------ 토큰 세기

def test_count_tokens_conv_patches_and_llm_length():
    import torch
    import torch.nn as nn

    from benchmark import profile_perf

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch = nn.Conv2d(3, 8, kernel_size=16, stride=16)     # 32x32 → 2x2 = 4패치
            self.stem = nn.Conv2d(3, 8, kernel_size=3, stride=1)         # 패치 층이 아니다

        def embed(self, images):
            x = torch.zeros(len(images), 3, 32, 32)
            self.patch(x)
            self.stem(x)

    counts = profile_perf.count_tokens(Fake(), list(range(6)), batch_size=4)
    assert counts == {"vision_tokens_per_image": 4.0, "llm_tokens_per_image": None}

    class LM(nn.Module):
        def forward(self, x, attention_mask=None):
            return x

    class FakeVLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = LM()

        def embed(self, images):
            mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])[: len(images)]
            self.language_model(torch.zeros(1), attention_mask=mask)

    counts = profile_perf.count_tokens(FakeVLM(), [0, 1], batch_size=2)
    assert counts["llm_tokens_per_image"] == pytest.approx(3.5)
