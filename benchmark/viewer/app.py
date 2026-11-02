"""검색 결과 비교 뷰어 — 같은 쿼리를 모델별로 어떻게 찾는지 눈으로 본다.

실행:
    streamlit run benchmark/viewer/app.py

숫자만으로는 "왜 그 모델이 나은지"가 안 보인다. Recall@1이 같아도 2~10위에
무엇을 끌어오는지는 모델마다 전혀 다르다. 이 뷰어는 DreamSim(현행 파이프라인
기준선)을 맨 위에 고정하고 그 아래에 다른 모델을 같은 쿼리로 나란히 놓는다.
정확도와 함께 속도·메모리도 한 표에 모아 현업 투입 판단에 쓸 수 있게 한다.

GPU가 필요 없다. `.embed_cache/`에 있는 임베딩만 읽어 행렬곱으로 검색한다.

**새 모델을 추가하면** 아래 두 가지를 돌리는 것으로 끝이다. 뷰어 코드는 손대지
않는다 — 모델 목록도, 표도 파일을 보고 스스로 채운다.

    python -m benchmark.run_eval     --model <이름> --config <서브셋>   # 캐시 + 정확도
    python -m benchmark.profile_perf --model <이름>                     # 속도 + 메모리
"""

import base64
import io
import random
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))                             # 저장소 루트

from benchmark.viewer import core, remote

THUMB_PX = 150
st.set_page_config(page_title="검색 결과 비교", layout="wide")

CSS = """
<style>
  .vw-row { display: flex; gap: 10px; overflow-x: auto; padding: 2px 0 8px; }
  .vw-card { flex: 0 0 auto; width: THUMBpx; }
  .vw-card img { width: 100%; height: THUMBpx; object-fit: cover;
                 border-radius: 8px; display: block;
                 border: 3px solid rgba(128,128,128,.3); }
  .vw-card.hit img { border-color: #16a34a; }
  .vw-meta { font-size: 11px; line-height: 1.35; margin-top: 4px;
             color: rgba(128,128,128,1); display: flex; gap: 6px; }
  .vw-meta b { color: inherit; font-weight: 600; }
  .vw-head { display: flex; align-items: baseline; gap: 8px;
             margin: 2px 0 6px; flex-wrap: wrap; }
  .vw-name { font-size: 15px; font-weight: 700; }
  .vw-base { font-size: 11px; font-weight: 700; letter-spacing: .04em;
             padding: 2px 7px; border-radius: 999px;
             background: rgba(99,102,241,.18); color: #6366f1; }
  .vw-chip { font-size: 12px; padding: 2px 8px; border-radius: 999px;
             background: rgba(128,128,128,.14); }
  .vw-chip.dim { opacity: .75; }
  .vw-up { color: #16a34a; font-weight: 600; }
  .vw-down { color: #dc2626; font-weight: 600; }
  .vw-qmeta { font-size: 13px; line-height: 1.8; }
  .vw-qmeta .k { color: rgba(128,128,128,1); margin-right: 6px; }
  hr.vw-sep { margin: 10px 0 14px; border: 0;
              border-top: 1px solid rgba(128,128,128,.22); }
</style>
""".replace("THUMB", str(THUMB_PX))
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=4096)
def thumb_b64(config: str, kind: str, row: int, n_subset: int) -> str:
    """썸네일 JPEG를 base64로. 모델 간 같은 이미지가 자주 겹쳐 캐시 효과가 크다."""
    img = (core.query_image(config, row) if kind == "query"
           else core.gallery_image(config, row, n_subset))
    img = img.convert("RGB")
    img.thumbnail((THUMB_PX * 2, THUMB_PX * 2))     # 레티나 대비 2배
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def card_html(src: str, rank: int, score: float, hit: bool, label: str) -> str:
    cls = "hit" if hit else "miss"
    mark = "●" if hit else "○"
    return (f'<div class="vw-card {cls}">'
            f'<img src="data:image/jpeg;base64,{src}" alt="{label}">'
            f'<div class="vw-meta"><b>{mark} {rank}위</b><span>{score:.3f}</span></div>'
            f'<div class="vw-meta"><span>{label}</span></div></div>')


def delta_html(value: float, base: float) -> str:
    diff = value - base
    if abs(diff) < 1e-9:
        return '<span class="vw-chip">기준과 같음</span>'
    cls = "vw-up" if diff > 0 else "vw-down"
    return f'<span class="vw-chip {cls}">{diff:+.3f}</span>'


@st.cache_data(show_spinner="브랜치 목록을 가져오는 중", ttl=600)
def candidate_models() -> list:
    """브랜치에 올라와 있는 후보 모델. 10분 캐시하고 버튼으로 강제 갱신한다."""
    remote.fetch()
    return remote.candidate_models()


def start_remote_job(branch: str, model: str, configs: list, profile: bool):
    """브랜치 모델 평가를 백그라운드로 시작한다. 페이지를 새로고침해도 계속 돈다."""
    log_dir = Path(tempfile.gettempdir()) / "aisum_remote"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{branch.replace('/', '-')}__{model}.log"
    log_path.write_text("", encoding="utf-8")

    cmd = [sys.executable, "-m", "benchmark.viewer.remote",
           "--branch", branch, "--model", model,
           "--configs", ",".join(configs), "--log", str(log_path)]
    if not profile:
        cmd.append("--no-profile")
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    st.session_state.remote_job = {
        "proc": proc, "log": str(log_path), "model": model, "branch": branch,
        "configs": configs, "started": datetime.now().strftime("%H:%M:%S"),
    }


def render_remote_job():
    """실행 중이거나 끝난 작업의 상태와 로그 꼬리를 보여준다."""
    job = st.session_state.get("remote_job")
    if not job:
        return
    code = job["proc"].poll()
    label = f'{job["branch"]} · {job["model"]} ({job["started"]} 시작)'
    if code is None:
        st.info(f"실행 중 — {label}")
    elif code == 0:
        st.success(f"완료 — {label}. 아래 새로고침을 누르면 표에 들어옵니다.")
    else:
        st.error(f"실패 (exit {code}) — {label}")

    tail = Path(job["log"]).read_text(encoding="utf-8", errors="replace").splitlines()
    if tail:
        st.code("\n".join(tail[-6:]), language=None)
    if st.button("진행 상황 새로고침", width="stretch"):
        if code is not None:
            st.cache_data.clear()      # 새 캐시·결과가 목록에 잡히도록
        st.rerun()


def ratio_text(row: dict) -> str:
    """DreamSim=1 기준 배수. 1보다 크면 느리다."""
    if row["is_baseline"]:
        return "1.00 (기준)"
    if row["speed_ratio"] is None:
        return "—"
    mark = "느림" if row["speed_ratio"] > 1 else "빠름"
    return f"{row['speed_ratio']:.2f}× {mark}"


# ------------------------------------------------------------------ 사이드바
with st.sidebar:
    st.header("검색 결과 비교")
    config = st.selectbox("서브셋", core.subsets())
    models = core.available_models(config)

    if not models:
        st.error("이 서브셋에 임베딩 캐시가 없습니다. "
                 "`python -m benchmark.run_eval --model <이름> "
                 f"--config {config}` 를 먼저 돌리세요.")
        st.stop()

    criterion = st.radio(
        "정답 기준", ["coarse", "fine"], horizontal=True,
        help="coarse: 같은 카테고리면 정답 / fine: 카테고리와 속성이 모두 일치해야 정답",
    )
    topk = st.slider("표시 개수", 5, 20, 10)

    others = [m for m in models if m != core.BASELINE]
    chosen = st.multiselect("비교할 모델", others, default=others,
                            format_func=core.label_of)
    show = ([core.BASELINE] if core.BASELINE in models else []) + chosen
    if core.BASELINE not in models:
        st.warning("기준 모델(DreamSim) 캐시가 없어 비교 기준 없이 표시합니다. "
                   f"`python -m benchmark.run_eval --model dreamsim --config {config}`")
    if not show:
        st.stop()

    rows = core.query_rows(config, show[0])
    # 서브셋을 바꾸면 쿼리 수가 달라져 이전 선택이 범위를 벗어날 수 있다.
    # 위젯이 만들어지기 전에 손봐야 Streamlit이 예전 값을 다시 쓰지 않는다.
    if st.session_state.get("query_select", 0) >= len(rows):
        st.session_state.query_select = 0
    if st.button("무작위 쿼리", width="stretch"):
        st.session_state.query_select = random.randrange(len(rows))
    picked = st.selectbox("쿼리 이미지", [i for i, _ in rows],
                          format_func=lambda i: rows[i][1], key="query_select")

    missing = core.missing_models(config)
    if missing:
        st.caption("캐시 없음: " + ", ".join(core.label_of(m) for m in missing)
                   + f" — `python -m benchmark.run_eval --model <이름> --config {config}`")

    # ------------------------------------------------ 브랜치에서 모델 가져오기
    with st.expander("브랜치에서 모델 가져오기", expanded=False):
        st.caption("팀원 브랜치에 올라와 있는 후보 모델을 **체크아웃 없이** 바로 "
                   "돌립니다. 임시 워크트리에서 실행하므로 지금 작업 중인 브랜치는 "
                   "그대로예요. 이름 뒤 ✓는 이 서브셋에서 이미 볼 수 있는 모델이에요.")
        if st.button("브랜치 목록 갱신", width="stretch"):
            candidate_models.clear()
        catalog = candidate_models()

        if not catalog:
            st.caption("원격 브랜치에서 모델을 찾지 못했습니다.")
        else:
            done = set(core.available_models(config))
            pick, branch = st.selectbox(
                "모델", catalog,
                format_func=lambda mb: f"{core.label_of(mb[0])}  ·  {mb[1]}"
                                       + ("  ✓" if mb[0] in done else ""))
            targets = st.multiselect("서브셋", core.subsets(), default=[config])
            with_perf = st.checkbox("속도·메모리도 측정", value=True)
            running = (st.session_state.get("remote_job", {}).get("proc")
                       and st.session_state["remote_job"]["proc"].poll() is None)
            if st.button("가져와서 돌리기", type="primary", width="stretch",
                         disabled=not targets or bool(running)):
                start_remote_job(branch, pick, targets, with_perf)
                st.rerun()
            if running:
                st.caption("다른 작업이 끝나야 시작할 수 있어요 (GPU는 한 번에 하나).")
            else:
                st.caption("갤러리 6만 장을 처음 인코딩하는 모델은 10~25분 걸려요. "
                           "창을 닫아도 작업은 계속됩니다.")
        render_remote_job()

bad = core.verify_alignment(config, show)
if bad:
    st.error("갤러리 순서가 기준 모델과 다른 캐시가 있습니다: "
             + ", ".join(core.label_of(m) for m in bad)
             + ". 해당 모델을 --force-encode로 다시 인코딩하세요.")
    st.stop()

# ------------------------------------------------------------------ 요약표
@st.cache_data(show_spinner=False)
def subset_metrics(model: str, config: str) -> dict:
    return dict(core.eval_metrics(model, config))


perf_rows = core.perf_table(show)
summary = []
with st.spinner("서브셋 전체 지표 계산 중 (모델·서브셋당 한 번)"):
    scores_by_model = {r["model"]: subset_metrics(r["model"], config) for r in perf_rows}

for row in perf_rows:
    model = row["model"]
    scores = scores_by_model[model]
    summary.append({
        "모델": row["label"] + (" (기준)" if row["is_baseline"] else ""),
        f"{criterion} R@1": scores.get(f"{criterion}_recall@1"),
        f"{criterion} R@10": scores.get(f"{criterion}_recall@10"),
        f"{criterion} mAP@10": scores.get(f"{criterion}_map@10"),
        "1000장당(초)": row.get("sec_per_1000"),
        "속도 (DreamSim=1)": ratio_text(row),
        "초당 장수": row.get("images_per_sec"),
        "최대 VRAM(MB)": row.get("peak_vram_mb"),
        "모델 크기(MB)": row.get("model_size_mb"),
        "임베딩 차원": row.get("embed_dim"),
        "6만 장 인덱스(MB)": row.get("gallery_60k_mb"),
    })

st.markdown("#### 모델 요약 — 정확도 · 속도 · 메모리")
st.dataframe(pd.DataFrame(summary).set_index("모델"), width="stretch",
             column_config={
                 f"{criterion} R@1": st.column_config.NumberColumn(format="%.4f"),
                 f"{criterion} R@10": st.column_config.NumberColumn(format="%.4f"),
                 f"{criterion} mAP@10": st.column_config.NumberColumn(format="%.4f"),
                 "1000장당(초)": st.column_config.NumberColumn(format="%.2f"),
                 "초당 장수": st.column_config.NumberColumn(format="%.1f"),
             })

unmeasured = [r["label"] for r in perf_rows if not r["measured"]]
if unmeasured:
    st.caption("속도·메모리 미측정: " + ", ".join(unmeasured)
               + " — `python -m benchmark.profile_perf --model <이름>`")
if core.perf_mismatch(perf_rows):
    st.warning("속도 측정 조건(이미지 수·배치·GPU)이 모델마다 달라 배수 비교가 "
               "정확하지 않습니다. 같은 조건으로 다시 측정하세요.")
else:
    measured = next((r for r in perf_rows if r["measured"]), None)
    if measured:
        st.caption(f"속도는 {measured['config']}/query {measured['n_images']}장, "
                   f"batch {measured['batch_size']}, {measured['gpu']} 기준이며 "
                   "정확도는 서브셋 전체 평균입니다. "
                   "절대 시간은 머신 상태에 따라 흔들리므로 배수를 보세요.")

st.markdown('<hr class="vw-sep">', unsafe_allow_html=True)

# ------------------------------------------------------------------ 쿼리
_, q_metas = core.load_queries(show[0], config)
q_meta = q_metas[picked]

left, right = st.columns([1, 3])
with left:
    st.image(core.query_image(config, picked), width=260)
with right:
    n_rel = core.relevance_index(config).count_relevant(q_meta, criterion)
    attrs = ", ".join(q_meta["attrs"]) or "—"
    st.markdown(
        f'<div class="vw-qmeta">'
        f'<div><span class="k">서브셋</span>{config}'
        f'<span class="k" style="margin-left:18px">쿼리</span>#{picked}</div>'
        f'<div><span class="k">카테고리</span>{q_meta["category"]}</div>'
        f'<div><span class="k">속성</span>{attrs}</div>'
        f'<div><span class="k">갤러리 정답 수</span>{n_rel:,}개 '
        f'<span class="k">/ 갤러리</span>{len(core.load_gallery(show[0], config)[1]):,}장</div>'
        f'</div>', unsafe_allow_html=True)

st.markdown('<hr class="vw-sep">', unsafe_allow_html=True)

# ------------------------------------------------------------------ 모델별 결과
baseline_scores, baseline_rows = None, set()
perf_by_model = {r["model"]: r for r in perf_rows}

for model in show:
    idx, scores = core.search_one(model, config, picked, topk)
    _, g_metas, n_subset = core.load_gallery(model, config)
    hits = core.judge_hits(config, q_meta, idx, criterion)
    m = core.query_scores(config, q_meta, hits, criterion)

    is_base = model == core.BASELINE and baseline_scores is None
    if is_base:
        baseline_scores, baseline_rows = m, set(idx[:topk].tolist())

    chips = [f'<span class="vw-chip">R@1 {m["recall@1"]:.0f}</span>',
             f'<span class="vw-chip">R@10 {m["recall@10"]:.0f}</span>',
             f'<span class="vw-chip">mAP@10 {m["map@10"]:.3f}</span>',
             f'<span class="vw-chip">top-10 정답 {m["hits@10"]}개</span>']
    if is_base:
        chips.insert(0, '<span class="vw-base">기준</span>')
    elif baseline_scores:
        overlap = len(baseline_rows & set(idx[:topk].tolist()))
        chips.append(delta_html(m["map@10"], baseline_scores["map@10"]))
        chips.append(f'<span class="vw-chip">기준과 겹침 {overlap}/{topk}</span>')

    row = perf_by_model[model]
    if row["measured"]:
        chips.append(f'<span class="vw-chip dim">{row["sec_per_1000"]:.1f}초/1000장 '
                     f'· {ratio_text(row)}</span>')
        chips.append(f'<span class="vw-chip dim">VRAM {row["peak_vram_mb"]:.0f}MB '
                     f'· {row["embed_dim"]}차원</span>')

    st.markdown(f'<div class="vw-head"><span class="vw-name">{core.label_of(model)}'
                f'</span>{"".join(chips)}</div>', unsafe_allow_html=True)

    cards = []
    for rank, (grow, score, hit) in enumerate(zip(idx, scores, hits), start=1):
        meta = g_metas[grow]
        tag = "noise" if grow >= n_subset else meta["category"]
        cards.append(card_html(thumb_b64(config, "gallery", int(grow), n_subset),
                               rank, float(score), bool(hit), tag))
    st.markdown(f'<div class="vw-row">{"".join(cards)}</div>', unsafe_allow_html=True)
    st.markdown('<hr class="vw-sep">', unsafe_allow_html=True)

st.caption("테두리가 초록색이면 현재 기준(coarse/fine)으로 정답입니다. "
           "숫자는 코사인 유사도이고, 캡션의 라벨은 갤러리 아이템의 카테고리입니다. "
           "noise는 공용 distractor 풀에서 온 이미지입니다.")
