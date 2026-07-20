"""
유사 이미지 검색 UI

실행:  streamlit run search/app.py
"""

import tempfile
from pathlib import Path

import streamlit as st

from pipeline import get_embedding, search_similar

# layout은 기본값(centered) — wide로 하면 이미지가 화면을 꽉 채워서 스크롤이 생긴다
st.set_page_config(page_title="유사 상품 검색")

# ---------------------------------------------------- 1. 입력 (사이드바)
with st.sidebar:
    st.header("검색")
    uploaded = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])
    top_k = st.slider("결과 개수", min_value=1, max_value=10, value=5)
    do_search = st.button(
        "유사한 상품 검색",
        type="primary",
        width="stretch",
        disabled=uploaded is None,
    )

if do_search:
    # streamlit이 준 파일은 메모리에 있으므로, 경로를 받는 함수에 넘기려면 임시 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    with st.spinner("검색 중..."):
        vector = get_embedding(tmp_path)
        results = search_similar(vector, top_k=top_k)

    # 썸네일을 클릭할 때마다 화면이 다시 그려지므로, 결과를 세션에 보관해둔다
    st.session_state.query_image = uploaded.getvalue()
    st.session_state.results = results
    st.session_state.selected = 0

# ---------------------------------------------------- 2. 결과
st.title("유사 상품 검색")

results = st.session_state.get("results")

if results is None:
    st.info("왼쪽에서 이미지를 업로드하고 검색 버튼을 눌러주세요.")

elif not results:
    st.warning("결과가 없습니다. search/mock_images/ 폴더에 샘플 이미지를 넣어주세요.")

else:
    selected = st.session_state.selected
    item = results[selected]

    left, right = st.columns(2)
    with left:
        st.caption("내가 올린 이미지")
        st.image(st.session_state.query_image, width="stretch")
    with right:
        st.caption(f"유사 상품 {selected + 1}위 · 유사도 {item['score']}")
        st.image(item["image_path"], width="stretch")

    # 썸네일 — 클릭하면 오른쪽 큰 이미지가 바뀐다
    st.divider()
    st.caption("다른 결과 보기")
    for row_start in range(0, len(results), 5):
        row = results[row_start:row_start + 5]
        for offset, (col, item) in enumerate(zip(st.columns(5), row)):
            i = row_start + offset
            with col:
                st.image(item["image_path"], width="stretch")
                if st.button(
                    f"{i + 1}위",
                    key=f"thumb_{i}",
                    width="stretch",
                    type="primary" if i == selected else "secondary",
                ):
                    st.session_state.selected = i
                    st.rerun()
