import time

import streamlit as st

from src.pipeline import Pipeline


@st.cache_resource
def load_pipeline():
    return Pipeline()


def render_sources(chunks: list[dict]) -> None:
    with st.expander("檢索結果", expanded=False):
        if not chunks:
            st.caption("無檢索結果")
            return
        for i, chunk in enumerate(chunks, 1):
            st.markdown(f"**[{i}]** `{chunk['source_file']}`，第 {chunk['page']} 頁")
            if "bm25_score" in chunk:
                st.caption(f"BM25: {chunk['bm25_score']:.4f}")
            if "vector_score" in chunk:
                st.caption(f"Vector: {chunk['vector_score']:.4f}")
            st.markdown(chunk["text"])
            st.divider()


pipeline = load_pipeline()

st.set_page_config(page_title="NLP 課程問答", page_icon="🔍", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 920px;
        padding-top: 3rem;
        padding-bottom: 5rem;
    }
    [data-testid="stSidebar"] {
        background: #f7f3ee;
    }
    .chat-title {
        font-size: 1.55rem;
        font-weight: 680;
        line-height: 1.35;
        margin-bottom: 0.2rem;
        padding-top: 0.1rem;
    }
    .chat-subtitle {
        color: #6f6860;
        font-size: 0.95rem;
        margin-bottom: 1.4rem;
    }
    .mode-caption {
        color: #6f6860;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("設定")
    answer_mode = st.radio(
        "回答模式",
        options=["extractive", "gemma"],
        format_func=lambda value: "一般檢索" if value == "extractive" else "Gemma 生成",
        horizontal=False,
    )
    bm25_k = st.slider("BM25 初篩數量", 1, 50, 20)
    final_k = st.slider("最終答案數量", 1, 10, 5)
    show_time = st.checkbox("顯示推論時間", value=True)
    if st.button("清除對話"):
        st.session_state.messages = []
        st.rerun()

st.markdown('<div class="chat-title">NLP 課程問答系統</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="chat-subtitle">從課程講義檢索答案，並保留每次查詢的來源段落。</div>',
    unsafe_allow_html=True,
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if show_time and "elapsed" in message:
                st.caption(f"推論時間：{message['elapsed']:.2f}s · 模式：{message['mode_label']}")
            render_sources(message.get("chunks", []))

prompt = st.chat_input("輸入問題...")

if prompt and prompt.strip():
    query = prompt.strip()
    mode_label = "一般檢索" if answer_mode == "extractive" else "Gemma 生成"

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("檢索中..."):
            t0 = time.time()
            try:
                result = pipeline.query(
                    query,
                    bm25_k=bm25_k,
                    final_k=final_k,
                    answer_mode=answer_mode,
                )
            except Exception as exc:
                st.error("推論失敗，請稍後再試")
                st.exception(exc)
                st.stop()
            elapsed = time.time() - t0

        st.markdown(result["answer"])
        if show_time:
            st.caption(f"推論時間：{elapsed:.2f}s · 模式：{mode_label}")
        render_sources(result["chunks"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "chunks": result["chunks"],
            "elapsed": elapsed,
            "mode_label": mode_label,
        }
    )
