import time

import streamlit as st

from src.pipeline import Pipeline


@st.cache_resource
def load_pipeline():
    return Pipeline()


pipeline = load_pipeline()

st.title("🔍 NLP 課程問答系統")

with st.sidebar:
    st.header("⚙️ 設定")
    bm25_k = st.slider("BM25 初篩數量", 1, 50, 20)
    final_k = st.slider("最終答案數量", 1, 10, 5)
    show_time = st.checkbox("顯示推論時間", value=True)

query = st.text_input("輸入您的問題", placeholder="例：什麼是 Time-homogeneous Markov process？")

if st.button("送出") and query.strip():
    with st.spinner("檢索中..."):
        t0 = time.time()
        try:
            result = pipeline.query(query, bm25_k=bm25_k, final_k=final_k)
        except Exception as exc:
            st.error("推論失敗，請稍後再試")
            st.exception(exc)
            st.stop()
        elapsed = time.time() - t0

    st.success(result["answer"])
    if show_time:
        st.caption(f"推論時間：{elapsed:.2f}s")

    with st.expander("📄 參考來源"):
        for i, chunk in enumerate(result["chunks"], 1):
            st.markdown(f"**[CHUNK {i}]** `{chunk['source_file']}`，第 {chunk['page']} 頁")
            st.markdown(chunk["text"])
            st.divider()
