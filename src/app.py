# app.py: Streamlit UI
import streamlit as st
from rag import RAGAgent

st.set_page_config(page_title="AI-агент поиска статей", layout="wide")
st.title("🔍 AI-агент для поиска статей")

if "agent" not in st.session_state:
    st.session_state.agent = RAGAgent()

query = st.text_input("Введите запрос", placeholder="Как работают трансформеры?")

if st.button("Найти") and query:
    result = st.session_state.agent.generate_answer(query)
    
    st.subheader("Ответ")
    st.write(result["answer"])

    st.subheader("Источники")
    for src in result["sources"]:
        st.markdown(f"- [{src['title']}]({src['url']})")

    st.subheader("Вопросы для самопроверки")
    for q in result["questions"]:
        st.markdown(f"- {q}")
