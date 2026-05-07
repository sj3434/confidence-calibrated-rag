import streamlit as st
import os
from rag import ConfidenceCalibratedRAG

st.set_page_config(page_title="Confidence-Calibrated RAG", page_icon="🤖", layout="wide")

st.title("🤖 Confidence-Calibrated RAG Demo")
st.markdown("This application demonstrates an LLM system equipped with a **Dual-Layered Abstention Guardrail**. It prevents hallucinations by automatically saying *'I don't know'* when information is insufficient.")

@st.cache_resource
def load_rag():
    data_path = os.path.join(os.path.dirname(__file__), "../data/corpus.txt")
    rag = ConfidenceCalibratedRAG(data_path=data_path)
    
    # Auto-calibrate on startup
    calibration_data = [
        ("in", "What is Generative AI?"),
        ("in", "When was Columbia University founded?"),
        ("out", "How do I bake a cake?"),
        ("out", "What is the capital of France?")
    ]
    rag.calibrate(calibration_data)
    return rag

try:
    rag = load_rag()
except Exception as e:
    st.error(f"Failed to initialize RAG. Did you set OPENAI_API_KEY? Error: {e}")
    st.stop()

# Sidebar controls
st.sidebar.header("Configuration")
enable_abstention = st.sidebar.toggle("Enable Abstention Guardrails", value=True)
st.sidebar.markdown(f"**Current Calibration Threshold:** `{rag.threshold:.3f}`")
st.sidebar.info("When enabled, the system uses similarity scoring and an LLM-as-a-judge to decide whether to abstain. When disabled, it acts as a Naive RAG, which may hallucinate.")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "context_score" in msg and msg["context_score"]:
            st.caption(f"Context Similarity Score: {msg['context_score']:.3f}")

if prompt := st.chat_input("Ask a question (e.g., 'What is DeepEval?' or 'How to cook pasta?'):"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing and retrieving..."):
            answer, docs = rag.query(prompt, enable_abstention=enable_abstention)
            
            best_score = docs[0][1] if docs else 0.0
            
            st.markdown(answer)
            if docs:
                st.caption(f"🔍 Top Context Similarity Score: {best_score:.3f}")
            
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "context_score": best_score if docs else None
    })
