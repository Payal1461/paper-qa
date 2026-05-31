import os
import re
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
import streamlit as st
import numpy as np

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = None

ai = genai.Client(api_key=api_key)

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash-lite"


def make_chunks(text):
    text = re.sub(r"[ \t]+", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) < 1200:
            current = current + " " + s
        else:
            if len(current.strip()) > 50:
                chunks.append(current.strip())
            current = s
    if len(current.strip()) > 50:
        chunks.append(current.strip())
    return chunks


def embed_texts(texts):
    vectors = []
    for i in range(0, len(texts), 50):
        batch = texts[i:i + 50]
        result = ai.models.embed_content(model=EMBED_MODEL, contents=batch)
        for e in result.embeddings:
            vectors.append(np.array(e.values))
    return vectors


def top_chunks(question, chunks, vectors, k=8):
    q = ai.models.embed_content(model=EMBED_MODEL, contents=question)
    qv = np.array(q.embeddings[0].values)
    scores = []
    for v in vectors:
        scores.append(float(np.dot(qv, v) / (np.linalg.norm(qv) * np.linalg.norm(v))))
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    return [chunks[idx] for idx in order[:k]]


st.set_page_config(page_title="Research Paper Q&A", layout="centered")

st.markdown(
    """
    <style>
    .stApp { background-color: #9AADBF; }
    .block-container { padding-top: 2.5rem; max-width: 960px; }
    .app-header {
        background: linear-gradient(120deg, #6D98BA 0%, #C17767 100%);
        padding: 52px 44px; border-radius: 18px; margin-bottom: 36px;
    }
    .app-header h1 {
        color: #ffffff; margin: 0; font-size: 42px; font-weight: 700; letter-spacing: -0.5px;
    }
    .app-header p { color: #F3E7E0; margin: 12px 0 0 0; font-size: 19px; }
    .section-label {
        font-size: 21px; font-weight: 600; color: #210203 !important; margin: 24px 0 10px 0;
    }
    .stTextInput input { padding: 16px 18px; font-size: 18px; }
    [data-testid="stFileUploaderDropzone"] { padding: 32px; }
    div[data-testid="stExpander"] details { font-size: 17px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <h1>Research Paper Q&A</h1>
        <p>Upload a paper, ask a question, and get a clear answer backed by its sources.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Upload a paper</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a PDF", type="pdf", label_visibility="collapsed")

if uploaded_file and st.button("Process PDF", type="primary"):
    with st.spinner("Reading your paper..."):
        reader = PdfReader(uploaded_file)
        all_text = "\n".join((page.extract_text() or "") for page in reader.pages)
        chunks = make_chunks(all_text)
        vectors = embed_texts(chunks)
        st.session_state["chunks"] = chunks
        st.session_state["vectors"] = vectors
    st.caption("Processed " + str(len(chunks)) + " sections. Ask a question below.")

st.markdown('<div class="section-label">Ask a question</div>', unsafe_allow_html=True)
question = st.text_input(
    "Your question",
    placeholder="What problem does this paper solve?",
    label_visibility="collapsed",
)

if question and "chunks" in st.session_state:
    with st.spinner("Thinking..."):
        best = top_chunks(
            question, st.session_state["chunks"], st.session_state["vectors"], 8
        )
        context = "\n\n".join(best)
        instruction = (
            "You are a helpful research assistant. Using ONLY the context below from a "
            "research paper, answer the question clearly and in detail. If the answer is "
            "not in the context, say you don't know."
        )
        prompt = instruction + "\n\nContext:\n" + context + "\n\nQuestion: " + question
        response = ai.models.generate_content(model=CHAT_MODEL, contents=prompt)
    st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.write(response.text)
    with st.expander("Sources"):
        for i, c in enumerate(best):
            st.markdown("**" + str(i + 1) + ".** " + c[:300] + "...")
elif question:
    st.warning("Please upload a PDF and click 'Process PDF' first.")
