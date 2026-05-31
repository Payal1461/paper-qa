# 📄 Research Paper Q&A

An AI web app that lets you **upload any research paper (PDF) and ask questions about it** in plain English — and get clear answers backed by the exact sources from the paper.

Built with **Retrieval-Augmented Generation (RAG)**.

🔗 **Live demo:** https://ask-my-paper.streamlit.app/

---

## ✨ Features

- Upload any PDF and ask questions about it
- Answers are grounded **only** in the paper (it says "I don't know" instead of making things up)
- Shows the **source passages** each answer is based on
- Clean, simple web interface

---

## How it works (the RAG pipeline)

1. **Read** — extracts the text from the uploaded PDF
2. **Chunk** — splits the text into clean, sentence-aware pieces
3. **Embed** — converts each chunk into a vector (numbers that capture meaning) using Google's `gemini-embedding-001`
4. **Retrieve** — for a question, finds the most relevant chunks using cosine similarity
5. **Generate** — sends those chunks + the question to Gemini, which writes a grounded answer with sources

---

## Tech Stack

- **Python**
- **Google Gemini API** (`google-genai`) — embeddings + answer generation
- **Streamlit** — web interface
- **pypdf** — PDF text extraction
- **NumPy** — semantic similarity search

---

## Run it locally

```bash
# 1. Clone the repo
git clone https://github.com/payal1461/paper-qa.git
cd paper-qa

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Gemini API key in a .env file
echo "GEMINI_API_KEY=your_key_here" > .env

# 5. Run the app
streamlit run app.py
```

Get a free Gemini API key at https://aistudio.google.com/apikey

---

## What I learned

LLM APIs, embeddings, vector search, the full RAG pipeline, prompt design, handling API rate limits, and deploying a live web app.
