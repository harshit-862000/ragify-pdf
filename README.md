# 🌧️ ragify-pdf

> Chat with any PDF using Google Gemini embeddings and retrieval-augmented generation — built with LangChain and Streamlit.

---

## ✨ Features

- 📄 Upload any PDF and instantly index it
- 🔍 Ask questions in natural language
- 🤖 Answers powered by **Google Gemini 2.5 Flash**
- 🧠 Embeddings via **Gemini Embedding 001**
- 🗄️ Vector search with **ChromaDB** (in-memory)
- 💬 Conversation history within the session
- 📚 View source chunks used to generate each answer
- 💡 Suggested starter questions on first load

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| LLM | Google Gemini 2.5 Flash |
| Embeddings | Google Gemini Embedding 001 |
| Vector Store | ChromaDB |
| Orchestration | LangChain |
| PDF Loader | PyPDFLoader |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ragify-pdf.git
cd ragify-pdf
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your environment

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

> Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 4. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
ragify-pdf/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── .env                 # API keys (do NOT commit this)
├── .env.example         # Template for environment variables
├── .gitignore           # Ignores .env, __pycache__, etc.
└── README.md
```

---

## 🔒 Environment Variables

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Your Google Gemini API key |

Copy `.env.example` to `.env` and fill in your key.

---

## ☁️ Deploy on Streamlit Cloud

1. Push your code to GitHub (without the `.env` file)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. In **Settings → Secrets**, add:

```toml
GOOGLE_API_KEY = "your_actual_key_here"
```

Streamlit Cloud injects this automatically — no `.env` file needed.

---

## 🧠 How It Works

```
PDF Upload
    │
    ▼
PyPDFLoader → pages
    │
    ▼
RecursiveCharacterTextSplitter → chunks (size: 1000)
    │
    ▼
Gemini Embedding 001 → vector embeddings
    │
    ▼
ChromaDB (in-memory vector store)
    │
    ▼
User Question → similarity search (top 5 chunks)
    │
    ▼
Gemini 2.5 Flash → answer grounded in retrieved context
```

---

## 📦 Requirements

```
langchain
langchain_community
langchain-google-genai
langchain_chroma
python-dotenv
streamlit
pypdf
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

[MIT](LICENSE)
