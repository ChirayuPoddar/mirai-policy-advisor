# Autonomous MirAI Student Policy Advisor

A production-grade Retrieval-Augmented Generation (RAG) system built with FastAPI, Streamlit, LangChain, and ChromaDB to automate student policy guidance strictly grounded in official handbook documentation.

---

## 1. Project Overview & Architecture

The **Autonomous MirAI Student Policy Advisor** is designed to answer student inquiries regarding attendance, medical leave procedures, club and society formation, and disciplinary actions without hallucinating or inventing institutional policies.

### Architecture

```
                                 ┌────────────────────────────────┐
                                 │   Streamlit Frontend (UI)      │
                                 │       (frontend.py)            │
                                 └───────────────┬────────────────┘
                                                 │ HTTP / JSON
                                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (backend.py)                 │
│                                                                 │
│  ┌───────────────────────┐           ┌───────────────────────┐  │
│  │     POST /ingest      │           │      POST /chat       │  │
│  └───────────┬───────────┘           └───────────┬───────────┘  │
│              │                                   │              │
│       PyPDFLoader                         MultiQueryRetriever   │
│              │                                   │              │
│  RecursiveCharacterTextSplitter                  │ (Query       │
│  (chunk_size=1000, overlap=200)                  │  Expansion)  │
│              │                                   │              │
│  GoogleGenerativeAIEmbeddings                    ▼              │
│  (gemini-embedding-2)                   ChromaDB Vector Store   │
│              │                                   │              │
│              ▼                                   ▼              │
│     ChromaDB Persistent              Retrieved Policy Chunks    │
│       Vector Store                               │              │
│        (./chroma_db)                             ▼              │
│                                       ChatGoogleGenerativeAI    │
│                                       (gemini-3.8-flash, T=0.0) │
│                                                  │              │
│                                             LCEL Chain          │
└──────────────────────────────────────────────────┼──────────────┘
                                                   │
                                                   ▼
                                       Grounded Answer + Sources
```

- **Document Processing**: `PyPDFLoader` extracts text from the official PDF handbook.
- **Semantic Chunking**: `RecursiveCharacterTextSplitter` with `chunk_size=1000` and `chunk_overlap=200` to preserve context across policy clauses.
- **Embeddings & Persistence**: `GoogleGenerativeAIEmbeddings` (`gemini-embedding-2`) stored in a persistent local `ChromaDB` instance (`./chroma_db`).
- **Query Expansion**: `MultiQueryRetriever` addresses the vocabulary gap between casual student language and formal institutional phrasing.
- **Generation & Guardrails**: `ChatGoogleGenerativeAI` (`gemini-3.8-flash`) configured with `temperature=0.0` and a strict system prompt forbidding pretrained knowledge reliance, policy fabrication, or invented fines/dates.
- **Evaluation**: LLM-as-a-judge benchmarking script (`evaluate.py`) that exports grading scores to `rag_eval_scores.csv`.

---

## 2. Python Environment Setup

### Prerequisites
- Python 3.10+ (tested with Python 3.14)
- A Google Gemini API key with access to `gemini-embedding-2` and `gemini-3.8-flash`

### Setup Instructions

1. Clone or navigate to the project directory:
   ```bash
   cd mirai-policy-advisor
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   # On Windows (Command Prompt): .venv\Scripts\activate
   # On Windows (PowerShell): .venv\Scripts\Activate.ps1
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 3. API Key Setup

1. Copy the example environment file to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Open `.env` and fill in your Gemini API key:
   ```env
   GOOGLE_API_KEY=your_actual_gemini_api_key_here
   BACKEND_HOST=0.0.0.0
   BACKEND_PORT=8000
   BACKEND_URL=http://localhost:8000
   CHROMA_PERSIST_DIRECTORY=./chroma_db
   ```

> **Security Note:** Never commit your `.env` file to version control. It is excluded by `.gitignore`.

---

## 4. Launching Backend & Frontend

Run the backend and frontend in two separate terminal windows.

### Terminal 1: Launch FastAPI Backend
```bash
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```
- API will be accessible at: `http://localhost:8000`
- Interactive API documentation (Swagger): `http://localhost:8000/docs`
- Health/Status endpoint: `http://localhost:8000/health`

### Terminal 2: Launch Streamlit Frontend
```bash
streamlit run frontend.py
```
- Web interface will open in your browser at: `http://localhost:8501`

---

## 5. Ingesting the Official Handbook

Before asking policy questions, the official student handbook (`Mirai_SoT_Policy_Handbook_2026.pdf`) must be uploaded and indexed.

### Option A: Via Streamlit UI (Recommended)
1. Navigate to the Streamlit app at `http://localhost:8501`.
2. In the left sidebar under **Policy Handbook**, click **Browse files** and select `Mirai_SoT_Policy_Handbook_2026.pdf`.
3. Click the **Upload & Index Handbook** button.
4. A success notification will display the processed page and chunk counts once indexing completes.

### Option B: Via cURL / API
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@Mirai_SoT_Policy_Handbook_2026.pdf"
```

---

## 6. Certification Test Suite

The system is designed to satisfy the four mandatory certification audit benchmarks:

1. **Precision Verification**
   - **Query:** `"I have 72% attendance. How many attendance marks will I get?"`
   - **Expected Outcome:** The system must state that the student will receive 4 marks.

2. **Multi-Hop Reasoning**
   - **Query:** `"I study at the Ratnam campus. I got sick and need medical leave. Who do I email and how many days do I have to submit my documents?"`
   - **Expected Outcome:** The system must synthesize information across different sections, instructing the user to email Yashaswini Ma'am within exactly 7 days of the illness or treatment.

3. **Process Verification**
   - **Query:** `"We want to start a new Cybersecurity society under the Tech Club. Do we ask Management directly?"`
   - **Expected Outcome:** The system must state that 40% batch support is required, and the proposal must be submitted to the Faculty Coordinator first, not Management.

4. **Negative Constraint Testing**
   - **Query:** `"How much is the fine for smoking a cigarette on campus?"`
   - **Expected Outcome:** The system must state that tobacco is prohibited and leads to Disciplinary Committee action, but it must not fabricate a specific monetary fine.

---

## 7. Running the Automated Evaluator

The automated benchmarking pipeline executes the four certification queries through the running system, evaluates them using an LLM-as-a-judge rubric (scoring 1 to 5 with reasoning), and writes the outputs to `rag_eval_scores.csv`.

### Steps to Run Evaluation

1. Make sure your FastAPI backend is running and the official handbook is ingested.
2. In your terminal with the virtual environment activated, execute:
   ```bash
   python evaluate.py
   ```
3. The script will log each question's evaluation to the console and export the results to:
   ```
   rag_eval_scores.csv
   ```

*Note: Execution requires active backend connectivity and a valid `GOOGLE_API_KEY`.*
