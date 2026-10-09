import asyncio
import os
import tempfile
from typing import Any, List, Optional

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Support MultiQueryRetriever across standard and classic packages
try:
    from langchain.retrievers.multi_query import MultiQueryRetriever
except (ImportError, ModuleNotFoundError):
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

# Configuration
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
COLLECTION_NAME = "mirai_handbook_policy"
EMBEDDING_MODEL = "gemini-embedding-2"
LLM_MODEL = "gemini-3.8-flash"

SYSTEM_PROMPT = """You are the Autonomous MirAI Student Policy Advisor. Answer the student's question strictly and exclusively based on the official handbook excerpts provided below.

Strict Directives:
1. Ground every statement exclusively in the retrieved handbook context.
2. NEVER rely on pretrained knowledge, prior assumptions, or external information to supply institutional policies.
3. NEVER invent, extrapolate, or hallucinate fines, penalties, dates, deadlines, names, offices, contact info, procedures, or numerical values.
4. If the retrieved context does not contain sufficient evidence to answer the student's question completely, you must politely and clearly state: "The provided handbook does not contain sufficient information to answer this question." Do not attempt to guess or provide hypothetical answers.
5. Maintain an objective, helpful, and professional advisory tone.
6. Whenever available in the context, reference relevant handbook page numbers or sections.

Retrieved Handbook Context:
{context}"""

app = FastAPI(
    title="Autonomous MirAI Student Policy Advisor API",
    description="Backend service providing PDF policy ingestion and LCEL RAG-based advising.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request & Response Schemas
class ChatRequest(BaseModel):
    question: Optional[str] = Field(None, description="The student's policy query")
    query: Optional[str] = Field(None, description="Alternative field for student's policy query")


class ContextChunk(BaseModel):
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    retrieved_context: List[ContextChunk]


class IngestResponse(BaseModel):
    status: str
    message: str
    filename: str
    num_pages: int
    num_chunks: int


class StatusResponse(BaseModel):
    status: str
    persist_directory: str
    collection_name: str
    indexed_chunks: int


# Helpers
def get_validated_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or not api_key.strip():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GOOGLE_API_KEY environment variable is not configured. Please set it in .env or the system environment.",
        )
    return api_key.strip()


def sanitize_message(msg: str) -> str:
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if api_key and api_key in msg:
        msg = msg.replace(api_key, "[REDACTED_API_KEY]")
    return msg


def get_chroma_client() -> chromadb.PersistentClient:
    os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)


def format_context_docs(docs: List[Document]) -> str:
    if not docs:
        return "No relevant handbook context found."
    formatted = []
    for idx, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Handbook")
        page = doc.metadata.get("page", None)
        page_info = f", Page {page + 1}" if isinstance(page, int) else (f", Page {page}" if page else "")
        formatted.append(f"--- Context Excerpt {idx} (Source: {source}{page_info}) ---\n{doc.page_content.strip()}")
    return "\n\n".join(formatted)


# Endpoints
@app.get("/health", response_model=StatusResponse)
async def check_health():
    client = get_chroma_client()
    try:
        coll = client.get_collection(COLLECTION_NAME)
        count = coll.count()
    except Exception:
        count = 0

    return StatusResponse(
        status="healthy",
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
        indexed_chunks=count,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_handbook_pdf(file: UploadFile = File(...)):
    """
    Ingests an uploaded PDF handbook:
    - Saves upload temporarily and safely
    - Loads using PyPDFLoader
    - Splits text with RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    - Embeds with GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
    - Safely replaces/refreshes the local persistent ChromaDB collection
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF files are supported.",
        )

    api_key = get_validated_api_key()

    temp_path: Optional[str] = None
    try:
        # Safely stream file to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            temp_path = tmp.name
            bytes_written = 0
            while chunk := await file.read(1024 * 1024):  # 1MB buffer
                tmp.write(chunk)
                bytes_written += len(chunk)

        if bytes_written == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded PDF file is empty (0 bytes).",
            )

        # Load document using PyPDFLoader asynchronously
        try:
            loader = PyPDFLoader(temp_path)
            docs = await asyncio.to_thread(loader.load)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read or parse PDF handbook: {sanitize_message(str(e))}",
            )

        if not docs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No readable pages or text found in the uploaded PDF.",
            )

        # Split text using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = await asyncio.to_thread(text_splitter.split_documents, docs)

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid text chunks could be extracted from the PDF.",
            )

        # Preserve source metadata
        for chunk in chunks:
            chunk.metadata["source"] = file.filename

        # Initialize embeddings with gemini-embedding-2
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=api_key,
        )

        # Safely replace existing handbook index
        client = get_chroma_client()
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # Collection did not exist yet

        vectorstore = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )

        # Persist documents asynchronously
        await vectorstore.aadd_documents(chunks)

        return IngestResponse(
            status="success",
            message="Policy handbook ingested and indexed successfully.",
            filename=file.filename,
            num_pages=len(docs),
            num_chunks=len(chunks),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Handbook ingestion failed: {sanitize_message(str(e))}",
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


@app.post("/chat", response_model=ChatResponse)
async def chat_policy_advisor(payload: ChatRequest):
    """
    Answers student questions strictly grounded in handbook context using:
    - Persistent ChromaDB vector store
    - MultiQueryRetriever
    - ChatGoogleGenerativeAI (gemini-3.8-flash, temperature=0.0)
    - LCEL RAG chain
    """
    user_query = payload.question or payload.query
    if not user_query or not user_query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A non-empty 'question' must be provided in the request body.",
        )
    user_query = user_query.strip()

    api_key = get_validated_api_key()

    # Validate vector store exists and has documents
    client = get_chroma_client()
    try:
        coll = client.get_collection(COLLECTION_NAME)
        if coll.count() == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Vector store is empty. Please upload and ingest a policy handbook first via POST /ingest.",
            )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No handbook has been ingested yet. Please upload and ingest a policy handbook first via POST /ingest.",
        )

    try:
        # Initialize Embeddings and VectorStore
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=api_key,
        )

        vectorstore = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )

        # Initialize LLM with strict deterministic settings
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0.0,
            google_api_key=api_key,
        )

        # Build MultiQueryRetriever to expand query
        retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(),
            llm=llm,
        )

        # Retrieve documents asynchronously
        retrieved_docs: List[Document] = await retriever.ainvoke(user_query)

        # Build LCEL Chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: format_context_docs(x["documents"])
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        # Execute LCEL chain asynchronously
        answer = await rag_chain.ainvoke({
            "documents": retrieved_docs,
            "question": user_query,
        })

        return ChatResponse(
            answer=answer,
            retrieved_context=[
                ContextChunk(
                    content=doc.page_content,
                    metadata=doc.metadata,
                )
                for doc in retrieved_docs
            ],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate policy advisor answer: {sanitize_message(str(e))}",
        )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", "8000"))
    uvicorn.run("backend:app", host=host, port=port, reload=True)
