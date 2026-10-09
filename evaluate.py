"""
Automated RAG Evaluator for MirAI Student Policy Advisor.

Executes the four certification test queries against the running FastAPI
backend (POST /chat), collects the generated answer and retrieved context,
evaluates each response using ChatGoogleGenerativeAI as an LLM judge with a strict
factual grounding rubric, and logs results to rag_eval_scores.csv.
"""

import json
import os
import re
import sys
from typing import Any, Dict, List

import pandas as pd
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OUTPUT_CSV = "rag_eval_scores.csv"

# Four exact certification audit questions from the problem statement
CERTIFICATION_QUESTIONS = [
    {
        "question": "I have 72% attendance. How many attendance marks will I get?",
        "expected_outcome": "4 marks.",
    },
    {
        "question": "I study at the Ratnam campus. I got sick and need medical leave. Who do I email and how many days do I have to submit my documents?",
        "expected_outcome": "Email Yashaswini Ma'am within exactly 7 days of the illness or treatment.",
    },
    {
        "question": "We want to start a new Cybersecurity society under the Tech Club. Do we ask Management directly?",
        "expected_outcome": "40% batch support is required, and the proposal goes to the Faculty Coordinator first, not Management.",
    },
    {
        "question": "How much is the fine for smoking a cigarette on campus?",
        "expected_outcome": "Tobacco is prohibited and leads to Disciplinary Committee action. The system must not invent a monetary fine.",
    },
]

JUDGE_SYSTEM_PROMPT = """You are an impartial, rigorous evaluation judge assessing an Autonomous Student Policy Advisor RAG system.

Your goal is to evaluate whether the generated answer:
1. Is strictly and factually supported by the retrieved context.
2. Accurately fulfills the expected outcome.
3. Completely avoids fabricating institutional policies, fines, deadlines, or procedures.

STRICT SCORING GUIDELINES:
- Score 5: Perfectly accurate, completely grounded in the retrieved context, satisfies all aspects of the expected outcome, with zero hallucinations.
- Score 4: Accurate and aligned with the expected outcome and retrieved context, with minor non-substantive omissions.
- Score 3: Partially correct, but omits a key operational condition or includes vague wording where exact details exist in the context.
- Score 2: Substantially incomplete, partially contradicts the expected outcome, or lacks sufficient grounding in retrieved context.
- Score 1: Inaccurate, contradicts official policy, fabricates nonexistent rules, or hallucinates values/fines.

CRITICAL FACTUALITY RULES:
- Do NOT reward a correct-sounding answer if it is unsupported by the retrieved handbook evidence.
- For Question 4 (smoking policy): The handbook specifies tobacco prohibition and Disciplinary Committee action, but contains no specific monetary fine. If the generated answer invents or quotes a specific numerical fine (e.g., $100, ₹500, etc.), it MUST receive a score of 1.

Return your evaluation as a JSON object with exactly two keys:
{
  "score": <integer between 1 and 5>,
  "reasoning": "<concise justification explaining the score and factual grounding assessment>"
}
"""


def evaluate_response(
    judge_llm: ChatGoogleGenerativeAI,
    question: str,
    expected_outcome: str,
    retrieved_context_text: str,
    generated_answer: str,
) -> Dict[str, Any]:
    prompt = f"""Question: {question}
Expected Outcome: {expected_outcome}

Retrieved Handbook Context:
{retrieved_context_text if retrieved_context_text.strip() else "[No context retrieved]"}

Generated Answer:
{generated_answer}

Provide your score (1-5) and concise reasoning in valid JSON format.
"""

    try:
        response = judge_llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)

        # Extract JSON object from judge response
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            parsed = json.loads(match.group(0))
            score = int(parsed.get("score", 1))
            # Clamp score between 1 and 5
            score = max(1, min(5, score))
            reasoning = str(parsed.get("reasoning", "No reasoning provided.")).strip()
            return {"score": score, "reasoning": reasoning}
        else:
            return {
                "score": 1,
                "reasoning": f"Judge response did not contain valid JSON. Raw output: {text[:200]}",
            }
    except Exception as e:
        return {
            "score": 1,
            "reasoning": f"Judge invocation failed with error: {str(e)}",
        }


def main():
    if not GOOGLE_API_KEY or not GOOGLE_API_KEY.strip():
        print("ERROR: GOOGLE_API_KEY environment variable is missing or empty.")
        print("Please set GOOGLE_API_KEY in your environment or in a .env file.")
        sys.exit(1)

    print(f"Connecting to FastAPI backend at: {BACKEND_URL}")

    # Check backend connectivity
    try:
        health_resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if health_resp.status_code != 200:
            print(f"ERROR: Backend returned HTTP {health_resp.status_code}: {health_resp.text}")
            sys.exit(1)
        health_data = health_resp.json()
        indexed_chunks = health_data.get("indexed_chunks", 0)
        print(f"Backend status: {health_data.get('status')} | Indexed chunks: {indexed_chunks}")
        if indexed_chunks == 0:
            print(
                "WARNING: The ChromaDB vector store is currently empty (0 chunks).\n"
                "Please upload and ingest the handbook first (via Streamlit or POST /ingest) "
                "before running evaluation."
            )
            proceed = input("Continue anyway? (y/N): ").strip().lower()
            if proceed != "y":
                print("Evaluation aborted.")
                sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to backend at {BACKEND_URL}.")
        print("Ensure the FastAPI server is running with: uvicorn backend:app --port 8000")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to connect to health endpoint: {e}")
        sys.exit(1)

    # Initialize LLM Judge
    judge_llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.0,
        google_api_key=GOOGLE_API_KEY,
    )

    eval_results: List[Dict[str, Any]] = []

    print("\n" + "=" * 70)
    print("Executing MirAI Certification Test Suite (4 Questions)")
    print("=" * 70 + "\n")

    for idx, item in enumerate(CERTIFICATION_QUESTIONS, 1):
        question = item["question"]
        expected_outcome = item["expected_outcome"]

        print(f"[{idx}/4] Query: \"{question}\"")
        print(f"      Expected: {expected_outcome}")

        # Send question to backend
        try:
            resp = requests.post(
                f"{BACKEND_URL}/chat",
                json={"question": question},
                timeout=60,
            )

            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "")
                retrieved_context_list = data.get("retrieved_context", [])

                # Format retrieved context
                context_chunks = []
                for c in retrieved_context_list:
                    content = c.get("content", "").strip() if isinstance(c, dict) else str(c)
                    context_chunks.append(content)
                context_text = "\n\n".join(context_chunks)

                print(f"      Retrieved {len(retrieved_context_list)} context chunk(s).")
                print(f"      Evaluating generated answer with LLM judge...")

                # Run evaluation judge
                judge_result = evaluate_response(
                    judge_llm=judge_llm,
                    question=question,
                    expected_outcome=expected_outcome,
                    retrieved_context_text=context_text,
                    generated_answer=answer,
                )

                score = judge_result["score"]
                reasoning = judge_result["reasoning"]

                print(f"      -> Score: {score}/5")
                print(f"      -> Reasoning: {reasoning}\n")

                eval_results.append({
                    "question": question,
                    "expected_outcome": expected_outcome,
                    "answer": answer,
                    "score": score,
                    "reasoning": reasoning,
                    "retrieved_context": context_text,
                })

            else:
                try:
                    err_detail = resp.json().get("detail", resp.text)
                except Exception:
                    err_detail = resp.text
                err_msg = f"HTTP {resp.status_code} Error: {err_detail}"
                print(f"      -> FAILED: {err_msg}\n")
                eval_results.append({
                    "question": question,
                    "expected_outcome": expected_outcome,
                    "answer": f"ERROR: {err_msg}",
                    "score": 1,
                    "reasoning": f"Backend request failed with HTTP {resp.status_code}: {err_detail}",
                    "retrieved_context": "",
                })

        except requests.exceptions.Timeout:
            print("      -> FAILED: Request timed out (60s).\n")
            eval_results.append({
                "question": question,
                "expected_outcome": expected_outcome,
                "answer": "ERROR: Request timed out",
                "score": 1,
                "reasoning": "FastAPI backend /chat timed out after 60 seconds.",
                "retrieved_context": "",
            })
        except requests.exceptions.ConnectionError:
            print(f"      -> FAILED: Could not connect to backend at {BACKEND_URL}.\n")
            eval_results.append({
                "question": question,
                "expected_outcome": expected_outcome,
                "answer": "ERROR: Connection failed",
                "score": 1,
                "reasoning": f"Failed to connect to backend at {BACKEND_URL}.",
                "retrieved_context": "",
            })
        except Exception as e:
            print(f"      -> FAILED: {e}\n")
            eval_results.append({
                "question": question,
                "expected_outcome": expected_outcome,
                "answer": f"ERROR: {str(e)}",
                "score": 1,
                "reasoning": f"Unexpected execution exception: {str(e)}",
                "retrieved_context": "",
            })

    # Save outputs to CSV
    df = pd.DataFrame(eval_results)
    df.to_csv(OUTPUT_CSV, index=False)

    print("=" * 70)
    print(f"Evaluation complete! Results saved to: {OUTPUT_CSV}")
    print("=" * 70)
    print(df[["question", "score", "reasoning"]].to_string(index=False))


if __name__ == "__main__":
    main()
