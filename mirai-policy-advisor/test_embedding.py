"""
Smoke test for Google Generative AI embeddings in mirai-policy-advisor.

Embeds a short sentence using GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
and reports success or the exact API error.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key or not api_key.strip():
    print("❌ ERROR: GOOGLE_API_KEY environment variable is missing or empty.")
    print("Please set GOOGLE_API_KEY in your environment or .env file.")
    sys.exit(1)

model_name = "gemini-embedding-2"
test_sentence = "The MirAI Student Policy Advisor provides answers strictly grounded in official handbook documentation."

print(f"Testing GoogleGenerativeAIEmbeddings with model '{model_name}'...")

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    embeddings = GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=api_key.strip(),
    )
    vector = embeddings.embed_query(test_sentence)

    print("✅ SUCCESS: Embedding generated successfully!")
    print(f"   Model: {model_name}")
    print(f"   Input text: \"{test_sentence}\"")
    print(f"   Vector dimensions: {len(vector)}")
    sys.exit(0)

except Exception as ex:
    print(f"❌ EMBEDDING API ERROR: {str(ex)}")
    sys.exit(1)
