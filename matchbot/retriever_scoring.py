# retriever_scoring.py

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from scoring_engine import score_candidate

# Load env for OpenAI key
load_dotenv('api.env')

VECTORSTORE_PATH = r"c:\users\chica\1_Projects\4_GenAI\8_RAG\vectorstore"

# Load vectorstore + retriever
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})

# Input user query
query = "Looking for CEOs in software who have driven revenue growth and exits."

# Retrieve top candidates
retrieved_docs = retriever.invoke(query)

# Run LLM scoring
results = []
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(score_candidate, doc.page_content, query) for doc in retrieved_docs]
    
    for future in as_completed(futures):
        result = future.result()
        if result:
            try:
                parsed = json.loads(result)
                parsed['MatchScore'] = float(parsed.get('MatchScore', 0))
                results.append(parsed)
            except Exception as e:
                print("Parsing failed:", e)

# Sort & display top results
results.sort(key=lambda x: x['MatchScore'], reverse=True)
top_results = results[:5]

for i, c in enumerate(top_results):
    print(f"Rank {i+1}: {c['CandidateName']} | Score: {c['MatchScore']}")
    print(f"Title: {c['CurrentTitle']} | Company: {c['CurrentCompany']}")
    print(f"Reasoning: {c['Reasoning']}")
    print("-" * 50)
