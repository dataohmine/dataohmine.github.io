
# vectorstore_builder.py

import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from flatten_resume import flatten_resume

# Input directory of enriched JSON resumes
DATA_DIR = r"your output direcctory"
VECTORSTORE_PATH = r"your vector store path"

# Load embedding model
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Build document list
documents = []
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".json"):
        with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
            resume_json = json.load(f)
            flattened_text = flatten_resume(resume_json)
            documents.append(Document(page_content=flattened_text, metadata={}))

# Build vectorstore
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local(VECTORSTORE_PATH)

print(" Vectorstore built and saved.")
