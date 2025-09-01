# scoring_engine.py

import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Extractive prompt
ranking_prompt_template = """
You are an expert candidate evaluator.

You are given a candidate's resume data (flattened text below). Use only the provided data. Do not assume or invent any information.

Here is the hiring query:

{query}

Here is the candidate resume data:

{resume_text}

Strict JSON Output:

{{
  "CandidateName": "<Candidate Name>",
  "CurrentTitle": "<Current Title>",
  "CurrentCompany": "<Current Company>",
  "IndustryExperience": "<Yes/No with 1-line reasoning>",
  "LeadershipExperience": "<Yes/No with 1-line reasoning>",
  "TrackRecordOfSuccess": "<Yes/No with 1-line reasoning>",
  "PrivateEquityExperience": "<Yes/No with 1-line reasoning>",
  "FunctionalExpertise": "<Yes/No with 1-line reasoning>",
  "CompanySizeExperience": "<Yes/No with 1-line reasoning>",
  "TenureStability": "<Yes/No with 1-line reasoning>",
  "MatchScore": "<0-10 numeric score>",
  "Reasoning": "<Use candidate name. Summarize fit concisely>"
}}
"""

ranking_prompt = PromptTemplate(
    input_variables=["query", "resume_text"],
    template=ranking_prompt_template
)

# Scoring function
def score_candidate(resume_text, query):
    prompt = ranking_prompt.format(query=query, resume_text=resume_text)
    response = llm.invoke(prompt)
    return response.content.strip()
