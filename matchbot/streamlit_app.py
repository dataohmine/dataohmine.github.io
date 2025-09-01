
# ───────────────────────────────────────────────────────────
# 0) Page config must be first
# ───────────────────────────────────────────────────────────
import streamlit as st
st.set_page_config(page_title="PE Operator Finder", layout="wide")

# ───────────────────────────────────────────────────────────
# 1) Core imports
# ───────────────────────────────────────────────────────────
import os, json, re
# VECTORSTORE_PATH = os.path.join(os.getcwd(), "vectorstore")  # or "vectorstore_json", etc

# st.write("Looking in", VECTORSTORE_PATH)
# st.write(os.listdir(VECTORSTORE_PATH))

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
#from langchain.embeddings import HuggingFaceEmbeddings

# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from concurrent.futures import ThreadPoolExecutor, as_completed

# ───────────────────────────────────────────────────────────
# 2) Load API key
# ───────────────────────────────────────────────────────────
# comment out load_dotenv() if you're using Streamlit Secrets only
load_dotenv("api.env")  
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("🔑 Please set OPENAI_API_KEY in Streamlit Secrets or api.env/.env")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ───────────────────────────────────────────────────────────
# 3) Define FAISS loader with KeyError guard
# ───────────────────────────────────────────────────────────
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)  #gpt-4-0613

VECTORSTORE_PATH = os.path.join(os.getcwd(), "vectorstore")
#st.write("🔍 Looking in:", VECTORSTORE_PATH)
if not os.path.isdir(VECTORSTORE_PATH):
    st.error(f"Folder not found: {VECTORSTORE_PATH}")
    st.stop()
#st.write("Contents:", os.listdir(VECTORSTORE_PATH))
# You should see exactly: ['index.faiss','index.pkl']

# 4) Initialize embeddings
embeddings = OpenAIEmbeddings()  # uses OPENAI_API_KEY from env

# 5) Load FAISS index (pickle + FAISS file)
vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# try:
#     vectorstore = FAISS.load_local(
#         VECTORSTORE_PATH,
#         embeddings,
#         allow_dangerous_deserialization=True
#     )
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
#     st.success("✅ FAISS vectorstore loaded!")
# except Exception as e:
#     st.error(f"❌ Failed loading vectorstore: {e}")
#     st.stop()


# ───────────────────────────────────────────────────────────
# 5) Prompt setup
# ───────────────────────────────────────────────────────────

ranking_prompt_template = """
You are an expert Private Equity evaluator. Your task is to assess a candidate for a CEO/Operator role in a PE-backed business.

Here is the hiring query:

{query}

Here is the candidate’s parsed resume data (flattened text or JSON):

{resume_text}

Evaluate the following selection criteria strictly from the data provided:

1. **Industry Experience**: Hands-on experience in the target industry? List companies, years, sector relevance.  
2. **Leadership Experience**: Held CEO, President, or Division Head roles? Note P&L size, team scale, executive altitude.  
3. **Track Record of Success**: Evidence of revenue/EBITDA growth, successful exits or turnarounds? Quantifiable metrics.  
4. **Private Equity Experience**: Worked at or with PE-backed firms? Include board/reporting familiarity.  
5. **Functional Expertise**: Strong skills in key functions (Sales, Marketing, Finance, Tech, Ops)?  
6. **Company Size Experience**: Managed businesses at a scale similar to typical PE investments?  
7. **Tenure & Stability**: Demonstrated sufficient tenure and stability across roles?

**Strict JSON output** (no extra text):

{{
  "CandidateName": "<FirstName LastName>",
  "CurrentTitle": "<Most recent job title>",
  "CurrentCompany": "<Most recent company>",
  "IndustryExperience": "<Yes/No + 1-line reasoning>",
  "LeadershipExperience": "<Yes/No + 1-line reasoning>",
  "TrackRecordOfSuccess": "<Yes/No + 1-line reasoning>",
  "PrivateEquityExperience": "<Yes/No + 1-line reasoning>",
  "FunctionalExpertise": "<Yes/No + 1-line reasoning>",
  "CompanySizeExperience": "<Yes/No + 1-line reasoning>",
  "TenureStability": "<Yes/No + 1-line reasoning>",
  "MatchScore": "<0-100 numeric score>",
  "Reasoning": "<Begin by naming the candidate and their current title, then in three fluid sentences highlight how their industry and leadership experience, 
  measurable successes, private-equity background, core functional expertise, familiarity with similar company sizes, and demonstrated tenure and stability combine to make them an excellent fit.>"
}}
"""

#   "Reasoning": "<Use the candidate’s name to introduce, then in up to 3 sentences blend qualitative observations and key metrics to explain the overall fit.>"

ranking_prompt = PromptTemplate(
    input_variables=["query", "resume_text"],
    template=ranking_prompt_template
)


# ───────────────────────────────────────────────────────────
# 6) Helpers for JSON extraction & parsing
# ───────────────────────────────────────────────────────────
def extract_json(txt: str) -> str:
    txt = txt.strip()
    return re.sub(r"^```json|```$", "", txt, flags=re.IGNORECASE).strip()

def parse_candidate(content: str) -> dict | None:
    """
    Parse the JSON the LLM returned, trusting its 'Reasoning' field verbatim.
    """
    try:
        cleaned = extract_json(content)
        result = json.loads(cleaned)
        return {
            "CandidateName":            result.get("CandidateName", "Unknown"),
            "CurrentTitle":             result.get("CurrentTitle", "Unknown"),
            "CurrentCompany":           result.get("CurrentCompany", "Unknown"),
            "IndustryExperience":       result.get("IndustryExperience", ""),
            "LeadershipExperience":     result.get("LeadershipExperience", ""),
            "TrackRecordOfSuccess":     result.get("TrackRecordOfSuccess", ""),
            "PrivateEquityExperience":  result.get("PrivateEquityExperience", ""),
            "FunctionalExpertise":      result.get("FunctionalExpertise", ""),
            "CompanySizeExperience":    result.get("CompanySizeExperience", ""),
            "TenureStability":          result.get("TenureStability", ""),
            "MatchScore":               int(result.get("MatchScore", 0)),
            "Reasoning":                result.get("Reasoning", "")
        }
    except Exception as e:
        st.error(f"⚠️ Failed parsing candidate JSON: {e}")
        return None


# ───────────────────────────────────────────────────────────
# 7) Streamlit UI — clear input & show results immediately
# ───────────────────────────────────────────────────────────

st.title("🔎 PE Operator Finder")

# 1) Ensure our state keys exist
if "query_text" not in st.session_state:
    st.session_state.query_text = ""
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "results" not in st.session_state:
    st.session_state.results = []

# 2) Define the callback
def do_search():
    q = st.session_state.query_text.strip()
    if not q:
        return

    # Clear the box for the next time
    st.session_state.query_text = ""
    # Save what was searched
    st.session_state.last_query = q

    # Kick off retrieval + scoring
    docs = retriever.get_relevant_documents(q)
    ranked = []
    with ThreadPoolExecutor(max_workers=4) as exe:
        futures = [
            exe.submit(
                lambda doc=doc: llm.invoke(
                    ranking_prompt.format(query=q, resume_text=doc.page_content)
                ).content
            )
            for doc in docs
        ]
        for fut in as_completed(futures):
            raw = fut.result()
            cand = parse_candidate(raw)
            if cand:
                ranked.append(cand)

    # store top-5 in state
    if ranked:
        st.session_state.results = sorted(ranked, key=lambda x: x["MatchScore"], reverse=True)[:5]
    else:
        st.session_state.results = []

# 3) UI
# Text-area bound to session_state
st.text_area(
    "Enter your operator search query:",
    key="query_text",
    height=120
)

# Button with on_click → do_search
st.button("Run Search", on_click=do_search)

# 4) After any run, if we have a last_query, show it + results
if st.session_state.last_query:
    st.markdown(
    f"<div style='font-size:18px; font-weight:500; margin-bottom:0.5em;'>"
    f"You searched for: {st.session_state.last_query}"
    "</div>",
    unsafe_allow_html=True
)

    if not st.session_state.results:
        st.warning("No candidates found.")
    else:
        for rank, c in enumerate(st.session_state.results, start=1):
            st.markdown(f"#### Rank {rank}: {c['CandidateName']} — {c['CurrentTitle']} @ {c['CurrentCompany']}")
            st.write(f"**Match Score:** {c['MatchScore']}")
            st.write(f"**Reasoning:** {c['Reasoning']}")
            st.markdown("---")
