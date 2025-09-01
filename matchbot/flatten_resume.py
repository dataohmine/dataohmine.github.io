# flatten_resume.py

import json

def flatten_resume(resume_json):
    name = f"{resume_json.get('name', {}).get('first', '')} {resume_json.get('name', {}).get('last', '')}".strip()
    work = resume_json.get("work_experience", [{}])[0]
    enrichment = resume_json.get("enrichment", {})
    narrative = resume_json.get("narrative_summary", "Unknown")

    flattened = {
        "Candidate Name": name or "Unknown",
        "Current Title": work.get("job_title", "Unknown"),
        "Current Company": work.get("organization", "Unknown"),
        "Domain Space": enrichment.get("domain_space", "Unknown"),
        "Functional Background": enrichment.get("functional_background", "Unknown"),
        "PE Experience": enrichment.get("pe_backed_experience", "Unknown"),
        "PE Companies": ", ".join(enrichment.get("pe_backed_companies", [])),
        "Exit Experience": enrichment.get("exit_experience", "Unknown"),
        "Company Financials": enrichment.get("company_financials", "Unknown"),
        "Operating Metrics": enrichment.get("operating_metrics", "Unknown"),
        "Tenure Stability": enrichment.get("tenure_stability", "Unknown"),
        "Narrative Summary": narrative
    }

    formatted = "\n".join([f"{k}: {v}" for k, v in flattened.items()])
    return formatted
