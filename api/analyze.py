import os
import logging
from fastapi import APIRouter
from core.vector_db import search_chunks, get_chunks_by_order, get_all_chunks
from models.qa_pipeline import answer_question

logger = logging.getLogger(__name__)

router = APIRouter()

CUAD_QUESTIONS = {
    # --- Core Identifiers ---
    "Document Name": "What is the name or title of this contract or agreement?",
    "Parties": "Who are the parties to this agreement? List the full legal names of all companies or individuals.",
    "Effective Date": "What is the effective date or execution date of this contract?",
    "Expiration Date": "What is the expiration date or end date of this contract?",

    # --- Key Terms ---
    "Governing Law": "What is the governing law or jurisdiction for this agreement?",
    "Assignment": "Does the contract restrict the right to assign or transfer the agreement? What are the restrictions?",
    "Renewal Term": "Is there an automatic renewal clause? What are the renewal terms and conditions?",

    # --- Financial Terms ---
    "Payment Terms": "What are the payment terms, fees, or compensation specified in this agreement?",
    "Limitation of Liability": "What is the limitation of liability? What is the maximum liability cap?",
    "Indemnification": "What are the indemnification obligations of each party?",

    # --- Restrictive Covenants ---
    "Non-Compete": "Is there a non-compete clause? What are the specific restrictions on competition?",
    "Confidentiality": "What are the confidentiality or non-disclosure obligations?",
    "Non-Solicitation": "Is there a non-solicitation clause? What does it restrict?",

    # --- Termination ---
    "Termination for Convenience": "Under what circumstances can this agreement be terminated for convenience (without cause)?",
    "Termination for Cause": "Under what circumstances can this agreement be terminated for cause or breach?",

    # --- IP ---
    "Intellectual Property Ownership": "Who owns the intellectual property created under this agreement?",
}

HIGH_CONFIDENCE = 6.0
LOW_CONFIDENCE = 3.0

CRITICAL_CATEGORIES = {
    "Limitation of Liability": ("HIGH", "Missing Limitation of Liability clause", 5),
    "Indemnification": ("HIGH", "Missing Indemnification clause", 4),
    "Governing Law": ("MEDIUM", "Missing Governing Law clause", 3),
    "Termination for Convenience": ("MEDIUM", "No Termination for Convenience clause found", 2),
    "Confidentiality": ("MEDIUM", "Missing Confidentiality clause", 2),
}

PREAMBLE_CATEGORIES = {"Parties", "Effective Date", "Document Name", "Expiration Date"}

SECTION_KEYWORDS = {
    "Termination for Convenience": ["termination", "terminate", "convenience"],
    "Termination for Cause": ["termination", "terminate", "cause", "breach", "default"],
    "Governing Law": ["governing law", "jurisdiction", "applicable law"],
    "Limitation of Liability": ["liability", "limitation", "damages", "indemnif"],
    "Indemnification": ["indemnif", "hold harmless", "defend"],
    "Confidentiality": ["confidential", "non-disclosure", "nda", "proprietary"],
    "Non-Compete": ["non-compete", "noncompete", "compete", "competition", "restrictive"],
    "Non-Solicitation": ["non-solicitation", "nonsolicitation", "solicit"],
    "Assignment": ["assign", "transfer", "delegate"],
    "Renewal Term": ["renewal", "renew", "auto-renew", "extend"],
    "Payment Terms": ["payment", "fee", "compensation", "commission", "price"],
    "Intellectual Property Ownership": ["intellectual property", "ip", "patent", "copyright", "trademark", "ownership"],
}


def _extract_document_name_from_filename(filename: str) -> str | None:
    """Derive a cleaner document title from a noisy SEC-style filename."""
    import re

    if not filename:
        return None

    name_part = filename.rsplit(".", 1)[0]
    name_part = re.sub(r"^[0-9a-fA-F-]{36}_", "", name_part)
    name_part = re.sub(r"^\s*[^A-Za-z0-9]+", "", name_part)

    exhibit_split = re.split(r"-EX-[\d.]+-", name_part, flags=re.IGNORECASE, maxsplit=1)
    if len(exhibit_split) > 1 and exhibit_split[1].strip():
        candidate = exhibit_split[1]
    else:
        candidate = name_part

    candidate = candidate.replace("_", " ")
    candidate = re.sub(r"^\s*[-:]+\s*", "", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()

    return candidate or None


def _assess_risk(category: str, answer: str, score: float) -> tuple:
    """Assess risk based on category, answer content, and confidence score."""
    risk_level = "LOW"
    risk_flag = None
    risk_points = 0
    
    # Missing critical clauses
    if category in CRITICAL_CATEGORIES and (not answer or score < LOW_CONFIDENCE):
        level, flag, points = CRITICAL_CATEGORIES[category]
        return level, flag, points
    
    # Dangerous phrasing detection
    if answer:
        text = answer.lower()
        
        if category in ("Termination for Convenience", "Termination for Cause"):
            if "without notice" in text or "immediately" in text:
                return "HIGH", "Allows immediate termination without notice period", 5
            if "sole discretion" in text:
                return "MEDIUM", "Termination at sole discretion of one party", 3
        
        if category == "Limitation of Liability":
            if "unlimited" in text or "no limit" in text:
                return "HIGH", "No cap on liability exposure", 5
        
        if category == "Non-Compete":
            if "worldwide" in text or "perpetual" in text or "indefinite" in text:
                return "HIGH", "Overly broad non-compete restrictions", 4
        
        if category == "Indemnification":
            if "sole" in text and "expense" in text:
                return "MEDIUM", "Potentially one-sided indemnification", 3
    
    # Low confidence flag
    if answer and score < LOW_CONFIDENCE:
        return "MEDIUM", "Low confidence — needs human review", 1
    
    return risk_level, risk_flag, risk_points


def _get_document_metadata(job_id: str) -> dict:
    """Extract metadata about the uploaded document from Redis."""
    from core.redis_client import redis_client
    
    metadata = {
        "filename": None,
        "total_files": None,
    }
    
    try:
        job_data = redis_client.hgetall(f"job:{job_id}")
        if job_data:
            filenames = job_data.get("filenames", "") or job_data.get(b"filenames", b"").decode()
            total = job_data.get("total_files", "") or job_data.get(b"total_files", b"").decode()
            metadata["filename"] = filenames if filenames else None
            metadata["total_files"] = int(total) if total else None
    except Exception as e:
        logger.warning(f"Could not retrieve document metadata: {e}")
    
    return metadata

def _get_best_chunks(job_id: str, category: str, question: str) -> list:
    """
    Get the best chunks for a given question category.
    Uses semantic search + section-aware boosting + preamble retrieval.
    """
    # Semantic search for this question
    chunks = search_chunks(job_id=job_id, query=question, top_k=10)
    
    # For preamble categories, aggressively gather context
    if category in PREAMBLE_CATEGORIES:
        # Get the first 10 chunks of the document (preamble is always at the top)
        early_chunks = get_chunks_by_order(job_id, limit=10)
        existing_texts = {c["text"] for c in chunks}
        for c in early_chunks:
            if c["text"] not in existing_texts:
                chunks.insert(0, c)  # Put preamble first
        
        # Additional semantic searches with preamble-specific queries
        alt_queries = {
            "Parties": [
                "This agreement is entered into by and between",
                "The parties to this agreement",
            ],
            "Effective Date": [
                "This agreement is made and entered into as of",
                "effective as of the date",
            ],
            "Document Name": [
                "This agreement titled",
                "agreement contract exhibit",
            ],
        }
        
        for alt_q in alt_queries.get(category, []):
            alt_chunks = search_chunks(job_id=job_id, query=alt_q, top_k=5)
            for c in alt_chunks:
                if c["text"] not in existing_texts:
                    chunks.append(c)
                    existing_texts.add(c["text"])
    
    # Boost chunks from sections that match the category keywords
    if category in SECTION_KEYWORDS:
        keywords = SECTION_KEYWORDS[category]
        
        def section_relevance(chunk):
            title = chunk.get("section_title", "").lower()
            for kw in keywords:
                if kw in title:
                    return 0  # Highest priority
            return 1  # Normal priority
        
        chunks.sort(key=section_relevance)
    
    return chunks


@router.get("/analyze/{job_id}")
def analyze_contract(job_id: str):
    """
    Runs the full RAG + LLM pipeline for a given document.
    Returns extracted entities/clauses, risk assessment, and document metadata.
    """
    results = {}
    total_risk_score = 0
    high_risk_count = 0
    medium_risk_count = 0
    
    # Get document metadata early (for filename-based fallbacks)
    doc_metadata = _get_document_metadata(job_id)
    filename = doc_metadata.get("filename", "") or ""
    
    for category, question in CUAD_QUESTIONS.items():
        chunks = _get_best_chunks(job_id, category, question)
        
        query_with_context = question
        if category in PREAMBLE_CATEGORIES and filename:
            query_with_context = f"{question}\n(Document filename for reference: {filename})"
        
        answer_data = answer_question(query=query_with_context, chunks=chunks)
        answer = answer_data.get("answer")
        score = answer_data.get("score", 0)
        
        # Parties — comprehensive multi-step extraction
        if category == "Parties":
            import re
            
            needs_improvement = (not answer or " and " not in answer)
            
            if needs_improvement:
                all_chunks = get_all_chunks(job_id)
                full_text = " ".join(c["text"] for c in all_chunks)
                
                entity_pattern = r'([A-Z][A-Za-z\s&,\.\'-]+(?:Inc\.|Corp\.|LLC|L\.P\.|Ltd\.|L\.L\.C\.|Co\.|N\.A\.|Limited|Corporation|Company))'
                entities = re.findall(entity_pattern, full_text)
                
                seen = set()
                unique_entities = []
                for e in entities:
                    cleaned = e.strip().rstrip(',').strip()
                    normalized = cleaned.lower()
                    if normalized not in seen and len(cleaned) > 5:
                        seen.add(normalized)
                        unique_entities.append(cleaned)
                
                if len(unique_entities) >= 2:
                    entity_list = ", ".join(unique_entities[:10])
                    party_q = (
                        f"From this list of entities found in the contract, "
                        f"which ones are the actual PARTIES (signatories) to this agreement? "
                        f"List only the party names separated by ' and '.\n\n"
                        f"Entities found: {entity_list}\n\n"
                        f"Contract context: {full_text[:2000]}"
                    )
                    party_data = answer_question(query=party_q, chunks=[])
                    party_answer = party_data.get("answer")
                    if party_answer and len(party_answer) > 5:
                        answer = party_answer
                        score = 8.0
                elif len(unique_entities) == 1 and not answer:
                    answer = unique_entities[0]
                    score = 6.0
                
                if answer and " and " not in answer and unique_entities:
                    other_entities = [
                        e for e in unique_entities 
                        if e.lower() not in answer.lower() and answer.lower() not in e.lower()
                    ]
                    if other_entities:
                        followup_q = (
                            f"This contract involves {answer}. "
                            f"The other entity mentioned is '{other_entities[0]}'. "
                            f"Is '{other_entities[0]}' the other party to this agreement? "
                            f"Answer with just the full legal name, or NOT_FOUND."
                        )
                        followup_data = answer_question(query=followup_q, chunks=chunks[:5])
                        other_party = followup_data.get("answer")
                        if other_party and other_party.lower() != answer.lower():
                            answer = f"{answer} and {other_party}"
                            score = 8.0
        
        # Document Name — filename fallback + SEC cleanup
        if category == "Document Name":
            import re
            if not answer and filename:
                answer = _extract_document_name_from_filename(filename)
                if answer:
                    score = 7.0
            
            # Clean up SEC filing numbers from Document Name
            if answer:
                answer = re.sub(r'^\d+\s+', '', answer)  # Remove leading numbers
                answer = re.sub(r'^EX-[\d.]+\s*', '', answer, flags=re.IGNORECASE)  # Remove EX-10.10
        
        # Retry with ALL chunks for critical NOT_FOUND fields
        retry_categories = {
            "Expiration Date", "Renewal Term", "Effective Date", 
            "Assignment", "Limitation of Liability", "Indemnification",
            "Termination for Cause",
        }
        if not answer and category in retry_categories:
            logger.warning(f"[RETRY] {category} returned NOT_FOUND — retrying with all chunks")
            all_chunks = get_all_chunks(job_id)
            if all_chunks:
                retry_data = answer_question(query=question, chunks=all_chunks)
                retry_answer = retry_data.get("answer")
                retry_score = retry_data.get("score", 0)
                if retry_answer:
                    answer = retry_answer
                    score = retry_score
        
        # Assess risk
        risk_level, risk_flag, risk_points = _assess_risk(category, answer, score)
        total_risk_score += risk_points
        
        if risk_level == "HIGH":
            high_risk_count += 1
        elif risk_level == "MEDIUM":
            medium_risk_count += 1
        
        # Determine confidence label
        if not answer:
            confidence_label = "NOT_FOUND"
        elif score >= HIGH_CONFIDENCE:
            confidence_label = "HIGH"
        elif score >= LOW_CONFIDENCE:
            confidence_label = "MEDIUM"
        else:
            confidence_label = "LOW"
        
        results[category] = {
            "question": question,
            "extracted_answer": answer,
            "confidence_score": round(score, 4),
            "confidence_label": confidence_label,
            "risk_level": risk_level,
            "risk_flag": risk_flag,
        }
    
    if total_risk_score >= 15:
        overall_risk = "CRITICAL"
    elif total_risk_score >= 8:
        overall_risk = "HIGH"
    elif total_risk_score >= 3:
        overall_risk = "MEDIUM"
    else:
        overall_risk = "LOW"
    
    return {
        "job_id": job_id,
        "document": doc_metadata,
        "risk_summary": {
            "overall_risk": overall_risk,
            "total_risk_score": total_risk_score,
            "high_risk_flags": high_risk_count,
            "medium_risk_flags": medium_risk_count,
            "categories_analyzed": len(CUAD_QUESTIONS),
        },
        "extraction_results": results,
    }


@router.get("/debug/chunks/{job_id}")
def debug_chunks(job_id: str):
    """Debug endpoint: view all stored text chunks with section metadata."""
    from core.vector_db import client, COLLECTION_NAME
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    query_filter = Filter(
        must=[FieldCondition(key="job_id", match=MatchValue(value=job_id))]
    )
    
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=query_filter,
        limit=200,
    )
    
    chunks = []
    for point in results[0]:
        chunks.append({
            "chunk_index": point.payload.get("chunk_index", 0),
            "section_title": point.payload.get("section_title", ""),
            "text": point.payload.get("text", ""),
        })
    
    chunks.sort(key=lambda x: x["chunk_index"])
    
    return {
        "job_id": job_id,
        "total_chunks": len(chunks),
        "sections": list(set(c["section_title"] for c in chunks)),
        "chunks": chunks,
    }
