import time
import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import settings
from app.models.schemas import (
    TicketClassificationRequest, TicketClassificationResponse,
    IntentDetectionRequest, IntentDetectionResponse,
    SentimentAnalysisRequest, SentimentAnalysisResponse,
    RAGQueryRequest, RAGQueryResponse, KnowledgeDocument,
    ResponseGenerationRequest, MultiStyleResponse,
    HITLSubmitRequest, HITLSubmitResponse,
    SummarizationRequest, StructuredSummary,
    GuardrailCheckRequest, GuardrailCheckResponse,
    PIIMaskRequest, PIIMaskResponse,
    AgentRunRequest, AgentRunResponse,
    FeedbackRequest, ObservabilityMetricsResponse
)
from app.core.llm_engine import LLMEngine
from app.core.rag_engine import rag_service
from app.core.guardrails import GuardrailsEngine
from app.core.pii_masker import PIIMasker
from app.core.agentic_workflow import AgenticWorkflowEngine
from app.core.observability import observability_service
from app.core.feedback import feedback_service

app = FastAPI(
    title="Virtual Assistant Customer Support API",
    description="Support-agent prototype with local heuristics, RAG, workflow tools, guardrails, PII masking, and observability.",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

# Mount static files directory if exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "mode": "LIVE_OPENAI" if settings.is_live_openai_available() else "HEURISTIC_LOCAL_FALLBACK",
        "api_keys_configured": {
            "openai": settings.is_live_openai_available()
        }
    }

# --- Module 1: Ticket Classification ---
@app.post("/api/tickets/classify", response_model=TicketClassificationResponse)
def classify_ticket(req: TicketClassificationRequest):
    start = time.time()
    result = LLMEngine.classify_ticket(req.ticket_text)
    latency = (time.time() - start) * 1000
    observability_service.log_request("/api/tickets/classify", req.ticket_text, str(result), latency)
    return result

# --- Module 2: Intent Detection ---
@app.post("/api/tickets/intent", response_model=IntentDetectionResponse)
def detect_intent(req: IntentDetectionRequest):
    start = time.time()
    result = LLMEngine.detect_intent(req.ticket_text)
    latency = (time.time() - start) * 1000
    observability_service.log_request("/api/tickets/intent", req.ticket_text, str(result), latency)
    return result

# --- Module 3: Sentiment Analysis & Routing ---
@app.post("/api/tickets/sentiment", response_model=SentimentAnalysisResponse)
def analyze_sentiment(req: SentimentAnalysisRequest):
    start = time.time()
    result = LLMEngine.analyze_sentiment(req.ticket_text)
    latency = (time.time() - start) * 1000
    observability_service.log_request("/api/tickets/sentiment", req.ticket_text, str(result), latency)
    return result

# --- Module 4: RAG Solution Recommendation ---
@app.post("/api/rag/search", response_model=RAGQueryResponse)
def rag_search(req: RAGQueryRequest):
    start = time.time()
    docs = rag_service.retrieve(req.query, top_k=req.top_k)
    latency = (time.time() - start) * 1000
    
    formatted_docs = [KnowledgeDocument(**d) for d in docs]
    rec_answer = f"Based on retrieved policy: {docs[0]['content']}" if docs else "No relevant policy documents found."
    
    observability_service.log_request("/api/rag/search", req.query, rec_answer, latency, docs_count=len(docs))
    return {
        "query": req.query,
        "retrieved_documents": formatted_docs,
        "reranked_documents": formatted_docs,
        "recommended_answer": rec_answer
    }

# --- Module 5: Multi-Style AI Response Generation ---
@app.post("/api/response/generate", response_model=MultiStyleResponse)
def generate_responses(req: ResponseGenerationRequest):
    start = time.time()
    styles = LLMEngine.generate_responses(req.ticket_text, req.context_docs)
    latency = (time.time() - start) * 1000
    
    selected = req.style.lower()
    if selected not in styles:
        selected = "professional"
    chosen = styles.get(selected, styles["professional"])
    
    observability_service.log_request("/api/response/generate", req.ticket_text, chosen, latency)
    return {
        "professional": styles["professional"],
        "short": styles["short"],
        "empathetic": styles["empathetic"],
        "selected_style": selected,
        "chosen_response": chosen
    }

# --- Module 6: Human in the Loop (HITL) ---
@app.post("/api/hitl/submit", response_model=HITLSubmitResponse)
def hitl_submit(req: HITLSubmitRequest):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    delivered = req.action != "rejected"
    return {
        "status": "success" if delivered else "rejected",
        "message": f"Response {req.action} by Human Agent ({req.agent_id})."
        + (" Delivered to customer." if delivered else " Not sent to customer."),
        "logged_timestamp": timestamp
    }

# --- Module 7: Conversation Summarization ---
@app.post("/api/summarize", response_model=StructuredSummary)
def summarize_conversation(req: SummarizationRequest):
    start = time.time()
    msgs = [{"sender": m.sender, "text": m.text} for m in req.conversation]
    summary = LLMEngine.summarize_conversation(msgs)
    latency = (time.time() - start) * 1000
    observability_service.log_request("/api/summarize", str(msgs), str(summary), latency)
    return summary

# --- Module 8 & 9: Agentic Workflow & Tool Calling ---
@app.post("/api/agent/run", response_model=AgentRunResponse)
def run_agentic_workflow(req: AgentRunRequest):
    start = time.time()
    res = AgenticWorkflowEngine.run_agent(req.user_query, req.customer_id or "CUST-9921")
    latency = (time.time() - start) * 1000
    observability_service.log_request("/api/agent/run", req.user_query, res["final_response"], latency)
    return res

# --- Module 10: Guardrails Check ---
@app.post("/api/guardrails/check", response_model=GuardrailCheckResponse)
def check_guardrails(req: GuardrailCheckRequest):
    in_res = GuardrailsEngine.validate_input(req.text) if req.check_type in ("input", "both") else None
    out_res = GuardrailsEngine.validate_output(req.text) if req.check_type in ("output", "both") else None
    input_safe = in_res is None or in_res["is_safe"]
    output_safe = out_res is None or out_res["is_safe"]
    flags = list(dict.fromkeys((in_res["flags"] if in_res else []) + (out_res["flags"] if out_res else [])))
    selected = in_res if in_res and not input_safe else out_res
    sanitized = selected["sanitized_text"] if selected else req.text
    block_reason = selected["block_reason"] if selected else None

    return {
        "is_safe": input_safe and output_safe,
        "input_flagged": bool(in_res and in_res["input_flagged"]),
        "output_flagged": bool(out_res and out_res["output_flagged"]),
        "flags": flags,
        "sanitized_text": sanitized,
        "block_reason": block_reason
    }

# --- Module 11: PII Protection ---
@app.post("/api/pii/mask", response_model=PIIMaskResponse)
def mask_pii(req: PIIMaskRequest):
    masked, types = PIIMasker.mask_text(req.text)
    return {
        "original_text": req.text,
        "masked_text": masked,
        "detected_types": types
    }

# --- Module 12: Observability Metrics ---
@app.get("/api/observability/stats", response_model=ObservabilityMetricsResponse)
def get_observability_stats():
    return observability_service.get_metrics_summary()

@app.get("/metrics", include_in_schema=False)
def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/api/knowledge-base")
def get_knowledge_base():
    docs = rag_service.documents
    by_file = {}
    for doc in docs:
        by_file.setdefault(doc["file_name"], []).append(doc)
    return [
        {
            "title": file_name.replace("_", " ").removesuffix(".md").title(),
            "category": chunks[0]["category"],
            "chunks": len(chunks),
            "status": "Indexed"
        }
        for file_name, chunks in sorted(by_file.items())
    ]

@app.post("/api/knowledge-base/reindex")
def reindex_knowledge_base():
    rag_service.load_and_index_documents()
    return {"status": "success", "documents": len(rag_service.documents)}

# --- Module 13: Feedback Loop ---
@app.post("/api/feedback")
def record_feedback(req: FeedbackRequest):
    res = feedback_service.save_feedback(
        ticket_id=req.ticket_id,
        question=req.question,
        retrieved_context=req.retrieved_context,
        ai_response=req.ai_response,
        agent_edited_response=req.agent_edited_response,
        rating=req.rating,
        comments=req.comments
    )
    observability_service.record_feedback(req.rating)
    return res

# Mock Ticket Data Listing Endpoint for UI
@app.get("/api/mock-tickets")
def get_mock_tickets():
    ticket_file = BASE_DIR / "app" / "data" / "mock_tickets.json"
    if ticket_file.exists():
        with open(ticket_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# Catch-all web route serving frontend UI
@app.get("/", response_class=HTMLResponse)
def read_root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Virtual Assistant Customer Support backend is running.</h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
