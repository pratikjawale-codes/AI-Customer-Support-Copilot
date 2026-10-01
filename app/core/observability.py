import time
from threading import Lock
from typing import Any, Dict, List

from prometheus_client import Counter, Histogram

from app.config import settings

REQUESTS = Counter("virtual_assistant_requests_total", "Virtual assistant requests", ["endpoint"])
REQUEST_LATENCY = Histogram("virtual_assistant_request_latency_seconds", "Request latency", ["endpoint"])
TOKENS = Counter("virtual_assistant_tokens_total", "Estimated tokens processed", ["direction"])


class ObservabilityTracker:
    """In-memory request telemetry with Prometheus-compatible metrics."""

    def __init__(self):
        self.total_requests = 0
        self.total_latency_ms = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.feedback_counts: Dict[str, int] = {"helpful": 0, "not_helpful": 0}
        self.logs: List[Dict[str, Any]] = []
        self._lock = Lock()

    @staticmethod
    def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
        input_cost = (prompt_tokens / 1_000_000) * 0.15
        output_cost = (completion_tokens / 1_000_000) * 0.60
        return round(input_cost + output_cost, 6)

    def log_request(self, endpoint: str, prompt: str, response_text: str, latency_ms: float, docs_count: int = 0):
        prompt_tokens = len(prompt) // 4 + 10
        completion_tokens = len(response_text) // 4 + 10
        REQUESTS.labels(endpoint=endpoint).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency_ms / 1000)
        TOKENS.labels(direction="prompt").inc(prompt_tokens)
        TOKENS.labels(direction="completion").inc(completion_tokens)

        with self._lock:
            self.total_requests += 1
            self.total_latency_ms += latency_ms
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.logs.append({
                "request_id": f"REQ-{self.total_requests:05d}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "endpoint": endpoint,
                "latency_ms": round(latency_ms, 2),
                "retrieved_docs_count": docs_count,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost_usd": self.calculate_cost(prompt_tokens, completion_tokens),
                "provider_mode": "LIVE_OPENAI" if settings.is_live_openai_available() else "HEURISTIC_LOCAL"
            })
            self.logs[:] = self.logs[-100:]

    def record_feedback(self, rating: str):
        with self._lock:
            self.feedback_counts[rating] = self.feedback_counts.get(rating, 0) + 1

    def get_metrics_summary(self) -> Dict[str, Any]:
        with self._lock:
            total_tokens = self.prompt_tokens + self.completion_tokens
            return {
                "total_requests": self.total_requests,
                "avg_latency_ms": round(self.total_latency_ms / max(self.total_requests, 1), 2),
                "total_tokens_used": total_tokens,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_estimated_cost_usd": self.calculate_cost(self.prompt_tokens, self.completion_tokens),
                "feedback_breakdown": dict(self.feedback_counts),
                "provider_mode": "LIVE_OPENAI" if settings.is_live_openai_available() else "HEURISTIC_LOCAL",
                "recent_request_logs": list(self.logs[-10:])
            }


observability_service = ObservabilityTracker()
