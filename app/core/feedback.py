import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from threading import Lock

class FeedbackCollector:
    """
    Feedback & Fine-Tuning Dataset Generator
    Fulfills PDF Section 13:
    - Capture Agent Feedback (Helpful / Not Helpful)
    - Store Question, Retrieved Context, AI Response, Agent Edited Response
    - Save to dataset.jsonl for fine-tuning dataset creation
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or (Path(__file__).resolve().parent.parent / "data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_file = self.data_dir / "dataset.jsonl"
        self._write_lock = Lock()

    def save_feedback(
        self,
        ticket_id: str,
        question: str,
        retrieved_context: List[str],
        ai_response: str,
        agent_edited_response: Optional[str],
        rating: str,
        comments: Optional[str] = None
    ) -> Dict[str, Any]:
        
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ticket_id": ticket_id,
            "messages": [
                {"role": "user", "content": question},
                {"role": "system", "content": f"Context: {' | '.join(retrieved_context)}"},
                {"role": "assistant", "content": agent_edited_response or ai_response}
            ],
            "ai_original_response": ai_response,
            "agent_edited_response": agent_edited_response,
            "rating": rating,
            "comments": comments
        }

        with self._write_lock:
            with open(self.dataset_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        return {"status": "success", "file_saved": str(self.dataset_file)}

    def get_dataset_stats(self) -> Dict[str, Any]:
        count = 0
        if self.dataset_file.exists():
            with open(self.dataset_file, "r", encoding="utf-8") as f:
                count = sum(1 for line in f if line.strip())
        return {
            "total_fine_tuning_samples": count,
            "dataset_file_path": str(self.dataset_file)
        }

feedback_service = FeedbackCollector()
