import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from threading import Lock
from difflib import SequenceMatcher
from app.core.pii_masker import PIIMasker

class FeedbackCollector:
    """
    Feedback & Learning Loop Engine
    Fulfills Layer 7 (Human-in-the-Loop) and Layer 9 (Summarization & Feedback):
    - Stores complete audit record: ticket_id, question, retrieved_context, AI response, agent_edited_response
    - Computes edit distance ratio via difflib SequenceMatcher
    - Tags reason for edit/rejection (e.g. wrong facts, wrong tone, missing info)
    - Automatically masks PII before writing to disk for dataset safety and compliance
    - Prepares dataset.jsonl for future prompt improvement and fine-tuning
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or (Path(__file__).resolve().parent.parent / "data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_file = self.data_dir / "dataset.jsonl"
        self._write_lock = Lock()

    @staticmethod
    def calculate_edit_ratio(draft: str, final: str) -> float:
        """
        Calculates Levenshtein-based edit ratio.
        0.0 = accepted unchanged, 1.0 = completely rewritten.
        """
        if not draft and not final:
            return 0.0
        matcher = SequenceMatcher(None, draft.strip(), final.strip())
        return round(1.0 - matcher.ratio(), 4)

    def save_feedback(
        self,
        ticket_id: str,
        question: str,
        retrieved_context: List[str],
        ai_response: str,
        agent_edited_response: Optional[str],
        rating: str,
        comments: Optional[str] = None,
        reasons: Optional[List[str]] = None,
        prompt_version: str = "v1.2",
        agent_id: str = "agent_007"
    ) -> Dict[str, Any]:
        
        # Ensure PII is masked before writing to persistent storage (Layer 2 & 9 privacy rule)
        masked_question, _ = PIIMasker.mask_text(question)
        final_text = agent_edited_response if (agent_edited_response and agent_edited_response.strip()) else ai_response
        masked_final, _ = PIIMasker.mask_text(final_text)
        masked_ai, _ = PIIMasker.mask_text(ai_response)
        
        edit_ratio = self.calculate_edit_ratio(ai_response, final_text)

        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ticket_id": ticket_id,
            "agent_id": agent_id,
            "prompt_version": prompt_version,
            "question": masked_question,
            "retrieved_context": retrieved_context,
            "ai_original_response": masked_ai,
            "agent_final_response": masked_final,
            "edit_ratio": edit_ratio,
            "rating": rating,
            "reasons": reasons or [],
            "comments": comments,
            "messages": [
                {"role": "user", "content": masked_question},
                {"role": "system", "content": f"Context: {' | '.join(retrieved_context)}"},
                {"role": "assistant", "content": masked_final}
            ]
        }

        with self._write_lock:
            with open(self.dataset_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        return {
            "status": "success",
            "ticket_id": ticket_id,
            "edit_ratio": edit_ratio,
            "file_saved": str(self.dataset_file)
        }

    def get_dataset_stats(self) -> Dict[str, Any]:
        count = 0
        total_edit_ratio = 0.0
        helpful_count = 0
        if self.dataset_file.exists():
            with open(self.dataset_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            count += 1
                            total_edit_ratio += rec.get("edit_ratio", 0.0)
                            if rec.get("rating") == "helpful":
                                helpful_count += 1
                        except Exception:
                            continue
        return {
            "total_feedback_records": count,
            "helpful_count": helpful_count,
            "avg_edit_ratio": round(total_edit_ratio / max(count, 1), 4),
            "acceptance_rate": round((helpful_count / max(count, 1)) * 100, 2),
            "dataset_file_path": str(self.dataset_file)
        }

feedback_service = FeedbackCollector()
