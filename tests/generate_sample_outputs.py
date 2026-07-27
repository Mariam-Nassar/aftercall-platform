"""
Generates a synthetic outputs/ directory that mimics what
storage.py + evaluation.py would produce, so dashboard.py can
be tested end-to-end without running the real Gemini/RAG pipeline.
"""
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

BASE = Path(__file__).resolve().parent.parent / "outputs"

FOLDERS = {
    "records": "READY_TO_SAVE",
    "reviews": "PENDING_REVIEW",
    "escalations": "ESCALATED",
    "archive": "ARCHIVED",
}
DECISIONS = {
    "records": "AUTO_SAVE",
    "reviews": "HUMAN_REVIEW",
    "escalations": "ESCALATE",
    "archive": "NON_INTERACTION",
}

CATEGORIES = ["Billing", "Technical Support", "Account Management", "Sales"]
SUBCATEGORIES = ["Refund", "Password Reset", "Upgrade", "Cancellation", "Login Issue"]
PRIORITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
DISPOSITIONS = ["Resolved", "Unresolved", "Follow-up Required"]
SENTIMENTS = ["Positive", "Neutral", "Negative"]
KEYWORDS_POOL = ["refund", "invoice", "password", "login", "upgrade", "cancel", "billing", "outage"]
TAGS_POOL = ["vip", "repeat_caller", "sla_risk", "first_contact", "callback_requested"]


def make_record(call_id: str, folder: str, day_offset: int) -> dict:
    ts = datetime(2026, 1, 1, 9, 0, 0) + timedelta(days=day_offset, hours=random.randint(0, 8))
    return {
        "call_id": call_id,
        "customer_id": f"C-{1000 + int(call_id.split('-')[1])}",
        "documentation": {
            "summary": f"Synthetic summary for {call_id}.",
            "issue": "Synthetic issue description.",
            "root_cause": "Synthetic root cause.",
            "resolution": "Synthetic resolution.",
            "pending_actions": [],
            "category": random.choice(CATEGORIES),
            "subcategory": random.choice(SUBCATEGORIES),
            "priority": random.choice(PRIORITIES),
            "disposition": random.choice(DISPOSITIONS),
            "sentiment": random.choice(SENTIMENTS),
            "keywords": random.sample(KEYWORDS_POOL, k=3),
            "tags": random.sample(TAGS_POOL, k=2),
            "confidence": round(random.uniform(0.55, 0.99), 2),
            "grounding": "handbook_chunk_ref",
            "escalation_recommended": folder == "escalations",
        },
        "decision": {"decision": DECISIONS[folder], "reasons": ["synthetic"]},
        "review": {"status": FOLDERS[folder]},
        "timestamp": ts.isoformat(),
    }


def make_evaluation(call_ids: list[str]) -> dict:
    per_call = []
    for cid in call_ids:
        per_call.append(
            {
                "call_id": cid,
                "overall_match": random.random() > 0.15,
                "category_match": random.random() > 0.1,
                "priority_match": random.random() > 0.1,
                "sentiment_match": random.random() > 0.1,
                "decision_match": random.random() > 0.1,
                "summary_similarity": round(random.uniform(0.6, 1.0), 3),
            }
        )
    return {
        "overall_accuracy": 0.87,
        "category_accuracy": 0.9,
        "priority_accuracy": 0.85,
        "disposition_accuracy": 0.88,
        "sentiment_accuracy": 0.91,
        "escalation_accuracy": 0.93,
        "decision_accuracy": 0.89,
        "per_call_comparison": per_call,
    }


def main():
    call_ids = []
    idx = 1
    for folder in FOLDERS:
        (BASE / folder).mkdir(parents=True, exist_ok=True)
        for _ in range(5):
            call_id = f"K-{idx:03d}"
            call_ids.append(call_id)
            record = make_record(call_id, folder, day_offset=idx % 6)
            (BASE / folder / f"{call_id}.json").write_text(json.dumps(record, indent=2))
            idx += 1

    (BASE / "evaluation").mkdir(parents=True, exist_ok=True)
    evaluation = make_evaluation(call_ids)
    (BASE / "evaluation" / "evaluation_report.json").write_text(json.dumps(evaluation, indent=2))

    print(f"Generated {len(call_ids)} synthetic records under {BASE}")


if __name__ == "__main__":
    main()