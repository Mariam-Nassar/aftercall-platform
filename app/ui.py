"""
app/ui.py
Human-in-the-loop Review Screen + Operational Summary Panel
for the After-Call Automation Platform.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
RECORDS_DIR = OUTPUTS / "records"
REVIEWS_DIR = OUTPUTS / "reviews"
ESCALATIONS_DIR = OUTPUTS / "escalations"
ARCHIVE_DIR = OUTPUTS / "archive"

FOLDERS = {
    "AUTO_SAVE": RECORDS_DIR,
    "HUMAN_REVIEW": REVIEWS_DIR,
    "ESCALATE": ESCALATIONS_DIR,
    "NON_INTERACTION": ARCHIVE_DIR,
}

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def list_queue(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(folder.glob("*.json"))


def move_record(src: Path, dest_folder: Path) -> Path:
    dest_folder.mkdir(parents=True, exist_ok=True)
    dest = dest_folder / src.name
    data = load_json(src)
    if data is None:
        raise ValueError(f"Cannot read {src}")
    save_json(dest, data)
    src.unlink(missing_ok=True)
    return dest


def get_doc(record: Dict) -> Dict:
    return record.get("documentation") or record.get("doc") or {}


def get_decision(record: Dict) -> str:
    d = record.get("decision")
    if isinstance(d, dict):
        return d.get("decision", "UNKNOWN")
    return str(d or "UNKNOWN")


# ─────────────────────────────────────────────────────────────
# Operational Summary Panel
# ─────────────────────────────────────────────────────────────
def render_operational_summary():
    st.subheader("📊 Operational Summary")

    counts = {
        "AUTO_SAVE": len(list_queue(RECORDS_DIR)),
        "HUMAN_REVIEW": len(list_queue(REVIEWS_DIR)),
        "ESCALATE": len(list_queue(ESCALATIONS_DIR)),
        "NON_INTERACTION": len(list_queue(ARCHIVE_DIR)),
    }
    total = sum(counts.values())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total processed", total)
    c2.metric("✅ Auto-saved", counts["AUTO_SAVE"])
    c3.metric("👤 Human review", counts["HUMAN_REVIEW"])
    c4.metric("🚨 Escalated", counts["ESCALATE"])
    c5.metric("📦 Non-interaction", counts["NON_INTERACTION"])

    # Breakdown by category & sentiment
    all_records = []
    for folder in FOLDERS.values():
        for p in list_queue(folder):
            rec = load_json(p)
            if rec:
                all_records.append(rec)

    if not all_records:
        st.info("No records yet. Run the pipeline first.")
        return

    cats: Dict[str, int] = {}
    sents: Dict[str, int] = {}
    for r in all_records:
        doc = get_doc(r)
        cat = doc.get("category") or "Unknown"
        sent = doc.get("sentiment") or "Unknown"
        cats[cat] = cats.get(cat, 0) + 1
        sents[sent] = sents.get(sent, 0) + 1

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**By Category**")
        st.bar_chart(cats)
    with col_b:
        st.markdown("**By Sentiment**")
        st.bar_chart(sents)


# ─────────────────────────────────────────────────────────────
# Review Queue
# ─────────────────────────────────────────────────────────────
def render_review_queue():
    st.subheader("👤 Human Review Queue")

    review_files = list_queue(REVIEWS_DIR)
    escalate_files = list_queue(ESCALATIONS_DIR)

    if not review_files and not escalate_files:
        st.success("Queue is empty — nothing waiting for human review.")
        return

    # Combine with source tag
    items = [("HUMAN_REVIEW", p) for p in review_files] + [
        ("ESCALATE", p) for p in escalate_files
    ]

    st.write(f"**{len(items)}** item(s) waiting")

    # Select which call to review
    options = [f"{src} · {p.stem}" for src, p in items]
    choice = st.selectbox("Select a call", options, key="queue_select")
    if not choice:
        return

    idx = options.index(choice)
    source, path = items[idx]
    record = load_json(path)
    if record is None:
        st.error("Failed to load record.")
        return

    doc = get_doc(record)
    decision = get_decision(record)

    st.divider()
    st.markdown(f"### Call `{path.stem}`  ·  Decision: **{decision}**")

    # ── Draft view (read-only side) ──
    with st.expander("📄 Full Draft (read-only)", expanded=True):
        st.json(doc)

    # ── Editable form ──
    st.markdown("#### Edit fields (optional)")

    with st.form(key=f"edit_form_{path.stem}"):
        summary = st.text_area("Summary", value=doc.get("summary", ""), height=100)
        issue = st.text_area("Issue", value=doc.get("issue", ""), height=80)
        root_cause = st.text_area("Root cause", value=doc.get("root_cause", ""), height=80)
        resolution = st.text_area("Resolution / Pending", value=doc.get("resolution", ""), height=80)

        col1, col2, col3 = st.columns(3)
        with col1:
            category = st.text_input("Category", value=doc.get("category", ""))
            subcategory = st.text_input("Subcategory", value=doc.get("subcategory", ""))
        with col2:
            priority = st.selectbox(
                "Priority",
                ["Low", "Medium", "High", "Urgent"],
                index=["Low", "Medium", "High", "Urgent"].index(doc.get("priority", "Medium"))
                if doc.get("priority") in ["Low", "Medium", "High", "Urgent"]
                else 1,
            )
            disposition = st.selectbox(
                "Disposition",
                ["Resolved", "Unresolved", "Pending", "Escalated"],
                index=["Resolved", "Unresolved", "Pending", "Escalated"].index(
                    doc.get("disposition", "Pending")
                )
                if doc.get("disposition") in ["Resolved", "Unresolved", "Pending", "Escalated"]
                else 2,
            )
        with col3:
            sentiment = st.selectbox(
                "Sentiment",
                ["Positive", "Neutral", "Negative", "Frustrated"],
                index=["Positive", "Neutral", "Negative", "Frustrated"].index(
                    doc.get("sentiment", "Neutral")
                )
                if doc.get("sentiment") in ["Positive", "Neutral", "Negative", "Frustrated"]
                else 1,
            )
            escalation_flag = st.checkbox(
                "Escalation recommended",
                value=bool(doc.get("escalation_recommended", False)),
            )

        keywords = st.text_input(
            "Keywords (comma-separated)",
            value=", ".join(doc.get("keywords") or []),
        )
        tags = st.text_input(
            "Tags (comma-separated)",
            value=", ".join(doc.get("tags") or []),
        )

        st.markdown("---")
        st.markdown("**Action**")
        action = st.radio(
            "What do you want to do?",
            [
                "✅ Approve & Save (move to records)",
                "✏️ Save edits only (keep in current queue)",
                "🚨 Override → Escalate",
                "📦 Override → Archive (Non-interaction)",
                "🗑️ Discard (delete)",
            ],
            index=0,
        )

        submitted = st.form_submit_button("Apply action", type="primary")

        if submitted:
            # Update documentation fields
            updated_doc = dict(doc)
            updated_doc.update(
                {
                    "summary": summary.strip(),
                    "issue": issue.strip(),
                    "root_cause": root_cause.strip(),
                    "resolution": resolution.strip(),
                    "category": category.strip(),
                    "subcategory": subcategory.strip(),
                    "priority": priority,
                    "disposition": disposition,
                    "sentiment": sentiment,
                    "escalation_recommended": escalation_flag,
                    "keywords": [k.strip() for k in keywords.split(",") if k.strip()],
                    "tags": [t.strip() for t in tags.split(",") if t.strip()],
                }
            )
            record["documentation"] = updated_doc

            # Also keep a review metadata stamp
            record.setdefault("review", {})
            record["review"]["human_action"] = action
            record["review"]["reviewed"] = True

            if action.startswith("✅ Approve"):
                save_json(path, record)  # write edits first
                move_record(path, RECORDS_DIR)
                st.success(f"Approved & saved → records/{path.name}")
                st.rerun()

            elif action.startswith("✏️ Save edits"):
                save_json(path, record)
                st.success("Edits saved (still in queue).")
                st.rerun()

            elif action.startswith("🚨 Override → Escalate"):
                record["decision"] = {"decision": "ESCALATE", "reasons": ["Human override"]}
                save_json(path, record)
                move_record(path, ESCALATIONS_DIR)
                st.warning("Moved to escalations/")
                st.rerun()

            elif action.startswith("📦 Override → Archive"):
                record["decision"] = {
                    "decision": "NON_INTERACTION",
                    "reasons": ["Human override"],
                }
                save_json(path, record)
                move_record(path, ARCHIVE_DIR)
                st.info("Moved to archive/")
                st.rerun()

            elif action.startswith("🗑️ Discard"):
                path.unlink(missing_ok=True)
                st.error("Record discarded.")
                st.rerun()


# ─────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Kalam CX · After-Call Review",
        page_icon="📞",
        layout="wide",
    )

    st.title("📞 Kalam CX — After-Call Automation")
    st.caption("Human Review Screen + Operational Summary")

    tab1, tab2 = st.tabs(["👤 Review Queue", "📊 Operational Summary"])

    with tab1:
        render_review_queue()

    with tab2:
        render_operational_summary()


if __name__ == "__main__":
    main()