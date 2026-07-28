"""
app/review_screen.py

The human review screen for the AI-Powered After-Call Automation
Platform.

Responsibility (and ONLY responsibility):
    Render, in a browser, the queue of records that business_rules.py
    routed to HUMAN_REVIEW or ESCALATE, let a reviewer approve, edit,
    or override each one, and show a simple operational counts panel.

This module contains NO decision logic of its own. Every action here
is a thin call into:
    - app/human_review.py  (approve / edit / override / list_pending)
    - app/dashboard.py     (run_dashboard, for the counts panel)
Nothing that needs a human is ever saved without this screen (or an
equivalent caller of human_review.py) explicitly acting on it first.

Run with:
    streamlit run app/review_screen.py
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import streamlit as st

from app.business_rules import DecisionType
from app.dashboard import DashboardError, run_dashboard
from app.human_review import (
    AlreadyFinalizedError,
    approve_record,
    edit_record,
    list_pending,
    override_record,
)
from app.review import ReviewStatus
from app.storage import DEFAULT_BASE_DIR, list_records_with_paths, load_record

st.set_page_config(page_title="Kalam CX — After-Call Review", layout="wide")

EDITABLE_TEXT_FIELDS = (
    "summary",
    "issue",
    "root_cause",
    "resolution",
    "category",
    "subcategory",
    "priority",
    "disposition",
    "sentiment",
)
EDITABLE_LIST_FIELDS = ("pending_actions", "keywords", "tags")


def _base_dir() -> Path:
    return Path(st.session_state.get("outputs_dir", str(DEFAULT_BASE_DIR)))


def _status_badge(status: ReviewStatus) -> str:
    colors = {
        ReviewStatus.READY_TO_SAVE: "🟢",
        ReviewStatus.PENDING_REVIEW: "🟡",
        ReviewStatus.ESCALATED: "🔴",
        ReviewStatus.ARCHIVED: "⚪",
    }
    return f"{colors.get(status, '⬜')} {status.value}"


def _render_record_panel(path: Path, result) -> None:
    doc = result.documentation
    decision = result.decision

    left, right = st.columns([3, 2])

    with left:
        st.markdown(f"**Summary:** {doc.summary}")
        st.markdown(f"**Issue:** {doc.issue}")
        st.markdown(f"**Root cause:** {doc.root_cause}")
        st.markdown(f"**Resolution:** {doc.resolution}")
        st.markdown(f"**Pending actions:** {', '.join(doc.pending_actions) or '—'}")
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Category**\n\n{doc.category} / {doc.subcategory}")
        c2.markdown(f"**Priority**\n\n{doc.priority}")
        c3.markdown(f"**Sentiment**\n\n{doc.sentiment}")
        st.markdown(f"**Disposition:** {doc.disposition}")
        st.markdown(f"**Keywords:** {', '.join(doc.keywords) or '—'}")
        st.markdown(f"**Tags:** {', '.join(doc.tags) or '—'}")
        st.caption(f"Model confidence: {doc.confidence:.2f} · Model recommends escalation: {doc.escalation_recommended}")

    with right:
        st.markdown(f"**Routing decision:** `{decision.decision.value}`")
        st.markdown(f"**Why:** {decision.reason}")
        st.markdown(f"**Triggered rules:** {', '.join(decision.triggered_rules)}")
        if result.review_reason:
            st.markdown(f"**Review note:** {result.review_reason}")
        with st.expander("Grounding (field → evidence)"):
            for field_name, evidence in doc.grounding.items():
                st.markdown(f"- **{field_name}**: {', '.join(evidence) if evidence else '_no evidence — should be Not stated_'}")

    st.divider()
    action_tab, edit_tab, override_tab = st.tabs(["✅ Approve", "✏️ Edit", "⚠️ Override"])

    with action_tab:
        note = st.text_input("Optional approval note", key=f"approve_note_{path}")
        if st.button("Approve exactly as drafted and save", key=f"approve_btn_{path}", type="primary"):
            try:
                approve_record(path, reviewer=st.session_state["reviewer"], note=note, base_dir=_base_dir())
                st.success(f"Approved and saved {path.stem}.")
                st.rerun()
            except AlreadyFinalizedError as exc:
                st.error(str(exc))

    with edit_tab:
        with st.form(key=f"edit_form_{path}"):
            new_values: dict[str, object] = {}
            for field_name in EDITABLE_TEXT_FIELDS:
                new_values[field_name] = st.text_input(
                    field_name.replace("_", " ").title(),
                    value=getattr(doc, field_name),
                    key=f"edit_{field_name}_{path}",
                )
            for field_name in EDITABLE_LIST_FIELDS:
                raw = st.text_input(
                    f"{field_name.replace('_', ' ').title()} (comma-separated)",
                    value=", ".join(getattr(doc, field_name)),
                    key=f"edit_{field_name}_{path}",
                )
                new_values[field_name] = [item.strip() for item in raw.split(",") if item.strip()]
            confidence = st.slider("Confidence", 0.0, 1.0, float(doc.confidence), 0.01, key=f"edit_conf_{path}")
            new_values["confidence"] = confidence

            submitted = st.form_submit_button("Save edits and re-evaluate")
            if submitted:
                changed = {
                    key: value
                    for key, value in new_values.items()
                    if value != getattr(doc, key)
                }
                if not changed:
                    st.info("No fields were changed.")
                else:
                    try:
                        new_path = edit_record(
                            path, changed, reviewer=st.session_state["reviewer"], base_dir=_base_dir()
                        )
                        updated = load_record(new_path)
                        st.success(
                            f"Saved edits to {path.stem}. New routing decision: "
                            f"{updated.decision.decision.value} ({updated.status.value})."
                        )
                        st.rerun()
                    except AlreadyFinalizedError as exc:
                        st.error(str(exc))
                    except ValueError as exc:
                        st.error(str(exc))

    with override_tab:
        forced = st.selectbox(
            "Force this record to:",
            options=[d for d in DecisionType],
            format_func=lambda d: d.value,
            key=f"override_select_{path}",
        )
        justification = st.text_area("Justification (required)", key=f"override_note_{path}")
        if st.button("Force decision", key=f"override_btn_{path}"):
            try:
                override_record(
                    path,
                    forced,
                    note=justification,
                    reviewer=st.session_state["reviewer"],
                    base_dir=_base_dir(),
                )
                st.success(f"Overrode {path.stem} to {forced.value}.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
            except AlreadyFinalizedError as exc:
                st.error(str(exc))


def render_review_queue() -> None:
    pending = list_pending(base_dir=_base_dir())

    if not pending:
        st.success("Nothing waiting on a human right now. 🎉")
        return

    st.caption(f"{len(pending)} record(s) waiting for a decision.")
    for path, result in pending:
        header = f"{_status_badge(result.status)}  ·  **{path.stem}**  ·  {result.documentation.category}/{result.documentation.subcategory}  ·  priority {result.documentation.priority}"
        with st.expander(header):
            _render_record_panel(path, result)


def render_saved_and_archived() -> None:
    """Read-only view of records already finalized, for demo purposes."""
    base_dir = _base_dir()
    saved = list_records_with_paths(status=ReviewStatus.READY_TO_SAVE, base_dir=base_dir)
    archived = list_records_with_paths(status=ReviewStatus.ARCHIVED, base_dir=base_dir)

    st.caption(f"{len(saved)} saved · {len(archived)} archived (non-interaction)")
    for path, result in saved:
        with st.expander(f"🟢 {path.stem} — {result.documentation.category}/{result.documentation.subcategory}"):
            st.json(dataclasses.asdict(result.documentation))
    for path, result in archived:
        with st.expander(f"⚪ {path.stem} — archived"):
            st.write(result.decision.reason)


def render_operational_summary() -> None:
    base_dir = _base_dir()
    try:
        report = run_dashboard(base_dir)
    except DashboardError as exc:
        st.warning(f"Could not build the summary yet: {exc}")
        return

    m = report.metrics
    cols = st.columns(5)
    cols[0].metric("Total processed", m.total_calls_processed)
    cols[1].metric("Auto-saved", m.auto_saved)
    cols[2].metric("Pending review", m.pending_human_review)
    cols[3].metric("Escalated", m.escalated_calls)
    cols[4].metric("Non-interaction", m.archived_calls)

    if m.average_confidence is not None:
        st.caption(f"Average model confidence: {m.average_confidence:.2f}")
    if m.overall_accuracy is not None:
        st.caption(f"Evaluation accuracy vs answer key: {m.overall_accuracy:.1%}")

    b1, b2 = st.columns(2)
    with b1:
        st.subheader("By category")
        st.bar_chart(report.statistics.category_distribution)
        st.subheader("By sentiment")
        st.bar_chart(report.statistics.sentiment_distribution)
    with b2:
        st.subheader("By decision")
        st.bar_chart(report.statistics.decision_distribution)
        st.subheader("By priority")
        st.bar_chart(report.statistics.priority_distribution)

    if report.warnings:
        for warning in report.warnings:
            st.caption(f"⚠️ {warning}")


def main() -> None:
    st.title("Kalam CX — After-Call Documentation Review")

    with st.sidebar:
        st.session_state.setdefault("reviewer", "human_reviewer")
        st.session_state["reviewer"] = st.text_input("Reviewer name", value=st.session_state["reviewer"])
        st.session_state["outputs_dir"] = st.text_input("Outputs directory", value=str(DEFAULT_BASE_DIR))
        if st.button("🔄 Refresh"):
            st.rerun()

    tab_queue, tab_saved, tab_summary = st.tabs(
        ["🗂️ Review Queue", "📁 Saved / Archived", "📊 Operational Summary"]
    )
    with tab_queue:
        render_review_queue()
    with tab_saved:
        render_saved_and_archived()
    with tab_summary:
        render_operational_summary()


if __name__ == "__main__":
    main()
