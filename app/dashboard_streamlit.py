"""
Kalam CX — Operations Console
Enterprise Streamlit analytics for the After-Call Automation Platform.
Read-only presentation layer over pipeline outputs + evaluation artifacts.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.dashboard import LoadedRecord, load_records

# ── brand ──────────────────────────────────────────────────────────────────
BRAND = "Kalam CX"
PRODUCT = "After-Call Intelligence"
VERSION = "1.0"

OUTPUTS = Path("outputs")
EVAL_PATH = OUTPUTS / "evaluation" / "evaluation_report.json"

DECISION_ORDER = ["AUTO_SAVE", "HUMAN_REVIEW", "ESCALATE", "NON_INTERACTION"]
DECISION_COLOR = {
    "AUTO_SAVE": "#12B76A",
    "HUMAN_REVIEW": "#F79009",
    "ESCALATE": "#F04438",
    "NON_INTERACTION": "#98A2B3",
}
STATUS_COLOR = {
    "READY_TO_SAVE": "#12B76A",
    "PENDING_REVIEW": "#F79009",
    "ESCALATED": "#F04438",
    "ARCHIVED": "#667085",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Segoe UI, sans-serif", color="#344054", size=12),
    margin=dict(l=8, r=8, t=36, b=8),
    height=300,
)


def _css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: Inter, Segoe UI, system-ui, sans-serif; }

.stApp { background: #F2F4F7; }
section[data-testid="stSidebar"] {
    background: #0C111D !important;
    border-right: 1px solid #1F242F;
}
section[data-testid="stSidebar"] * { color: #E4E7EC !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stSlider label { color: #98A2B3 !important; }

.block-container { padding: 1.25rem 1.75rem 2rem; max-width: 1280px; }

/* hero */
.hero {
    background: linear-gradient(135deg, #101828 0%, #1D2939 55%, #0B5FFF 140%);
    border-radius: 16px; padding: 22px 26px; color: #fff;
    margin-bottom: 18px; box-shadow: 0 8px 24px rgba(16,24,40,.18);
}
.hero h1 { margin: 0; font-size: 1.45rem; font-weight: 700; letter-spacing: -.02em; }
.hero p { margin: 6px 0 0; color: #D0D5DD; font-size: .92rem; }
.hero .chips { margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap; }
.chip {
    background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.14);
    border-radius: 999px; padding: 4px 12px; font-size: .75rem; color: #EAECF0;
}

/* metric cards */
div[data-testid="stMetric"] {
    background: #fff; border: 1px solid #EAECF0; border-radius: 14px;
    padding: 14px 16px; box-shadow: 0 1px 2px rgba(16,24,40,.04);
}
div[data-testid="stMetric"] label {
    color: #667085 !important; font-size: .72rem !important;
    text-transform: uppercase; letter-spacing: .04em; font-weight: 600 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #101828 !important; font-weight: 700 !important; font-size: 1.55rem !important;
}

/* section */
.section-label {
    font-size: .7rem; font-weight: 700; letter-spacing: .08em;
    text-transform: uppercase; color: #667085; margin: 8px 0 10px;
}

/* alert strip */
.strip {
    display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px;
}
.strip .item {
    background: #fff; border: 1px solid #EAECF0; border-radius: 10px;
    padding: 8px 14px; font-size: .82rem; color: #344054;
    border-left: 3px solid #0B5FFF;
}
.strip .warn { border-left-color: #F79009; }
.strip .bad { border-left-color: #F04438; }
.strip .ok { border-left-color: #12B76A; }

/* table polish */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
</style>
        """,
        unsafe_allow_html=True,
    )


# ── data ───────────────────────────────────────────────────────────────────


def _decision_value(raw: dict) -> str:
    d = raw.get("decision")
    if isinstance(d, dict):
        return str(d.get("decision") or "UNKNOWN")
    if isinstance(d, str):
        return d
    return "UNKNOWN"


def _status_value(raw: dict, folder: str) -> str:
    r = raw.get("review")
    if isinstance(r, dict) and r.get("status"):
        return str(r["status"])
    return {
        "records": "READY_TO_SAVE",
        "reviews": "PENDING_REVIEW",
        "escalations": "ESCALATED",
        "archive": "ARCHIVED",
    }.get(folder, "UNKNOWN")


@st.cache_data(ttl=20, show_spinner=False)
def load_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[LoadedRecord] = []
    try:
        records = load_records(OUTPUTS)
    except Exception:
        records = []

    rows = []
    for rec in records:
        raw = rec.raw or {}
        doc = raw.get("documentation") if isinstance(raw.get("documentation"), dict) else {}
        meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        decision = raw.get("decision") if isinstance(raw.get("decision"), dict) else {}
        ts = rec.timestamp or meta.get("created_at") or ""
        day, hour = "", ""
        if ts:
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                day, hour = dt.strftime("%Y-%m-%d"), f"{dt.hour:02d}"
            except ValueError:
                pass
        rows.append(
            {
                "call_id": rec.call_id,
                "customer_id": meta.get("customer_id") or raw.get("customer_id") or "—",
                "category": rec.category or "Not stated",
                "subcategory": rec.subcategory or "Not stated",
                "priority": rec.priority or "Not stated",
                "disposition": rec.disposition or "Not stated",
                "sentiment": rec.sentiment or "Not stated",
                "decision": rec.decision or _decision_value(raw),
                "status": rec.review_status or _status_value(raw, rec.source_folder),
                "confidence": rec.confidence,
                "summary": doc.get("summary") or "",
                "keywords": ", ".join(rec.keywords or []),
                "reason": decision.get("reason") or "",
                "rules": ", ".join(decision.get("triggered_rules") or [])
                if isinstance(decision.get("triggered_rules"), list)
                else str(decision.get("triggered_rules") or ""),
                "folder": rec.source_folder,
                "day": day,
                "hour": hour,
                "raw": raw,
            }
        )
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=[
            "call_id", "customer_id", "category", "subcategory", "priority",
            "disposition", "sentiment", "decision", "status", "confidence",
            "summary", "keywords", "reason", "rules", "folder", "day", "hour", "raw",
        ]
    )

    evaluation: dict[str, Any] = {}
    if EVAL_PATH.exists():
        try:
            evaluation = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
        except Exception:
            evaluation = {}
    return df, evaluation


def filter_df(
    df: pd.DataFrame,
    q: str,
    categories: list[str],
    decisions: list[str],
    statuses: list[str],
    conf: tuple[float, float],
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if categories:
        out = out[out["category"].isin(categories)]
    if decisions:
        out = out[out["decision"].isin(decisions)]
    if statuses:
        out = out[out["status"].isin(statuses)]
    lo, hi = conf
    c = out["confidence"]
    out = out[c.isna() | ((c >= lo) & (c <= hi))]
    q = (q or "").strip().lower()
    if q:
        blob = (
            out[["call_id", "customer_id", "category", "subcategory", "decision", "summary", "keywords"]]
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
        )
        out = out[blob.str.contains(q, regex=False)]
    return out.reset_index(drop=True)


# ── charts ─────────────────────────────────────────────────────────────────


def bar_counts(series: pd.Series, title: str, color_map: Optional[dict] = None) -> go.Figure:
    vc = series.fillna("—").value_counts()
    order = [x for x in DECISION_ORDER if x in vc.index] + [x for x in vc.index if x not in DECISION_ORDER]
    vc = vc.reindex(order).dropna()
    colors = [color_map.get(str(i), "#0B5FFF") for i in vc.index] if color_map else "#0B5FFF"
    fig = go.Figure(
        go.Bar(
            x=list(vc.index),
            y=list(vc.values),
            marker_color=colors,
            text=list(vc.values),
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(size=14, color="#101828")),
        yaxis_title=None,
        xaxis_title=None,
    )
    fig.update_yaxes(gridcolor="#F2F4F7", zeroline=False)
    fig.update_xaxes(tickangle=-15)
    return fig


def donut(series: pd.Series, title: str, color_map: Optional[dict] = None) -> go.Figure:
    vc = series.fillna("—").value_counts()
    colors = [color_map.get(str(i), "#0B5FFF") for i in vc.index] if color_map else None
    fig = go.Figure(
        go.Pie(
            labels=list(vc.index),
            values=list(vc.values),
            hole=0.62,
            marker=dict(colors=colors) if colors else None,
            textinfo="label+percent",
            textposition="outside",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(size=14, color="#101828")),
        showlegend=False,
    )
    return fig


def hist_conf(series: pd.Series) -> go.Figure:
    s = series.dropna()
    fig = go.Figure(go.Histogram(x=s, nbinsx=10, marker_color="#0B5FFF", opacity=0.85))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Model confidence", font=dict(size=14, color="#101828")),
        xaxis_title="Confidence",
        yaxis_title="Calls",
    )
    fig.update_yaxes(gridcolor="#F2F4F7", zeroline=False)
    return fig


def field_accuracy_chart(evaluation: dict) -> Optional[go.Figure]:
    fm = evaluation.get("field_metrics")
    if not isinstance(fm, dict) or not fm:
        return None
    names, accs = [], []
    for k, v in fm.items():
        if isinstance(v, dict) and v.get("accuracy") is not None:
            names.append(k)
            accs.append(float(v["accuracy"]))
    if not names:
        return None
    fig = go.Figure(
        go.Bar(
            x=names,
            y=accs,
            marker_color=["#12B76A" if a >= 0.7 else "#F79009" if a >= 0.45 else "#F04438" for a in accs],
            text=[f"{a:.0%}" for a in accs],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Evaluation · field accuracy", font=dict(size=14, color="#101828")),
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1.15],
    )
    fig.update_yaxes(gridcolor="#F2F4F7", zeroline=False)
    return fig


# ── layout sections ────────────────────────────────────────────────────────


def sidebar(df: pd.DataFrame) -> dict:
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding:8px 4px 18px">
              <div style="font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;color:#667085;font-weight:600">Platform</div>
              <div style="font-size:1.25rem;font-weight:700;color:#fff;margin-top:4px">{BRAND}</div>
              <div style="font-size:.82rem;color:#98A2B3">{PRODUCT} · v{VERSION}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        total = len(df)
        pending = int((df["status"] == "PENDING_REVIEW").sum()) if total else 0
        esc = int((df["decision"] == "ESCALATE").sum()) if total else 0
        auto = int((df["decision"] == "AUTO_SAVE").sum()) if total else 0

        st.markdown(
            f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:16px">
              <div style="background:#1F242F;border-radius:10px;padding:10px"><div style="font-size:.65rem;color:#98A2B3">RECORDS</div><div style="font-size:1.2rem;font-weight:700">{total}</div></div>
              <div style="background:#1F242F;border-radius:10px;padding:10px"><div style="font-size:.65rem;color:#98A2B3">AUTO-SAVE</div><div style="font-size:1.2rem;font-weight:700;color:#12B76A">{auto}</div></div>
              <div style="background:#1F242F;border-radius:10px;padding:10px"><div style="font-size:.65rem;color:#98A2B3">PENDING</div><div style="font-size:1.2rem;font-weight:700;color:#F79009">{pending}</div></div>
              <div style="background:#1F242F;border-radius:10px;padding:10px"><div style="font-size:.65rem;color:#98A2B3">ESCALATED</div><div style="font-size:1.2rem;font-weight:700;color:#F04438">{esc}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("↻  Refresh", use_container_width=True):
            load_frame.clear()
            st.rerun()

        st.markdown("---")
        st.caption("FILTERS")
        q = st.text_input("Search", placeholder="Call, category, keyword…", label_visibility="collapsed")
        cats = sorted(df["category"].dropna().unique()) if not df.empty else []
        decs = sorted(df["decision"].dropna().unique()) if not df.empty else []
        sts = sorted(df["status"].dropna().unique()) if not df.empty else []
        categories = st.multiselect("Category", cats)
        decisions = st.multiselect("Decision", decs)
        statuses = st.multiselect("Status", sts)
        conf = st.slider("Confidence range", 0.0, 1.0, (0.0, 1.0), 0.05)

        st.markdown("---")
        st.caption(datetime.now().strftime("%d %b %Y · %H:%M"))
        return {
            "q": q,
            "categories": categories,
            "decisions": decisions,
            "statuses": statuses,
            "conf": conf,
        }


def hero(df: pd.DataFrame, evaluation: dict) -> None:
    avg_c = df["confidence"].dropna().mean() if not df.empty else None
    overall = None
    om = evaluation.get("overall_metrics")
    if isinstance(om, dict) and om.get("overall_accuracy") is not None:
        overall = float(om["overall_accuracy"])
    chips = [
        f"{len(df)} calls in view",
        f"Avg confidence {avg_c:.0%}" if avg_c is not None else "Confidence —",
        f"Eval accuracy {overall:.0%}" if overall is not None else "Eval not loaded",
    ]
    chips_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    st.markdown(
        f"""
        <div class="hero">
          <h1>{PRODUCT}</h1>
          <p>Live operational view of automated after-call documentation, routing, and quality.</p>
          <div class="chips">{chips_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def alerts(df: pd.DataFrame) -> None:
    if df.empty:
        st.markdown('<div class="strip"><div class="item bad">No pipeline outputs found. Run the pipeline first.</div></div>', unsafe_allow_html=True)
        return
    items = []
    pending = int((df["status"] == "PENDING_REVIEW").sum())
    esc = int((df["decision"] == "ESCALATE").sum())
    low = int((df["confidence"].fillna(1) < 0.75).sum())
    if pending:
        items.append(("warn", f"{pending} awaiting human review"))
    if esc:
        items.append(("bad" if esc / len(df) > 0.3 else "warn", f"{esc} escalated"))
    if low:
        items.append(("warn", f"{low} below 0.75 confidence"))
    if not items:
        items.append(("ok", "Queues healthy"))
    html = "".join(f'<div class="item {k}">{v}</div>' for k, v in items)
    st.markdown(f'<div class="strip">{html}</div>', unsafe_allow_html=True)


def kpis(df: pd.DataFrame) -> None:
    n = len(df)
    def pct(mask) -> str:
        return f"{mask.sum() / n:.0%}" if n else "—"

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total", n)
    c2.metric("Auto-save", int((df["decision"] == "AUTO_SAVE").sum()), pct(df["decision"] == "AUTO_SAVE"))
    c3.metric("Review", int((df["decision"] == "HUMAN_REVIEW").sum()), pct(df["decision"] == "HUMAN_REVIEW"))
    c4.metric("Escalate", int((df["decision"] == "ESCALATE").sum()), pct(df["decision"] == "ESCALATE"))
    c5.metric("Non-interact", int((df["decision"] == "NON_INTERACTION").sum()))
    avg = df["confidence"].dropna().mean()
    c6.metric("Avg confidence", f"{avg:.0%}" if pd.notna(avg) else "—")


def page_overview(df: pd.DataFrame, evaluation: dict) -> None:
    hero(df, evaluation)
    alerts(df)
    kpis(df)
    st.markdown('<div class="section-label">Routing & status</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.plotly_chart(bar_counts(df["decision"], "Decision mix", DECISION_COLOR), use_container_width=True)
    with b:
        st.plotly_chart(donut(df["status"], "Review status", STATUS_COLOR), use_container_width=True)


def page_quality(df: pd.DataFrame, evaluation: dict) -> None:
    st.markdown('<div class="section-label">AI quality</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.plotly_chart(hist_conf(df["confidence"]), use_container_width=True)
    with b:
        fig = field_accuracy_chart(evaluation)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run evaluation to populate field accuracy.")

    om = evaluation.get("overall_metrics") if isinstance(evaluation, dict) else None
    if isinstance(om, dict):
        x, y, z = st.columns(3)
        x.metric("Overall accuracy", f"{float(om.get('overall_accuracy') or 0):.1%}")
        y.metric("Summary similarity", f"{float(om.get('summary_similarity_score') or 0):.1%}")
        ac = om.get("average_confidence")
        z.metric("Eval avg confidence", f"{float(ac):.1%}" if ac is not None else "—")

    st.markdown('<div class="section-label">Lowest confidence</div>', unsafe_allow_html=True)
    low = (
        df.dropna(subset=["confidence"])
        .nsmallest(10, "confidence")[["call_id", "category", "decision", "status", "confidence", "summary"]]
    )
    st.dataframe(low, use_container_width=True, hide_index=True, height=280)


def page_workload(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-label">Queues</div>', unsafe_allow_html=True)
    pending = df[df["status"] == "PENDING_REVIEW"]
    esc = df[df["decision"] == "ESCALATE"]
    a, b = st.columns(2)
    with a:
        st.caption(f"Pending review · {len(pending)}")
        st.dataframe(
            pending[["call_id", "category", "priority", "confidence", "summary"]],
            use_container_width=True,
            hide_index=True,
            height=300,
        )
    with b:
        st.caption(f"Escalations · {len(esc)}")
        st.dataframe(
            esc[["call_id", "category", "priority", "rules", "summary"]],
            use_container_width=True,
            hide_index=True,
            height=300,
        )

    st.markdown('<div class="section-label">Contact mix</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(bar_counts(df["category"], "Categories"), use_container_width=True)
    with c2:
        st.plotly_chart(bar_counts(df["sentiment"], "Sentiment"), use_container_width=True)


def page_explorer(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-label">Call explorer</div>', unsafe_allow_html=True)
    show = df[
        [
            "call_id",
            "customer_id",
            "category",
            "subcategory",
            "priority",
            "disposition",
            "sentiment",
            "decision",
            "status",
            "confidence",
        ]
    ]
    st.dataframe(show, use_container_width=True, hide_index=True, height=340)

    export = df.drop(columns=["raw"], errors="ignore")
    c1, c2 = st.columns(2)
    c1.download_button(
        "Download CSV",
        export.to_csv(index=False).encode("utf-8"),
        file_name=f"kalam_export_{datetime.now():%Y%m%d}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    c2.download_button(
        "Download JSON",
        export.to_json(orient="records", force_ascii=False, indent=2),
        file_name=f"kalam_export_{datetime.now():%Y%m%d}.json",
        mime="application/json",
        use_container_width=True,
    )

    st.markdown('<div class="section-label">Record detail</div>', unsafe_allow_html=True)
    if df.empty:
        st.info("No rows in current filter.")
        return
    pick = st.selectbox("Call", df["call_id"].tolist())
    row = df.loc[df["call_id"] == pick].iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Decision", row["decision"])
    m2.metric("Status", row["status"])
    m3.metric("Confidence", f"{row['confidence']:.0%}" if pd.notna(row["confidence"]) else "—")
    m4.metric("Priority", row["priority"])

    st.write(row["summary"] or "_No summary_")
    if row["reason"]:
        st.caption(f"Rule reason · {row['reason']}")
    if row["rules"]:
        st.caption(f"Triggered · {row['rules']}")

    t1, t2 = st.tabs(["Documentation", "Full payload"])
    raw = row["raw"] if isinstance(row["raw"], dict) else {}
    with t1:
        st.json(raw.get("documentation") or {})
    with t2:
        st.json(raw)


# ── main ───────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title=f"{BRAND} · Operations",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _css()

    df_all, evaluation = load_frame()
    f = sidebar(df_all)
    df = filter_df(
        df_all,
        f["q"],
        f["categories"],
        f["decisions"],
        f["statuses"],
        f["conf"],
    )

    overview, quality, workload, explorer = st.tabs(
        ["Overview", "Quality", "Workload", "Explorer"]
    )
    with overview:
        page_overview(df, evaluation)
    with quality:
        page_quality(df, evaluation)
    with workload:
        page_workload(df)
    with explorer:
        page_explorer(df)


if __name__ == "__main__":
    main()
