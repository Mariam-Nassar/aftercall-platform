# Evaluation Write-up — Kalam After-Call Automation Platform

**Date:** 2026-07-28  
**Dataset:** 27 labeled calls (`data/answer_key.json`)  
**Coverage:** 27/27 generated records under `outputs/{records,reviews,escalations,archive}`

## Method

Each transcript was processed end-to-end:

Intake → Customer Lookup → Handbook RAG → Gemini documentation →
deterministic business rules → review routing → JSON storage.

Records were scored field-by-field against the answer key.  
**Overall call accuracy** requires every scored field to match.  
**Field-level accuracy** is the primary quality signal.

Non-interaction calls with `customer_id=unknown` use a placeholder
customer so the pipeline can complete without inventing identity data.

## Results (field-level)

| Field | Approx. Accuracy | Notes |
|-------|------------------|--------|
| summary | ~93% | Optional when answer key has no gold text |
| escalation_recommended | ~81% | Mapped from answer_key `escalate` |
| decision | ~59% | Routing often correct on inspected samples |
| disposition | ~59% | |
| sentiment | ~56% | |
| category | ~44% | Label wording drift vs taxonomy |
| priority | ~37% | Model vs key scale mismatch (e.g. Routine/High) |
| subcategory | ~33% | Finest-grained; most brittle |

Average confidence on generated records: **~95%**.  
Strict overall accuracy remains low because one mismatched field fails the whole call.

## What worked

- Grounded generation with explicit **Not stated** when evidence is missing.
- Routing decisions live in **Python business rules**, not in the LLM prompt.
- Escalations, human review, auto-save, and non-interaction paths all fire and persist.
- Full audit trail as JSON per call (decision, rules, queue, status).
- K-022 correctly classified as **NON_INTERACTION** and archived.

## Gaps

- Priority/subcategory taxonomy alignment with the handbook enums.
- Occasional shortened categories (e.g. "Billing" vs "Billing & Payments").
- K-023 (silent/dropped style non-interaction) routed to review due to incomplete fields rather than `NON_INTERACTION` — safe, but stricter non-interaction detection could help.
- No interactive human review UI yet (next deliverable for the live demo).

## Conclusion

The pipeline satisfies the project’s core constraints: retrieval-before-generation,
code-enforced limits, and auditable outputs. Field-level metrics show usable
disposition, sentiment, and decision behavior; category/priority need tighter
enum constraints in the documentation prompt. Human review remains the correct
path for incomplete or sensitive drafts.