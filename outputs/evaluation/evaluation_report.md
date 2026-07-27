# Exology Pioneer Program — After-Call Automation Platform — Evaluation Report

**Evaluation Date:** 2026-07-27T21:35:39
**Dataset Size:** 27 calls
**Overall Accuracy:** 0.00%

## Metric Summary

| Metric | Value |
|---|---|
| Total Calls | 27 |
| Correct Predictions | 0 |
| Incorrect Predictions | 27 |
| Overall Accuracy | 0.00% |
| Summary Similarity Score | 0.00% |
| Average Confidence | 72.80% |

## Per-Field Accuracy

| Field | Correct | Incorrect | Total | Accuracy |
|---|---|---|---|---|
| summary | 0 | 27 | 27 | 0.00% |
| category | 0 | 27 | 27 | 0.00% |
| subcategory | 0 | 27 | 27 | 0.00% |
| priority | 2 | 25 | 27 | 7.41% |
| disposition | 3 | 24 | 27 | 11.11% |
| sentiment | 3 | 24 | 27 | 11.11% |
| escalation_recommended | 0 | 27 | 27 | 0.00% |
| decision | 0 | 27 | 27 | 0.00% |

## Failed Calls (27)

| Call ID | Result | Mismatched Fields |
|---|---|---|
| K-001 | INCORRECT | summary, category, subcategory, priority, escalation_recommended, decision |
| K-002 | INCORRECT | summary, category, subcategory, sentiment, escalation_recommended, decision |
| K-003 | INCORRECT | summary, category, subcategory, priority, disposition, escalation_recommended, decision |
| K-004 | INCORRECT | summary, category, subcategory, priority, sentiment, escalation_recommended, decision |
| K-005 | INCORRECT | summary, category, subcategory, disposition, escalation_recommended, decision |
| K-006 | MISSING_GENERATED | summary, category, subcategory, priority, disposition, sentiment, escalation_recommended, decision |
| K-007 | MISSING_GENERATED | summary, category, subcategory, priority, disposition, sentiment, escalation_recommended, decision |
| K-008 | MISSING_GENERATED | summary, category, subcategory, priority, disposition, sentiment, escalation_recommended, decision |
| K-009 | MISSING_GENERATED | summary, category, subcategory, priority, disposition, sentiment, escalation_recommended, decision |
| K-010 | MISSING_GENERATED | summary, category, subcategory, priority, disposition, sentiment, escalation_recommended, decision |
| K-011 | MISSING_GENERATED | summary, category, subcategory, priority, disposition, sentiment, escalation_recommended, decision |
| K-012 | MISSING_GENERATED | summary, category, subcategory, priority, disposition, sentiment, escalation_recommended, decision |
| K-013 | MISSING_GENERATED | summary, category, subcategory, priority, disposition, sentiment, escalation_recommended, decision |
| K-014 | MISSING_GENERATED | summary, category, subcategory, priority, disposition, sentiment, escalation_recommended, decision |
| K-015 | MISSING_GENERATED | summary, category, subcategory, priority, disposition, sentiment, escalation_recommended, decision |
| K-016 | MISSING_GENERATED | summary, category, subcategory, priority, disposition, sentiment, escalation_recommended, decision |
| K-017 | MISSING_GENERATED | summary, category, subcategory, priority, disposition, sentiment, escalation_recommended, decision |
| K-018 | MISSING_GENERATED | summary, category, subcategory, priority, disposition, sentiment, escalation_recommended, decision |
| K-019 | MISSING_GENERATED | summary, category, subcategory, priority, disposition, sentiment, escalation_recommended, decision |
| K-020 | MISSING_GENERATED | summary, category, subcategory, priority, disposition, sentiment, escalation_recommended, decision |
| K-021 | MISSING_GENERATED | summary, category, subcategory, priority, disposition, sentiment, escalation_recommended, decision |
| K-022 | MISSING_GENERATED | summary, category, subcategory, priority, disposition, sentiment, escalation_recommended, decision |
| K-023 | MISSING_GENERATED | summary, category, subcategory, priority, disposition, sentiment, escalation_recommended, decision |
| K-024 | MISSING_GENERATED | summary, category, subcategory, priority, disposition, sentiment, escalation_recommended, decision |
| K-025 | MISSING_GENERATED | summary, category, subcategory, priority, disposition, sentiment, escalation_recommended, decision |
| K-026 | MISSING_GENERATED | summary, category, subcategory, priority, disposition, sentiment, escalation_recommended, decision |
| K-027 | MISSING_GENERATED | summary, category, subcategory, priority, disposition, sentiment, escalation_recommended, decision |

## Recommendations

- Overall accuracy is below 80%. Review documentation prompts and handbook retrieval quality before promoting this pipeline configuration.
- Field 'summary' has low accuracy (0.0%). Investigate prompt grounding and handbook context for this field.
- Field 'category' has low accuracy (0.0%). Investigate prompt grounding and handbook context for this field.
- Field 'subcategory' has low accuracy (0.0%). Investigate prompt grounding and handbook context for this field.
- Field 'priority' has low accuracy (7.4%). Investigate prompt grounding and handbook context for this field.
- Field 'disposition' has low accuracy (11.1%). Investigate prompt grounding and handbook context for this field.
- Field 'sentiment' has low accuracy (11.1%). Investigate prompt grounding and handbook context for this field.
- Field 'escalation_recommended' has low accuracy (0.0%). Investigate prompt grounding and handbook context for this field.
- Field 'decision' has low accuracy (0.0%). Investigate prompt grounding and handbook context for this field.
- Average summary similarity is below the configured threshold. Consider tightening summarization instructions to the LLM.

## Metadata

- **project**: Exology Pioneer Program — After-Call Automation Platform
- **answer_key_path**: data\answer_key.json
- **generated_records_path**: outputs\records
- **similarity_threshold**: 0.75
- **evaluated_fields**: ['summary', 'category', 'subcategory', 'priority', 'disposition', 'sentiment', 'escalation_recommended', 'decision']
- **similarity_backend**: difflib