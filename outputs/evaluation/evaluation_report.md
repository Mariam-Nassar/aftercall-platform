# Exology Pioneer Program — After-Call Automation Platform — Evaluation Report

**Evaluation Date:** 2026-07-28T14:12:45
**Dataset Size:** 27 calls
**Overall Accuracy:** 3.70%

## Metric Summary

| Metric | Value |
|---|---|
| Total Calls | 27 |
| Correct Predictions | 1 |
| Incorrect Predictions | 26 |
| Overall Accuracy | 3.70% |
| Summary Similarity Score | 100.00% |
| Average Confidence | 95.00% |

## Per-Field Accuracy

| Field | Correct | Incorrect | Total | Accuracy |
|---|---|---|---|---|
| summary | 27 | 0 | 27 | 100.00% |
| category | 12 | 15 | 27 | 44.44% |
| subcategory | 9 | 18 | 27 | 33.33% |
| priority | 10 | 17 | 27 | 37.04% |
| disposition | 18 | 9 | 27 | 66.67% |
| sentiment | 15 | 12 | 27 | 55.56% |
| escalation_recommended | 24 | 3 | 27 | 88.89% |
| decision | 17 | 10 | 27 | 62.96% |

## Failed Calls (26)

| Call ID | Result | Mismatched Fields |
|---|---|---|
| K-001 | INCORRECT | priority |
| K-002 | INCORRECT | category, subcategory, priority, sentiment |
| K-004 | INCORRECT | priority |
| K-005 | INCORRECT | category, subcategory |
| K-006 | INCORRECT | subcategory, sentiment |
| K-007 | INCORRECT | disposition |
| K-008 | INCORRECT | category, subcategory, priority, sentiment |
| K-009 | INCORRECT | category, subcategory, priority, disposition, sentiment |
| K-010 | INCORRECT | priority, disposition, sentiment, decision |
| K-011 | INCORRECT | category, subcategory, priority, disposition, sentiment, escalation_recommended, decision |
| K-012 | INCORRECT | category, subcategory, priority, decision |
| K-013 | INCORRECT | subcategory, disposition, decision |
| K-014 | INCORRECT | category, subcategory, disposition, escalation_recommended, decision |
| K-015 | INCORRECT | category, subcategory, priority, disposition, sentiment |
| K-016 | INCORRECT | subcategory |
| K-017 | INCORRECT | category, subcategory |
| K-018 | INCORRECT | category, subcategory, sentiment |
| K-019 | INCORRECT | priority |
| K-020 | INCORRECT | priority |
| K-021 | INCORRECT | priority, sentiment |
| K-022 | INCORRECT | category, subcategory, priority, sentiment |
| K-023 | INCORRECT | category, subcategory, priority, sentiment, decision |
| K-024 | INCORRECT | category, subcategory, priority, decision |
| K-025 | INCORRECT | priority, sentiment, decision |
| K-026 | INCORRECT | category, subcategory, priority, disposition, escalation_recommended, decision |
| K-027 | INCORRECT | category, subcategory, disposition, decision |

## Recommendations

- Overall accuracy is below 80%. Review documentation prompts and handbook retrieval quality before promoting this pipeline configuration.
- Field 'category' has low accuracy (44.4%). Investigate prompt grounding and handbook context for this field.
- Field 'subcategory' has low accuracy (33.3%). Investigate prompt grounding and handbook context for this field.
- Field 'priority' has low accuracy (37.0%). Investigate prompt grounding and handbook context for this field.
- Field 'disposition' has low accuracy (66.7%). Investigate prompt grounding and handbook context for this field.
- Field 'sentiment' has low accuracy (55.6%). Investigate prompt grounding and handbook context for this field.
- Field 'decision' has low accuracy (63.0%). Investigate prompt grounding and handbook context for this field.

## Metadata

- **project**: Exology Pioneer Program — After-Call Automation Platform
- **answer_key_path**: data\answer_key.json
- **generated_records_path**: outputs
- **similarity_threshold**: 0.75
- **evaluated_fields**: ['summary', 'category', 'subcategory', 'priority', 'disposition', 'sentiment', 'escalation_recommended', 'decision']
- **similarity_backend**: rapidfuzz