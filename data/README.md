# Kalam CX — After-Call Automation · Data

Everything the product works from. Plain files, no database.

## `handbook/`  — the knowledge the product grounds in (RAG corpus)
40 clauses across 7 files: the documentation standard and the golden rule (document only
what was said), the category & subcategory taxonomy, disposition & priority definitions,
the sentiment rubric, escalation & sensitive-call rules, grounding/QA rules, and per-client
industry notes. Deliberately too long to paste in one go, with tricky pairs (billing dispute
vs refund request, complaint vs escalation, resolved vs unresolved, high vs urgent) so the
product must retrieve the right rule, not guess.

## `customers.json`  — the CRM lookup (150 records)
Each record: customer_id, name, client account (which Kalam client/industry), tier, phone,
and open_tickets (prior issues with status/date). `lookup_customer` reads from here to link a
call to a customer, pull their history, and detect repeat contacts.

## `transcripts/`  — the calls (27)
Each call as a plain speaker-labelled transcript (Call ID, Customer ID, channel, date, then
the Agent/Customer dialogue). These are the product's input and they seed the test set. The
set spans every category, all four dispositions, every sentiment, and each decision the
product must make: auto-save, route-to-review, escalate, and non-interaction.

## `answer_key.json`  — ground truth (instructor copy)
For each call: the expected category/subcategory, disposition, sentiment, priority, escalation
flag, the save/route/escalate decision, the tools a correct product would use, and the handbook
clause that justifies it.

Decisions: **AUTO_SAVE** (complete, grounded, routine, non-sensitive) · **ROUTE_TO_REVIEW**
(incomplete, ambiguous, sensitive, "Other", or unverified identity) · **ESCALATE** (an
escalation trigger is present) · **NON_INTERACTION** (no genuine customer request).
