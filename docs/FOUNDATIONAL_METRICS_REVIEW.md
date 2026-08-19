# Foundational Metrics Label Review

This review is the gate that makes a gold case eligible for a measured
foundational quality score. Do not change a case to `reviewed` unless every
field below has been checked by a qualified reviewer.

## Review each case

1. Open the case in `data/retrieval_cases.json` and verify that the canonical
   query and every `query_variants` entry have the same intent.
2. Confirm `language` matches the question. English reference answers and
   claims must be English; Arabic reference answers and claims must be Arabic.
3. For evidence-seeking cases, verify the source, document name, page, and
   chunk ID (when known). Add one `relevant_items` entry for each relevant item
   and assign an integer `relevance_grade` from 1 (useful) to 3 (fully
   supporting).
4. Add accepted `reference_answers`, short `accepted_aliases` where needed,
   and atomic `required_claims`. Do not copy an evidence passage as an answer
   unless it is intentionally accepted as the answer.
5. Choose only deterministic `task_pass_rules` that fit the case. For example,
   an evidence answer may use `expected_status`, `required_claims_present`, and
   `certified_citation_present`; a vague query may use `generation_not_called`
   and `retrieval_not_called`.
6. Set the review record only after the checks above pass:

   ```json
   "review": {
     "status": "reviewed",
     "reviewer_role": "domain_expert",
     "reviewed_at": "2026-08-19T00:00:00Z"
   }
   ```

## Validate before merging

Run the review validator. It writes a queue but never modifies the labels:

```powershell
.\.venv\Scripts\python.exe scripts\review_gold_cases.py
```

Resolve the relevant rows in `reports/gold_case_review.csv`, then run the
foundational test file. A score remains **Not measured** until its data is
reviewed; zero is reserved for a measured failure.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_foundational_metrics.py -q --no-cov -p no:randomly -n 0
```

