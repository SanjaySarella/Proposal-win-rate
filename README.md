# Federal Award Record Research

A bounded public-data research case study for filtering, reconciling, and verifying federal award records. The application helps a user identify comparable records by agency, award type, value range, or description keyword and follow each result to its USAspending source page.

## Dataset scope

The repository contains 969 unique award records from a stored USAspending extraction. The extraction is amount-ranked and concentrated near its $50 million upper bound. It is therefore an **upper-value research snapshot**, not a representative sample of the federal market. The application makes that limitation visible and does not interpret snapshot totals as annual spending or market share.

## Workflow

1. Validate internal identifiers and award amounts.
2. Filter records using explicit, literal criteria.
3. Reconcile record count and total recorded value.
4. Summarize agency distribution, award-value bands, and recipient concentration within the selection.
5. Return comparable award records with direct source links.
6. Export the full filtered selection for further review.

## Decision supported

The project supports a research decision: which records in this bounded snapshot are comparable to a stated agency, award type, value range, or description? It does not predict bid outcomes, estimate the complete federal market, infer competitive position, or calculate savings.

## Validation

- Internal award identifiers must be complete and unique.
- Award amounts must be finite and nonnegative.
- Agency totals, recipient counts, and value-band counts reconcile to the filtered selection.
- Keyword filtering is literal rather than regular-expression based.
- Automated acceptance checks independently reconcile the source and API outputs.

## Tools

Python, pandas, FastAPI, React, pytest, Docker, Google Cloud Run. Portable SQL for the quality and reconciliation workflow is documented in `sql/analysis.sql`.

## Run locally

```bash
pip install -r requirements.txt
pytest -q
python validation/acceptance.py
```

## Live application

https://federal-contract-app-646m5mi6fq-uc.a.run.app/
