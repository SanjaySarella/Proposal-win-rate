# Federal Award Research

Procurement research by Sanjay Sarella. Three outputs: agency totals in a selected historical sample, recipient concentration, and matching awards with USAspending links and CSV export.

## Why the scope changed

The old label `won_definitive` exactly separated 318 definitive contracts from 651 delivery orders; both are awarded contracts. It did not describe winning and losing bids. The supported runtime therefore has no bid-win classifier, SHAP win drivers, AI strategy brief or savings claims. `/predict` returns HTTP 410. Historical models and scripts are retained for provenance only and are not loaded by this application.

## Data and interpretation

`data/processed/contracts_features.csv` contains 969 unique award identifiers. The collection script requests 2022–2024 award activity, award amounts $100,000–$50 million, ordered by descending amount and limited to ten pages. Acquisition timestamp is unknown. This is a selected historical snapshot, not a representative or complete federal market. Award start dates may precede the activity window. Sample composition does not establish market opportunity or award seasonality.

The app sums the recorded `Award Amount` once per unique award. This is neither annual expenditure nor cash paid. Recipient groups use UEI where available and name otherwise. Concentration is relative to the filtered sample. Source links may show updated records. The app exposes a SHA-256 fingerprint for reproducibility and explicitly handles empty selections. Missing/duplicate IDs or invalid amounts stop analysis.

## Run

```sh
python -m pip install -r backend/requirements.txt pytest httpx
cd frontend
npm ci --legacy-peer-deps
npm run build
cd ..
cp -r frontend/build static
python -m uvicorn backend.main:app --port 8080
```

`python -m pytest tests -q` checks independent Decimal reconciliation, filtering, empty results, literal text matching, input validation and retirement of prediction. Frontend tests exercise success and error rendering. Build and test results accompany the PR.

## Deployment boundary

Candidate implementation, not deployed. Preserve the existing `federal-contract-app` Cloud Run service and its URL. Confirm existing project, region, configuration and rollback revision first. No cloud CLI/session is available in the current build environment; a staging revision, container startup and live-browser acceptance are still required.
