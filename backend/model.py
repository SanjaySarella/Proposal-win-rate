import pickle, math, os, datetime
import pandas as pd

FEATURE_NAMES = [
    "log_award_amount", "is_dod", "contract_duration_days",
    "recipient_experience", "agency_activity",
    "is_dc", "award_year", "is_q4", "amount_bucket_encoded",
]

DISPLAY_NAMES = {
    "log_award_amount":       "Award Amount",
    "is_dod":                 "DoD Agency",
    "contract_duration_days": "Contract Duration",
    "recipient_experience":   "Vendor Experience",
    "agency_activity":        "Agency Activity",
    "is_dc":                  "DC Location",
    "award_year":             "Award Year",
    "is_q4":                  "Q4 Submission",
    "amount_bucket_encoded":  "Amount Bucket",
}

class ModelManager:
    def __init__(self):
        self.model     = None
        self.explainer = None
        self.roc_auc   = 0.74

    def load(self):
        base = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(base, "models", "contract_model.pkl"), "rb") as f:
            self.model = pickle.load(f)
        with open(os.path.join(base, "models", "shap_explainer.pkl"), "rb") as f:
            self.explainer = pickle.load(f)

    def predict(self, inp):
        features = self._features(inp)
        df = pd.DataFrame([features], columns=FEATURE_NAMES)
        proba = self.model.predict_proba(df)[0][1]
        pct   = int(round(proba * 100))

        sv = self.explainer.shap_values(df)
        sv = sv[1][0] if isinstance(sv, list) else sv[0]

        shap_list = sorted(
            [{"name": DISPLAY_NAMES.get(n, n), "value": round(float(v), 3)}
             for n, v in zip(FEATURE_NAMES, sv)],
            key=lambda x: abs(x["value"]), reverse=True
        )[:6]

        return pct, shap_list

    def _features(self, d):
        amt    = int(d.amount)
        bucket = (0 if amt < 500_000 else
                  1 if amt < 2_000_000 else
                  2 if amt < 5_000_000 else 3)
        return [
            math.log1p(amt),
            1 if d.isdod else 0,
            int(d.duration),
            int(d.experience),
            7.5,
            1 if d.isdc else 0,
            datetime.datetime.now().year,
            1 if d.isq4 else 0,
            bucket,
        ]
