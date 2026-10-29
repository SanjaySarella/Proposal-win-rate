"""Descriptive award research; no bid-outcome model."""
from pathlib import Path
from functools import lru_cache
from urllib.parse import quote
import hashlib
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / 'data/processed/contracts_features.csv'

@lru_cache(maxsize=1)
def load_data():
    df = pd.read_csv(DATA, dtype={'Award ID':str, 'generated_internal_id':str})
    key = 'generated_internal_id'
    if df[key].isna().any() or df.duplicated(key).any():
        raise ValueError('Missing or duplicate award identifiers; review the source before use.')
    df['Award Amount'] = pd.to_numeric(df['Award Amount'], errors='coerce')
    if not np.isfinite(df['Award Amount']).all() or (df['Award Amount'] < 0).any():
        raise ValueError('Invalid award amounts in this snapshot.')
    return df

def metadata():
    df = load_data()
    amounts = df['Award Amount']
    return {'records':len(df), 'agencies':sorted(df['Awarding Agency'].dropna().unique()),
            'types': sorted(df['Contract Award Type'].dropna().unique()),
            'sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),
            'quality':{'duplicate_internal_ids':int(df['generated_internal_id'].duplicated().sum()),
                       'missing_award_amounts':int(df['Award Amount'].isna().sum()),
                       'minimum_amount':float(amounts.min()),'median_amount':float(amounts.median()),
                       'maximum_amount':float(amounts.max())},
            'scope':'Bounded upper-value research snapshot, not a representative sample or estimate of the complete federal market. The stored extraction is amount-ranked and contains awards collected under a $100,000–$50 million request range. Acquisition timestamp is unavailable, and award start dates may fall outside 2022–2024.'}

def research(agency=None, award_type=None, minimum=0, maximum=50000000, keyword=''):
    df = load_data()
    selected = df.loc[df['Award Amount'].between(minimum, maximum)].copy()
    if agency:
        selected = selected.loc[selected['Awarding Agency'] == agency]
    if award_type:
        selected = selected.loc[selected['Contract Award Type'] == award_type]
    if keyword.strip():
        selected = selected.loc[selected['Description'].fillna('').str.contains(keyword.strip(), case=False, regex=False)]
    total = float(selected['Award Amount'].sum())
    # Group by UEI where available, otherwise by recipient name; disclose this choice.
    selected['Recipient key'] = selected['Recipient UEI'].fillna(selected['Recipient Name'])
    recipients = selected.groupby('Recipient key', dropna=False).agg(
        Recipient=('Recipient Name','first'), Amount=('Award Amount','sum'), Awards=('Award ID','size')
    ).sort_values('Amount',ascending=False).reset_index(drop=True)
    agency_totals = selected.groupby('Awarding Agency')['Award Amount'].sum().sort_values(ascending=False)
    selected['source_url'] = selected['generated_internal_id'].map(lambda x:'https://www.usaspending.gov/award/' + quote(x, safe=''))
    columns = ['Award ID','Recipient Name','Awarding Agency','Contract Award Type','Award Amount','Start Date','End Date','Description','source_url']
    records = selected.sort_values(['Award Amount','Award ID'],ascending=[False,True])[columns].fillna('').to_dict('records')
    if len(selected):
        bands = pd.cut(selected['Award Amount'], bins=[-np.inf,1_000_000,10_000_000,25_000_000,np.inf],
                       labels=['Under $1M','$1M–$10M','$10M–$25M','$25M+'])
        band_counts = bands.value_counts(sort=False)
        distribution = [{'Band':str(k),'Awards':int(v)} for k,v in band_counts.items()]
    else:
        distribution = []
    return {'count':len(selected), 'total':total,
            'median':float(selected['Award Amount'].median()) if len(selected) else None,
            'largest_recipient_share':float(recipients.Amount.iloc[0]/total) if total and len(recipients) else None,
            'recipients':recipients.to_dict('records'),
            'agencies':[{'Agency':k,'Amount':float(v)} for k,v in agency_totals.items()],
            'distribution':distribution,
            'records':records,
            'method':'Sum of Award Amount once per unique award in the filtered snapshot. Results describe only this amount-ranked snapshot; they are not annual spending, transaction flow, cash paid, or federal market share. Recipient concentration uses UEI, falling back to name. Matching awards are descriptive examples, not evidence of win probability.'}
