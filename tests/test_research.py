from decimal import Decimal
import csv
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.research import research,load_data,DATA
client=TestClient(app)

def test_source_reconciliation():
    with DATA.open() as f: rows=list(csv.DictReader(f))
    r=research()
    assert r['count']==len(rows)==969
    assert r['total']==pytest.approx(float(sum(Decimal(x['Award Amount']) for x in rows)))
    assert sum(x['Awards'] for x in r['recipients'])==969
    assert sum(x['Amount'] for x in r['agencies'])==pytest.approx(r['total'])
    assert sum(x['Awards'] for x in r['distribution'])==969

def test_filters_and_empty():
    agency=load_data()['Awarding Agency'].iloc[0]
    r=research(agency=agency,minimum=49000000)
    assert r['count']>0
    assert all(x['Awarding Agency']==agency and x['Award Amount']>=49000000 for x in r['records'])
    empty=research(keyword='no-match-983794987')
    assert empty['count']==0 and empty['median'] is None and empty['largest_recipient_share'] is None

def test_literal_search_not_regex():
    assert research(keyword='.*')['count']==0

def test_api_validation_and_retired_model():
    assert client.post('/research',json={'minimum':5,'maximum':1}).status_code==422
    assert client.post('/research',json={'agency':'not a real agency'}).status_code==422
    assert client.post('/research',json={'unknown':1}).status_code==422
    assert client.post('/predict',json={}).status_code==410
    assert client.post('/research',json={}).status_code==200
    metadata=client.get('/metadata').json()
    assert metadata['records']==969
    assert metadata['quality']['duplicate_internal_ids']==0
    assert 'amount-ranked' in metadata['scope']
