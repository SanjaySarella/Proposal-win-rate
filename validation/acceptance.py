import sys,csv,json
from decimal import Decimal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.research import research,DATA
with DATA.open() as f:raw=list(csv.DictReader(f))
checks=0;maxdiff=0
for agency in [None]+sorted(set(r['Awarding Agency'] for r in raw)):
 for typ in [None]+sorted(set(r['Contract Award Type'] for r in raw)):
  expected=[r for r in raw if (agency is None or r['Awarding Agency']==agency) and (typ is None or r['Contract Award Type']==typ)]
  result=research(agency=agency,award_type=typ)
  assert result['count']==len(expected)
  exact=sum((Decimal(r['Award Amount']) for r in expected),Decimal(0))
  difference=abs(Decimal(str(result['total']))-exact);maxdiff=max(maxdiff,float(difference))
  assert difference<Decimal('0.005'),(agency,typ,difference)
  assert len(result['records'])==len(expected)
  checks+=1
result={'source_rows':len(raw),'agency_type_combinations_reconciled':checks,'maximum_total_difference_dollars':maxdiff,'duplicate_award_ids':len(raw)-len(set(r['generated_internal_id'] for r in raw)),'scope':'Internal snapshot arithmetic and completeness, not independent confirmation that the historical source records remain current.'}
print(json.dumps(result,indent=2));Path(__file__).with_name('results.json').write_text(json.dumps(result,indent=2)+'\n')
