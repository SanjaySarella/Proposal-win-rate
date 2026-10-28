import {useEffect,useState} from 'react';
import './style.css';
const money = n => n == null ? 'Unavailable' : new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:2}).format(n);
const download = rows => {
 const keys=Object.keys(rows[0]||{}); const cell=v=>'"'+String(v??'').replaceAll('"','""')+'"';
 const csv=[keys.map(cell).join(','),...rows.map(r=>keys.map(k=>cell(r[k])).join(','))].join('\r\n');
 const url=URL.createObjectURL(new Blob([csv],{type:'text/csv'})); const a=document.createElement('a');a.href=url;a.download='filtered_awards.csv';a.click();URL.revokeObjectURL(url);
};
export default function App(){
 const [meta,setMeta]=useState(null),[result,setResult]=useState(null),[busy,setBusy]=useState(false),[error,setError]=useState('');
 const [form,setForm]=useState({agency:'',award_type:'',minimum:0,maximum:50000000,keyword:''});
 useEffect(()=>{fetch('/metadata').then(r=>{if(!r.ok)throw Error('Cannot load dataset');return r.json()}).then(setMeta).catch(e=>setError(e.message));},[]);
 const change=(key,value)=>{setForm({...form,[key]:value});setResult(null);};
 async function run(e){e.preventDefault();setBusy(true);setResult(null);setError('');try{const r=await fetch('/research',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...form,agency:form.agency||null,award_type:form.award_type||null})});const d=await r.json();if(!r.ok)throw Error(typeof d.detail==='string'?d.detail:'Check the filter values.');setResult(d);}catch(e){setError(e.message);}finally{setBusy(false);}}
 return <main><header><p className="eyebrow">SANJAY SARELLA · PROCUREMENT ANALYTICS</p><h1>Federal Award Research</h1><p>Explore agency activity, recipient concentration, and comparable historical awards.</p></header>
 {meta&&<aside><strong>{meta.records.toLocaleString()} records in this snapshot.</strong> {meta.scope}</aside>}
 <form onSubmit={run}><label>Agency<select value={form.agency} onChange={e=>change('agency',e.target.value)}><option value="">All agencies</option>{meta?.agencies.map(x=><option key={x}>{x}</option>)}</select></label>
 <label>Award type<select value={form.award_type} onChange={e=>change('award_type',e.target.value)}><option value="">All types</option>{meta?.types.map(x=><option key={x}>{x}</option>)}</select></label>
 <label>Minimum award amount<input required type="number" min="0" step="0.01" value={form.minimum} onChange={e=>change('minimum',Number(e.target.value))}/></label>
 <label>Maximum award amount<input required type="number" min="0" step="0.01" value={form.maximum} onChange={e=>change('maximum',Number(e.target.value))}/></label>
 <label>Description contains<input maxLength="200" value={form.keyword} onChange={e=>change('keyword',e.target.value)}/></label>
 <button disabled={busy||!meta}>{busy?'Analyzing…':'Explore awards'}</button></form>
 {error&&<p role="alert">{error}</p>}
 {result&&<><section className="cards"><article><span>Matching awards</span><h2>{result.count}</h2></article><article><span>Total recorded award amount</span><h2>{money(result.total)}</h2></article><article><span>Median award</span><h2>{money(result.median)}</h2></article></section>
 <p>{result.method}</p>{!result.count?<aside>No awards match these filters. This does not mean no such awards exist outside the sample.</aside>:<>
 <section><h2>1. Agency activity in the selection</h2><table><thead><tr><th>Agency</th><th>Recorded amount</th></tr></thead><tbody>{result.agencies.map(r=><tr key={r.Agency}><td>{r.Agency}</td><td>{money(r.Amount)}</td></tr>)}</tbody></table></section>
 <section><h2>2. Recipient concentration</h2><p>Largest recipient represents {(result.largest_recipient_share*100).toFixed(1)}% of selected award value. This is sample concentration, not market share.</p><table><thead><tr><th>Recipient</th><th>Awards</th><th>Recorded amount</th></tr></thead><tbody>{result.recipients.slice(0,10).map(r=><tr key={r.Recipient}><td>{r.Recipient}</td><td>{r.Awards}</td><td>{money(r.Amount)}</td></tr>)}</tbody></table><p>Top 10 recipients shown.</p></section>
 <section><h2>3. Matching historical awards</h2><button onClick={()=>download(result.records)}>Download all matching records</button><p>First 50 shown, ordered by award amount. Open the source to check its current record.</p>{result.records.slice(0,50).map(r=><article key={r.source_url}><h3><a href={r.source_url} target="_blank" rel="noreferrer">{r['Award ID']}</a> · {money(r['Award Amount'])}</h3><p>{r['Recipient Name']} · {r['Contract Award Type']}</p><p>{r.Description}</p><small>{r['Start Date']} to {r['End Date']}</small></article>)}</section></>}</>}
 <footer>Source: USAspending award snapshot. No bid-win model, recommendations to bid, or implied savings.<details><summary>Dataset fingerprint</summary><code>{meta?.sha256}</code></details></footer></main>;
}
