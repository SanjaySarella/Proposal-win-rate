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
 useEffect(()=>{fetch('/metadata').then(r=>{if(!r.ok)throw Error('Award data is currently unavailable.');return r.json()}).then(setMeta).catch(e=>setError(e.message));},[]);
 const change=(key,value)=>{setForm({...form,[key]:value});setResult(null);};
 async function run(e){e.preventDefault();setBusy(true);setResult(null);setError('');try{const r=await fetch('/research',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...form,agency:form.agency||null,award_type:form.award_type||null})});const d=await r.json();if(!r.ok)throw Error(typeof d.detail==='string'?d.detail:'Review the search criteria and try again.');setResult(d);}catch(e){setError(e.message);}finally{setBusy(false);}}
 return <main>
 <header><p className="eyebrow">SANJAY SARELLA · PROCUREMENT ANALYTICS</p><h1>Federal Award Research</h1><p>Turn a traceable federal award snapshot into focused agency, supplier, and comparable-award intelligence.</p></header>
 {meta&&<aside><strong>Research coverage: {meta.records.toLocaleString()} award records.</strong> {meta.scope}</aside>}
 <section className="workspace"><div className="section-intro"><h2>Build an award research view</h2><p>Choose the market segment and value range you want to examine. Every result links back to its underlying USAspending award record.</p></div>
 <form onSubmit={run}>
 <label>Funding agency<select value={form.agency} onChange={e=>change('agency',e.target.value)}><option value="">All agencies</option>{meta?.agencies.map(x=><option key={x}>{x}</option>)}</select></label>
 <label>Contract award type<select value={form.award_type} onChange={e=>change('award_type',e.target.value)}><option value="">All contract types</option>{meta?.types.map(x=><option key={x}>{x}</option>)}</select></label>
 <label>Minimum award value<input required aria-label="Minimum award value" type="number" min="0" step="0.01" value={form.minimum} onChange={e=>change('minimum',Number(e.target.value))}/></label>
 <label>Maximum award value<input required aria-label="Maximum award value" type="number" min="0" step="0.01" value={form.maximum} onChange={e=>change('maximum',Number(e.target.value))}/></label>
 <label>Award description keyword<input maxLength="200" placeholder="Examples: cybersecurity, logistics, research" value={form.keyword} onChange={e=>change('keyword',e.target.value)}/></label>
 <button className="primary" disabled={busy||!meta}>{busy?'Building analysis…':'Run award analysis'}</button></form></section>
 {error&&<p role="alert">{error}</p>}
 {result&&<><section className="cards"><article><span>Awards in selection</span><h2>{result.count.toLocaleString()}</h2></article><article><span>Combined recorded value</span><h2>{money(result.total)}</h2></article><article><span>Median award value</span><h2>{money(result.median)}</h2></article></section>
 <p className="method">{result.method}</p>{!result.count?<aside>No award records match the current criteria. Adjust the filters to broaden the research view; this result does not imply that comparable awards are absent from the wider federal market.</aside>:<>
 <section><h2>Agency distribution</h2><p className="section-note">Recorded value represented by each agency within the current selection.</p><table><thead><tr><th>Agency</th><th>Recorded value</th></tr></thead><tbody>{result.agencies.map(r=><tr key={r.Agency}><td>{r.Agency}</td><td>{money(r.Amount)}</td></tr>)}</tbody></table></section>
 <section><h2>Supplier concentration</h2><p>Largest recipient represents {(result.largest_recipient_share*100).toFixed(1)}% of the selected award value. This measures concentration within this snapshot, not federal market share.</p><table><thead><tr><th>Recipient</th><th>Award records</th><th>Recorded value</th></tr></thead><tbody>{result.recipients.slice(0,10).map(r=><tr key={r.Recipient}><td>{r.Recipient}</td><td>{r.Awards}</td><td>{money(r.Amount)}</td></tr>)}</tbody></table><p className="section-note">Ten recipients with the highest recorded value are shown.</p></section>
 <section><div className="section-heading"><div><h2>Comparable award records</h2><p className="section-note">The first 50 records are ordered by award value. Open an award ID to review the current source record.</p></div><button className="secondary" onClick={()=>download(result.records)}>Export complete selection</button></div>
 {result.records.slice(0,50).map(r=><article className="award" key={r.source_url}><h3><a href={r.source_url} target="_blank" rel="noreferrer">{r['Award ID']}</a><span>{money(r['Award Amount'])}</span></h3><p className="award-meta">{r['Recipient Name']} · {r['Contract Award Type']}</p><p>{r.Description}</p><small>Period of performance: {r['Start Date']} to {r['End Date']}</small></article>)}</section></>}</>}
 <footer><strong>Data source:</strong> USAspending award snapshot. The application supports descriptive award research and source verification; it does not estimate bid-win probability or imply savings.<details><summary>Dataset identity and reproducibility</summary><code>{meta?.sha256}</code></details></footer></main>;
}
