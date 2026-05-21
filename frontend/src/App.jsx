import { useState, useEffect } from "react";

const COLORS = {
  navy: "#1A2744",
  navyDeep: "#111C33",
  navyLight: "#243560",
  gold: "#C9A84C",
  parchment: "#F7F5F0",
  parchmentDark: "#EDE9DF",
  parchmentMid: "#F0EDE5",
  border: "#D8D2C4",
  muted: "#8A8070",
  ink: "#1A1F2E",
  mutedDark: "#6B6355",
};

const globalStyles = `
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: ${COLORS.parchment};
    font-family: 'DM Sans', sans-serif;
    color: ${COLORS.ink};
    -webkit-font-smoothing: antialiased;
  }

  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: ${COLORS.parchmentDark}; }
  ::-webkit-scrollbar-thumb { background: ${COLORS.border}; border-radius: 2px; }

  select, input[type=range] { font-family: 'DM Sans', sans-serif; }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .fade-up  { animation: fadeUp  0.45s ease both; }
  .fade-in  { animation: fadeIn  0.4s  ease both; }

  .stagger-row { opacity: 0; animation: fadeUp 0.4s ease forwards; }
`;

// ─── HEADER ────────────────────────────────────────────────────────────────────
function Header({ page, setPage }) {
  return (
    <header style={{
      background: COLORS.navyDeep,
      padding: "0 40px",
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      height: 64,
      borderBottom: "1px solid rgba(201,168,76,0.2)",
      position: "sticky",
      top: 0,
      zIndex: 100,
    }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        <span style={{
          fontFamily: "'DM Sans', sans-serif",
          fontSize: 10,
          letterSpacing: "0.22em",
          color: COLORS.gold,
          textTransform: "uppercase",
          fontWeight: 400,
        }}>
          Sanjay Sarella's
        </span>
        <span style={{
          fontFamily: "'Playfair Display', serif",
          fontSize: 17,
          color: COLORS.parchment,
          fontWeight: 500,
          letterSpacing: "0.01em",
        }}>
          Federal Contract Win Rate Intelligence
        </span>
      </div>

      <nav style={{ display: "flex", gap: 6 }}>
        {["Dashboard", "About"].map(label => (
          <button
            key={label}
            onClick={() => setPage(label)}
            style={{
              background: page === label ? "rgba(201,168,76,0.12)" : "transparent",
              border: page === label ? `0.5px solid ${COLORS.gold}` : "0.5px solid transparent",
              color: page === label ? COLORS.gold : COLORS.muted,
              padding: "6px 18px",
              borderRadius: 3,
              cursor: "pointer",
              fontSize: 12,
              fontFamily: "'DM Sans', sans-serif",
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              fontWeight: 500,
              transition: "all 0.2s",
            }}
          >
            {label}
          </button>
        ))}
      </nav>
    </header>
  );
}

// ─── SIDEBAR ────────────────────────────────────────────────────────────────────
function Sidebar({ inputs, setInputs, onRun, loading }) {
  // Keep isdod checkbox in sync with agency dropdown
  useEffect(() => {
    setInputs(prev => ({
      ...prev,
      isdod: prev.agency === "Dept. of Defense (DoD)",
    }));
  }, [inputs.agency]);

  const Field = ({ label, children }) => (
    <div style={{ marginBottom: 20 }}>
      <label style={{
        display: "block",
        fontSize: 10,
        letterSpacing: "0.14em",
        textTransform: "uppercase",
        color: COLORS.muted,
        marginBottom: 8,
        fontWeight: 500,
      }}>{label}</label>
      {children}
    </div>
  );

  const selectStyle = {
    width: "100%",
    background: COLORS.parchment,
    border: `0.5px solid ${COLORS.border}`,
    borderRadius: 4,
    padding: "9px 12px",
    fontSize: 13,
    fontFamily: "'DM Sans', sans-serif",
    color: COLORS.ink,
    appearance: "none",
    cursor: "pointer",
    outline: "none",
  };

  return (
    <aside style={{
      width: 272,
      minWidth: 272,
      background: COLORS.parchmentMid,
      borderRight: `0.5px solid ${COLORS.border}`,
      padding: "28px 22px",
      display: "flex",
      flexDirection: "column",
    }}>
      <div style={{
        fontSize: 9,
        letterSpacing: "0.18em",
        textTransform: "uppercase",
        color: COLORS.gold,
        marginBottom: 22,
        paddingBottom: 12,
        borderBottom: `0.5px solid ${COLORS.border}`,
        fontWeight: 500,
      }}>
        Contract Parameters
      </div>

      <Field label="Agency">
        <select
          style={selectStyle}
          value={inputs.agency}
          onChange={e => setInputs({ ...inputs, agency: e.target.value })}
        >
          <option>Dept. of Defense (DoD)</option>
          <option>Dept. of Health &amp; Human Services</option>
          <option>General Services Administration</option>
          <option>Dept. of Homeland Security</option>
          <option>Dept. of Energy</option>
          <option>NASA</option>
        </select>
      </Field>

      <Field label={`Award Amount — $${Number(inputs.amount).toLocaleString()}`}>
        <input type="range" min={10000} max={10000000} step={50000}
          value={inputs.amount}
          onChange={e => setInputs({ ...inputs, amount: e.target.value })}
          style={{ width: "100%", accentColor: COLORS.navy }} />
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: COLORS.muted, marginTop: 4 }}>
          <span>$10K</span><span>$10M</span>
        </div>
      </Field>

      <Field label={`Contract Duration — ${inputs.duration} days`}>
        <input type="range" min={30} max={1825} step={30}
          value={inputs.duration}
          onChange={e => setInputs({ ...inputs, duration: e.target.value })}
          style={{ width: "100%", accentColor: COLORS.navy }} />
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: COLORS.muted, marginTop: 4 }}>
          <span>30d</span><span>5 years</span>
        </div>
      </Field>

      <Field label={`Vendor Experience — ${inputs.experience} contracts`}>
        <input type="range" min={0} max={50} step={1}
          value={inputs.experience}
          onChange={e => setInputs({ ...inputs, experience: e.target.value })}
          style={{ width: "100%", accentColor: COLORS.navy }} />
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: COLORS.muted, marginTop: 4 }}>
          <span>0</span><span>50</span>
        </div>
      </Field>

      <div style={{
        fontSize: 9,
        letterSpacing: "0.18em",
        textTransform: "uppercase",
        color: COLORS.gold,
        marginBottom: 16,
        paddingBottom: 10,
        borderBottom: `0.5px solid ${COLORS.border}`,
        fontWeight: 500,
      }}>
        Conditions
      </div>

      {[
        { key: "isdc",  label: "Washington D.C. location" },
        { key: "isq4",  label: "Fiscal year-end (Q4)" },
        { key: "isdod", label: "DoD agency", readOnly: inputs.agency === "Dept. of Defense (DoD)" },
      ].map(({ key, label, readOnly }) => (
        <label key={key} style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          marginBottom: 12,
          cursor: readOnly ? "default" : "pointer",
          fontSize: 13,
          color: COLORS.mutedDark,
          opacity: readOnly ? 0.6 : 1,
        }}>
          <input
            type="checkbox"
            checked={inputs[key]}
            readOnly={readOnly}
            onChange={e => !readOnly && setInputs({ ...inputs, [key]: e.target.checked })}
            style={{ accentColor: COLORS.navy, width: 14, height: 14, cursor: readOnly ? "default" : "pointer" }}
          />
          {label}{readOnly ? " (auto)" : ""}
        </label>
      ))}

      <button
        onClick={onRun}
        disabled={loading}
        style={{
          marginTop: 20,
          background: loading ? COLORS.navyLight : COLORS.navy,
          color: COLORS.parchment,
          border: "none",
          padding: "13px 0",
          borderRadius: 4,
          cursor: loading ? "wait" : "pointer",
          fontSize: 12,
          fontFamily: "'DM Sans', sans-serif",
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          fontWeight: 500,
          transition: "background 0.2s",
          width: "100%",
        }}
      >
        {loading ? "Analyzing…" : "Run Analysis"}
      </button>
    </aside>
  );
}

// ─── WIN PROBABILITY GAUGE ─────────────────────────────────────────────────────
function WinGauge({ pct }) {
  const deg = (pct / 100) * 360;
  const color   = pct >= 60 ? "#3B6D11" : pct >= 40 ? "#BA7517" : "#A32D2D";
  const bgColor = pct >= 60 ? "#EAF3DE" : pct >= 40 ? "#FAEEDA" : "#FCEBEB";
  const label   = pct >= 60 ? "Favorable Outlook" : pct >= 40 ? "Moderate Risk"   : "High Risk";

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "12px 0" }}>
      <div style={{
        width: 148,
        height: 148,
        borderRadius: "50%",
        background: `conic-gradient(${COLORS.navy} 0deg ${deg}deg, #E0DBCF ${deg}deg)`,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}>
        <div style={{
          width: 110,
          height: 110,
          background: COLORS.parchment,
          borderRadius: "50%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
        }}>
          <span style={{ fontFamily: "'Playfair Display', serif", fontSize: 28, fontWeight: 500, color: COLORS.navy, lineHeight: 1 }}>{pct}%</span>
          <span style={{ fontSize: 9, letterSpacing: "0.1em", textTransform: "uppercase", color: COLORS.muted, marginTop: 4 }}>Win Rate</span>
        </div>
      </div>
      <div style={{
        marginTop: 16,
        background: bgColor,
        color: color,
        border: `0.5px solid ${color}`,
        borderRadius: 20,
        padding: "5px 16px",
        fontSize: 11,
        fontWeight: 500,
        letterSpacing: "0.08em",
        textTransform: "uppercase",
      }}>
        {label}
      </div>
    </div>
  );
}

// ─── SHAP CHART ────────────────────────────────────────────────────────────────
function ShapChart({ features }) {
  const max = Math.max(...features.map(f => Math.abs(f.value)));
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {features.map((f, i) => {
        const isPos = f.value >= 0;
        const pct   = (Math.abs(f.value) / max) * 100;
        return (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <span style={{ fontSize: 12, color: COLORS.mutedDark, minWidth: 130, textAlign: "right" }}>{f.name}</span>
            <div style={{ flex: 1, height: 8, background: "#E8E4DA", borderRadius: 4, overflow: "hidden", position: "relative" }}>
              <div style={{
                position: "absolute",
                height: "100%",
                width: `${pct}%`,
                borderRadius: 4,
                background: isPos ? COLORS.navy : COLORS.gold,
                ...(isPos ? { left: 0 } : { right: 0 }),
              }} />
            </div>
            <span style={{ fontSize: 12, fontWeight: 500, minWidth: 40, color: isPos ? COLORS.navy : COLORS.gold }}>
              {isPos ? "+" : ""}{f.value.toFixed(2)}
            </span>
          </div>
        );
      })}
      <div style={{ display: "flex", gap: 20, marginTop: 6, paddingTop: 10, borderTop: `0.5px solid ${COLORS.border}` }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, color: COLORS.muted }}>
          <div style={{ width: 10, height: 10, background: COLORS.navy, borderRadius: 2 }} /> Increases win chance
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, color: COLORS.muted }}>
          <div style={{ width: 10, height: 10, background: COLORS.gold, borderRadius: 2 }} /> Decreases win chance
        </div>
      </div>
    </div>
  );
}

// ─── STRATEGY BRIEF ────────────────────────────────────────────────────────────
function StrategyBrief({ points }) {
  return (
    <div style={{ background: COLORS.navyDeep, borderRadius: 8, padding: "24px 28px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 20 }}>
        <div style={{ width: 6, height: 22, background: COLORS.gold, borderRadius: 2 }} />
        <span style={{ fontFamily: "'Playfair Display', serif", fontSize: 15, color: COLORS.parchment, fontWeight: 500 }}>
          AI Strategy Brief
        </span>
        <span style={{ marginLeft: "auto", fontSize: 9, letterSpacing: "0.14em", textTransform: "uppercase", color: COLORS.gold, opacity: 0.8 }}>
          LangGraph · 4-node pipeline · Groq
        </span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px 32px" }}>
        {points.map((p, i) => (
          <div key={i} style={{ display: "flex", gap: 12 }}>
            <div style={{ width: 5, height: 5, background: COLORS.gold, borderRadius: "50%", marginTop: 7, flexShrink: 0 }} />
            <p style={{ fontSize: 13, color: "#BDB7AD", lineHeight: 1.75, fontWeight: 300 }}>{p}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── HERO BANNER ───────────────────────────────────────────────────────────────
function HeroBanner() {
  return (
    <div style={{
      background: COLORS.navyDeep,
      padding: "32px 40px",
      borderBottom: "1px solid rgba(201,168,76,0.15)",
      display: "flex",
      alignItems: "flex-end",
      justifyContent: "space-between",
      gap: 40,
    }}>
      <div style={{ maxWidth: 560 }}>
        <div style={{
          fontSize: 10,
          letterSpacing: "0.22em",
          textTransform: "uppercase",
          color: COLORS.gold,
          marginBottom: 14,
          fontWeight: 500,
        }}>
          Federal Procurement Intelligence
        </div>
        <h1 style={{
          fontFamily: "'Playfair Display', serif",
          fontSize: 28,
          fontWeight: 500,
          color: COLORS.parchment,
          lineHeight: 1.35,
          marginBottom: 16,
        }}>
          $700 billion in federal contracts is awarded every year.<br />
          Most proposals fail not because of price —{" "}
          <em style={{ color: COLORS.gold }}>but because of positioning.</em>
        </h1>
        <p style={{ fontSize: 14, color: "#ADA79E", lineHeight: 1.75, fontWeight: 300 }}>
          This tool tells you exactly where you stand before you bid. Trained on 969 real
          contracts from USASpending.gov, it predicts your win probability, breaks down
          what's working against you, and delivers a strategy brief — in one click.
        </p>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 12, flexShrink: 0 }}>
        {[
          { n: "969",    l: "Real contracts"  },
          { n: "74%",    l: "Model accuracy"  },
          { n: "4-node", l: "AI pipeline"     },
        ].map(({ n, l }) => (
          <div key={l} style={{ textAlign: "right" }}>
            <div style={{ fontFamily: "'Playfair Display', serif", fontSize: 22, color: COLORS.gold, fontWeight: 500 }}>{n}</div>
            <div style={{ fontSize: 10, color: COLORS.muted, letterSpacing: "0.1em", textTransform: "uppercase" }}>{l}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── DASHBOARD PAGE ────────────────────────────────────────────────────────────
function DashboardPage() {
  const [inputs, setInputs] = useState({
    agency:     "Dept. of Defense (DoD)",
    amount:     2400000,
    duration:   365,
    experience: 12,
    isdc:       true,
    isq4:       true,
    isdod:      true,
  });

  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);

  // ── Replace this block with the real API call when wiring backend ──
  const runAnalysis = async () => {
    setLoading(true);
    setError(null);
    try {
      // REAL API CALL — uncomment and replace mock block below:
      // const data = await predictChurn(inputs);
      // setResult(data);

      // ── MOCK (remove after backend wiring) ──
      await new Promise(r => setTimeout(r, 1400));
      const exp = parseInt(inputs.experience);
      const amt = parseInt(inputs.amount);
      const basePct = 42
        + (exp > 10 ? 18 : exp * 1.5)
        + (inputs.isdod ? 8  : 0)
        + (amt > 1000000 && amt < 5000000 ? 6 : 0)
        + (inputs.isdc  ? 4  : 0)
        - (inputs.isq4  ? 10 : 0);
      const pct = Math.min(92, Math.max(12, Math.round(basePct)));

      setResult({
        pct,
        features: [
          { name: "Vendor Experience",  value:  0.41 },
          { name: "Agency Activity",    value:  0.28 },
          { name: "Award Amount",       value:  0.19 },
          { name: "Contract Duration",  value:  0.11 },
          { name: "DC Location",        value:  inputs.isdc ? 0.07 : -0.04 },
          { name: "Q4 Submission",      value:  inputs.isq4 ? -0.15 : 0.06 },
        ].sort((a, b) => b.value - a.value),
        points: [
          "Vendor experience is the strongest predictor. Firms with 10+ prior contracts with the same agency win at 2.3× the baseline rate.",
          inputs.isq4
            ? "Q4 submissions reduce win odds by roughly 15% due to budget exhaustion. Move this proposal to Q1 if the timeline allows."
            : "Your non-Q4 timing is a genuine advantage — agencies have more discretionary budget and fewer competing bids at this point in the year.",
          "DoD contracts in the $1M–$5M range show the highest win probability for mid-tier vendors with a proven delivery track record.",
          "Agency activity is elevated. High-activity agencies move faster through procurement — submit a complete package, no follow-up prompts needed.",
        ],
      });
      // ── END MOCK ──
    } catch (err) {
      setError("Analysis failed. Please check your connection and try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <HeroBanner />
      <div style={{ display: "flex", minHeight: "calc(100vh - 64px - 152px)" }}>
        <Sidebar inputs={inputs} setInputs={setInputs} onRun={runAnalysis} loading={loading} />

        <main style={{ flex: 1, padding: "32px 36px", background: COLORS.parchment }}>

          {/* ── ERROR STATE ── */}
          {error && (
            <div style={{
              background: "#FCEBEB",
              border: "0.5px solid #A32D2D",
              borderLeft: "3px solid #A32D2D",
              borderRadius: 4,
              padding: "12px 16px",
              marginBottom: 20,
              fontSize: 13,
              color: "#A32D2D",
              display: "flex",
              alignItems: "center",
              gap: 10,
            }}>
              <span>⚠</span> {error}
            </div>
          )}

          {/* ── EMPTY STATE ── */}
          {!result && !loading && !error && (
            <div className="fade-in" style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
              minHeight: 340,
              gap: 16,
            }}>
              <div style={{
                width: 56,
                height: 56,
                border: `0.5px solid ${COLORS.border}`,
                borderRadius: "50%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                background: COLORS.parchmentMid,
              }}>
                <span style={{ fontSize: 22, color: COLORS.gold }}>◎</span>
              </div>
              <p style={{ fontFamily: "'Playfair Display', serif", fontSize: 17, color: COLORS.navy, fontWeight: 400 }}>
                Configure your contract and run analysis
              </p>
              <p style={{ fontSize: 13, color: COLORS.muted, maxWidth: 380, textAlign: "center", lineHeight: 1.65 }}>
                Set the parameters on the left and click Run Analysis to get your win
                probability, factor breakdown, and AI strategy brief.
              </p>
            </div>
          )}

          {/* ── LOADING STATE ── */}
          {loading && (
            <div className="fade-in" style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
              minHeight: 340,
              gap: 14,
            }}>
              <div style={{
                width: 48,
                height: 48,
                border: `2px solid ${COLORS.border}`,
                borderTopColor: COLORS.navy,
                borderRadius: "50%",
                animation: "spin 0.8s linear infinite",
              }} />
              <p style={{ fontSize: 13, color: COLORS.muted, letterSpacing: "0.08em" }}>
                Running 4-node AI pipeline…
              </p>
            </div>
          )}

          {/* ── RESULTS ── */}
          {result && !loading && (
            <div className="fade-up">
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 20 }}>

                {/* Win Probability */}
                <div style={{
                  background: COLORS.parchment,
                  border: `0.5px solid ${COLORS.border}`,
                  borderRadius: 8,
                  padding: "24px 26px",
                }}>
                  <div style={{
                    fontSize: 10,
                    letterSpacing: "0.14em",
                    textTransform: "uppercase",
                    color: COLORS.muted,
                    marginBottom: 18,
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                  }}>
                    <span style={{ display: "inline-block", width: 3, height: 14, background: COLORS.gold, borderRadius: 2 }} />
                    Win Probability
                  </div>
                  <WinGauge pct={result.pct} />
                </div>

                {/* SHAP */}
                <div style={{
                  background: COLORS.parchment,
                  border: `0.5px solid ${COLORS.border}`,
                  borderRadius: 8,
                  padding: "24px 26px",
                }}>
                  <div style={{
                    fontSize: 10,
                    letterSpacing: "0.14em",
                    textTransform: "uppercase",
                    color: COLORS.muted,
                    marginBottom: 18,
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                  }}>
                    <span style={{ display: "inline-block", width: 3, height: 14, background: COLORS.gold, borderRadius: 2 }} />
                    What's Driving This Score
                  </div>
                  <ShapChart features={result.features} />
                </div>
              </div>

              {/* Strategy Brief */}
              <StrategyBrief points={result.points} />


            </div>
          )}
        </main>
      </div>
    </div>
  );
}

// ─── ABOUT PAGE ────────────────────────────────────────────────────────────────
const techStack = [
  {
    name: "Random Forest",
    role: "The prediction engine",
    why: "We trained it on 969 real federal contracts. It learned what separates winning bids from losing ones — not from assumptions, but from actual procurement history.",
    ahead: "Most win-rate tools are spreadsheets with formulas. This one learned from real outcomes.",
  },
  {
    name: "SHAP",
    role: "The explanation layer",
    why: "Every score comes with a reason. SHAP breaks open the model and tells you exactly which factors helped your bid and which ones are dragging it down — in plain numbers.",
    ahead: "Knowing your score is useful. Knowing why it is that score is what lets you act on it.",
  },
  {
    name: "LangGraph",
    role: "The intelligence pipeline",
    why: "Four AI agents run in sequence — one predicts, one retrieves context, one analyzes, one writes the strategy brief. Each step checks the last. No single point of failure.",
    ahead: "Most AI tools are a single prompt. This is a structured reasoning chain — closer to how a consulting team actually works.",
  },
  {
    name: "ChromaDB",
    role: "The memory layer",
    why: "We indexed 500 historical contracts so the AI can reference real precedents when writing your strategy brief — not invented examples, not hallucinations.",
    ahead: "The strategy brief is grounded in actual data. That's the difference between advice you can trust and advice that sounds good.",
  },
  {
    name: "Groq + Llama 3.3-70B",
    role: "The language engine",
    why: "The strategy brief is generated and delivered in under two seconds. Fast enough for a live pitch room, sharp enough for a board-level conversation.",
    ahead: "Speed without quality is noise. Quality without speed misses the moment. This delivers both.",
  },
  {
    name: "GCP Cloud Run",
    role: "The deployment layer",
    why: "The entire system — model, pipeline, and interface — runs as a single container on Google Cloud. No servers to manage. It scales to zero when idle and spins up instantly on demand.",
    ahead: "Built like a production system, not a prototype. The same infrastructure used by engineering teams at scale.",
  },
];

function AboutPage() {
  return (
    <div style={{ maxWidth: 860, margin: "0 auto", padding: "52px 40px" }}>
      <div className="fade-up">
        <div style={{ fontSize: 10, letterSpacing: "0.22em", textTransform: "uppercase", color: COLORS.gold, marginBottom: 14, fontWeight: 500 }}>
          About this system
        </div>
        <h1 style={{
          fontFamily: "'Playfair Display', serif",
          fontSize: 32,
          fontWeight: 500,
          color: COLORS.navy,
          lineHeight: 1.3,
          marginBottom: 16,
        }}>
          Built to answer one question:<br />
          <em>Should we bid on this contract?</em>
        </h1>
        <p style={{ fontSize: 15, color: COLORS.mutedDark, lineHeight: 1.8, maxWidth: 640, marginBottom: 48, fontWeight: 300 }}>
          Federal procurement is slow, opaque, and expensive to get wrong. This system was
          built to cut through that — using real contract data, explainable AI, and a pipeline
          that thinks the way a consulting team would.
        </p>

        <div style={{ borderTop: `0.5px solid ${COLORS.border}`, paddingTop: 40 }}>
          <div style={{ fontSize: 10, letterSpacing: "0.18em", textTransform: "uppercase", color: COLORS.muted, marginBottom: 32, fontWeight: 500 }}>
            The technology — and why it matters
          </div>

          <div style={{ display: "flex", flexDirection: "column" }}>
            {techStack.map((t, i) => (
              <div
                key={i}
                className="stagger-row"
                style={{
                  display: "grid",
                  gridTemplateColumns: "180px 1fr",
                  gap: "0 36px",
                  padding: "28px 0",
                  borderBottom: `0.5px solid ${COLORS.border}`,
                  animationDelay: `${i * 0.08}s`,
                }}
              >
                <div>
                  <div style={{
                    fontFamily: "'Playfair Display', serif",
                    fontSize: 16,
                    fontWeight: 500,
                    color: COLORS.navy,
                    marginBottom: 5,
                  }}>{t.name}</div>
                  <div style={{
                    fontSize: 10,
                    letterSpacing: "0.12em",
                    textTransform: "uppercase",
                    color: COLORS.gold,
                    fontWeight: 500,
                  }}>{t.role}</div>
                </div>
                <div>
                  <p style={{ fontSize: 14, color: COLORS.ink, lineHeight: 1.75, marginBottom: 10, fontWeight: 400 }}>
                    {t.why}
                  </p>
                  <div style={{
                    background: COLORS.parchmentMid,
                    border: `0.5px solid ${COLORS.border}`,
                    borderLeft: `3px solid ${COLORS.gold}`,
                    borderRadius: "0 4px 4px 0",
                    padding: "10px 14px",
                  }}>
                    <span style={{ fontSize: 12, color: COLORS.mutedDark, lineHeight: 1.65, fontStyle: "italic" }}>{t.ahead}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Built By block */}
        <div style={{ marginTop: 56, padding: "28px 32px", background: COLORS.navyDeep, borderRadius: 8 }}>
          <div style={{ fontSize: 10, letterSpacing: "0.18em", textTransform: "uppercase", color: COLORS.gold, marginBottom: 14, fontWeight: 500 }}>
            Built by
          </div>
          <div style={{ fontFamily: "'Playfair Display', serif", fontSize: 22, color: COLORS.parchment, fontWeight: 500, marginBottom: 10 }}>
            Sanjay Sarella
          </div>
          <p style={{ fontSize: 13, color: "#ADA79E", lineHeight: 1.75, maxWidth: 520, fontWeight: 300 }}>
            M.S. Data Analytics · Oklahoma City University · USA<br />
            Specialising in agentic AI systems that translate data into business decisions —
            not just dashboards.
          </p>

        </div>
      </div>
    </div>
  );
}

// ─── ROOT APP ──────────────────────────────────────────────────────────────────
export default function App() {
  const [page, setPage] = useState("Dashboard");

  return (
    <>
      <style>{globalStyles}</style>
      <div style={{ minHeight: "100vh", background: COLORS.parchment }}>
        <Header page={page} setPage={setPage} />
        {page === "Dashboard" ? <DashboardPage /> : <AboutPage />}
      </div>
    </>
  );
}
