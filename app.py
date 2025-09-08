# app.py — TimeSculpt (complete build: AI autodetect, smart lens, custom loops+units, diagnostics fix, visuals)
import os, json, random, datetime as dt
import numpy as np, pandas as pd, altair as alt, streamlit as st
import sqlite3
import openai

# ================= AI KEY AUTODETECT =================
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
if api_key:
    openai.api_key = api_key
    st.sidebar.success("✅ AI is active")
    AI_ACTIVE = True
else:
    st.sidebar.info("ℹ️ AI narration is off (no key found).")
    AI_ACTIVE = False

# ================= DB (schema-safe) =================
DB = "timesculpt.db"

def _conn():
    c = sqlite3.connect(DB)
    c.row_factory = lambda cur,row: {d[0]: row[i] for i,d in enumerate(cur.description)}
    return c

def _table_cols(name):
    with _conn() as c:
        return [r["name"] for r in c.execute(f"PRAGMA table_info({name})").fetchall()]

def _ensure_table_sql():
    return """
    CREATE TABLE IF NOT EXISTS days(
      d TEXT PRIMARY KEY, note TEXT, state TEXT,
      focus REAL, energy REAL, progress REAL
    );
    CREATE TABLE IF NOT EXISTS loops(
      d TEXT, k TEXT, minutes REAL, unit TEXT,
      PRIMARY KEY(d,k)
    );
    CREATE TABLE IF NOT EXISTS lens(name TEXT PRIMARY KEY, data TEXT);
    CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, val TEXT);
    CREATE TABLE IF NOT EXISTS custom_loops(name TEXT PRIMARY KEY, category TEXT, polarity INTEGER);
    CREATE TABLE IF NOT EXISTS lens_memory(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      at TEXT, lens TEXT, kind TEXT, phrase TEXT, ctx TEXT
    );
    CREATE TABLE IF NOT EXISTS interventions_log(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      at TEXT, title TEXT,
      accepted INTEGER DEFAULT 0,
      helped INTEGER DEFAULT 0
    );
    """

def _ensure_columns():
    with _conn() as c:
        # lens_memory columns safety
        cols = _table_cols("lens_memory")
        for col, sqltype in (("at","TEXT"),("lens","TEXT"),("kind","TEXT"),("phrase","TEXT"),("ctx","TEXT")):
            if col not in cols:
                c.execute(f"ALTER TABLE lens_memory ADD COLUMN {col} {sqltype}")

def init_db():
    with _conn() as c:
        c.executescript(_ensure_table_sql())
    _ensure_columns()
init_db()

def settings_get(k, default=""):
    with _conn() as c:
        r = c.execute("SELECT val FROM settings WHERE key=?",(k,)).fetchone()
    return r["val"] if r else default

def settings_set(k,v):
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO settings VALUES(?,?)",(k,str(v)))

def lens_put(name,data):
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO lens VALUES(?,?)",(name,json.dumps(data)))

def lens_all():
    with _conn() as c:
        return c.execute("SELECT * FROM lens").fetchall()

def custom_loops_all():
    with _conn() as c:
        return c.execute("SELECT * FROM custom_loops").fetchall()

def custom_loop_add(name,cat,pol):
    if not (name or "").strip(): return
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO custom_loops VALUES(?,?,?)",(name.strip(),cat,int(pol)))

def _insert_loop_row(c, d, k, minutes, unit):
    cols = _table_cols("loops")
    values=[]
    for col in cols:
        if col == "d": values.append(d)
        elif col in ("k","name"): values.append(k)
        elif col in ("minutes","v"): values.append(float(minutes))
        elif col == "unit": values.append(unit or "minutes")
        else: values.append(None)
    placeholders=",".join("?"*len(values))
    c.execute(f"INSERT OR REPLACE INTO loops VALUES({placeholders})", values)

def save_day(d, note, loops_dict, state, F, E, P):
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO days VALUES (?,?,?,?,?,?)",(d,note,state,F,E,P))
        for k,val in loops_dict.items():
            minutes = float(val.get("minutes",0)) if isinstance(val,dict) else float(val or 0)
            unit    = (val.get("unit") or "minutes") if isinstance(val,dict) else "minutes"
            _insert_loop_row(c, d, k, minutes, unit)

def load_days(n=120):
    with _conn() as c:
        days = c.execute("SELECT * FROM days ORDER BY d ASC").fetchall()
        loops= c.execute("SELECT * FROM loops").fetchall()
    by={}
    for r in loops:
        d=r.get("d"); k=r.get("k") or r.get("name")
        if not d or not k: continue
        minutes = r.get("minutes", r.get("v",0)) or 0
        unit    = r.get("unit","minutes")
        by.setdefault(d,{})[k]={"minutes":float(minutes),"unit":unit}
    for d in days: d["loops"]=by.get(d["d"],{})
    return days[-n:] if n else days

# ================= Lens system =================
CORE_LENS = {
    "name":"Core",
    "collapse":[
        "Release what drags the timeline.",
        "Close the tab. End the loop.",
        "No path opens while you hold every door."
    ],
    "recursion":[
        "Repeat the action that proves the future.",
        "Small loops compound into fate.",
        "Consistency sculpts identity."
    ],
    "emergence":[
        "Invite the first true move.",
        "Begin poorly. Arrival happens mid-motion.",
        "Bend toward the version of you that acts."
    ],
    "neutral":[ "Attend to what is here. Choose again." ]
}

def get_active_lens():
    name = settings_get("active_lens","Core")
    if name=="Core": return CORE_LENS
    for r in lens_all():
        if r["name"]==name:
            try: return {"name":name, **json.loads(r["data"])}
            except: break
    return CORE_LENS

def log_lens_memory(lens, kind, phrase, ctx):
    with _conn() as c:
        c.execute(
            "INSERT INTO lens_memory(at,lens,kind,phrase,ctx) VALUES(?,?,?,?,?)",
            (dt.datetime.now().isoformat(), lens, kind, phrase, json.dumps(ctx or {}))
        )

def last_lens_memory(limit=50):
    with _conn() as c:
        return c.execute("SELECT * FROM lens_memory ORDER BY id DESC LIMIT ?",(limit,)).fetchall()

def smart_lens_line(kind, ctx):
    """Memory-aware, goal-biased phrase picker."""
    lens = get_active_lens()
    pool = lens.get(kind,[]) or CORE_LENS.get(kind,[])
    if not pool: return ""
    # anti-repetition
    recent = {r["phrase"] for r in last_lens_memory(10) if r.get("kind")==kind}
    candidates = [p for p in pool if p not in recent] or pool
    # goal bias
    goal=(ctx or {}).get("goal","").lower()
    if goal:
        weighted=[]; gl=set([w for w in goal.split() if len(w)>=4])
        for p in candidates:
            score=1+sum(1 for w in gl if w in p.lower())
            weighted.extend([p]*min(4,score))
        candidates=weighted or candidates
    phrase=random.choice(candidates)
    log_lens_memory(lens.get("name","Core"),kind,phrase,ctx or {})
    return phrase

# ================= Units & conversions =================
def default_units_cfg():
    return {
        "creation:writing":  {"units":["minutes","sessions"], "rate":{"sessions_to_minutes":15}},
        "creation:project":  {"units":["minutes","sessions"], "rate":{"sessions_to_minutes":20}},
        "mind:planning":     {"units":["minutes","sessions"], "rate":{"sessions_to_minutes":10}},
        "mind:reading":      {"units":["minutes","pages","chapters"], "rate":{"pages_to_minutes":1.0,"chapters_to_pages":5}},
        "mind:meditation":   {"units":["minutes","sessions"], "rate":{"sessions_to_minutes":15}},
        "body:walk":         {"units":["minutes","steps"], "rate":{"steps_to_minutes":0.01}},
        "body:exercise":     {"units":["minutes","reps","sets"], "rate":{"reps_to_minutes":0.2,"sets_to_minutes":3}},
        "body:sleep_good":   {"units":["hours"], "rate":{"hours_to_minutes":60}},
        "body:late_sleep":   {"units":["minutes"], "rate":{}},
        "consumption:scroll":{"units":["minutes"], "rate":{}},
        "consumption:youtube":{"units":["minutes"], "rate":{}},
        "food:junk":         {"units":["minutes","items"], "rate":{"items_to_minutes":5}},
        "finance:save_invest":{"units":["minutes","£","%"], "rate":{"pounds_to_minutes":0.05,"percent_to_minutes":0.2}},
        "finance:budget_check":{"units":["minutes","sessions"], "rate":{"sessions_to_minutes":10}},
        "finance:impulse_spend":{"units":["minutes","£"], "rate":{"pounds_to_minutes":0.1}},
    }

def get_units_cfg():
    raw = settings_get("units_cfg","")
    base = default_units_cfg()
    try:
        user = json.loads(raw) if raw else {}
    except:
        user = {}
    # merge user into base for built-ins
    for k,v in base.items():
        if k not in user: user[k]=v
        else:
            user[k].setdefault("units", v["units"])
            user[k].setdefault("rate", {})
            for rk,rv in v["rate"].items():
                user[k]["rate"].setdefault(rk, rv)
    # defaults for custom loops
    for r in custom_loops_all():
        key = f"{r['category']}:{r['name']}"
        if key not in user:
            user[key] = {"units":["minutes","sessions"], "rate":{"sessions_to_minutes":15}}
    return user

def save_units_cfg(cfg):
    settings_set("units_cfg", json.dumps(cfg))

def normalize_amount(loop_key, qty, unit, cfg):
    if unit=="minutes": return float(qty)
    r = cfg.get(loop_key,{}).get("rate",{})
    if unit=="hours":      return float(qty) * float(r.get("hours_to_minutes",60))
    if unit=="pages":      return float(qty) * float(r.get("pages_to_minutes",1.0))
    if unit=="chapters":   return float(qty) * float(r.get("chapters_to_pages",5)) * float(r.get("pages_to_minutes",1.0))
    if unit=="sessions":   return float(qty) * float(r.get("sessions_to_minutes",15))
    if unit=="reps":       return float(qty) * float(r.get("reps_to_minutes",0.2))
    if unit=="sets":       return float(qty) * float(r.get("sets_to_minutes",3))
    if unit=="steps":      return float(qty) * float(r.get("steps_to_minutes",0.01))
    if unit=="items":      return float(qty) * float(r.get("items_to_minutes",5))
    if unit=="£":          return float(qty) * float(r.get("pounds_to_minutes",0.05))
    if unit=="%":          return float(qty) * float(r.get("percent_to_minutes",0.2))
    return float(qty)

# ================= State model + forecast =================
W_POS,W_NEG,W_PROG,W_ENER=0.8,0.9,0.25,0.15

POS_BASE = {"creation:writing","creation:project","mind:planning","mind:reading",
            "mind:meditation","body:walk","body:exercise","body:sleep_good",
            "finance:save_invest","finance:budget_check"}
NEG_BASE = {"consumption:scroll","consumption:youtube","food:junk","body:late_sleep",
            "finance:impulse_spend"}

def compute_polarity_sets():
    pos, neg = set(POS_BASE), set(NEG_BASE)
    for r in custom_loops_all():
        key = f"{r['category']}:{r['name']}"
        if int(r.get("polarity", 1)) > 0: pos.add(key)
        else: neg.add(key)
    return pos, neg

def label_state(loops):
    POS, NEG = compute_polarity_sets()
    g=lambda k: loops.get(k,{}).get("minutes",0)
    posm=sum(g(k) for k in POS); negm=sum(g(k) for k in NEG)
    energy=min(100,(g("body:walk")*1.2+g("body:exercise")*1.6+g("body:sleep_good")*1.5)/2.0)
    progress=min(100,(g("creation:writing")*1.4+g("creation:project")*1.2+g("finance:save_invest")*1.1+g("mind:planning")*0.9))
    focus_raw=(posm*W_POS-negm*W_NEG)+progress*W_PROG+energy*W_ENER
    focus=max(0,min(100,focus_raw))
    if negm>posm*1.2 or g("consumption:scroll")>=45: state="Drift"
    elif posm>=negm and (g("creation:writing")+g("creation:project"))>=30: state="Focused"
    else: state="Mixed"
    return state,round(focus,1),round(energy,1),round(progress,1)

STATES=["Focused","Mixed","Drift"]; IDX={s:i for s,i in zip(STATES,range(3))}
DECAY,PRIOR,BLEND=0.97,0.5,0.08

def learn_matrix(days,decay=DECAY):
    C=np.ones((3,3))*PRIOR; last=None; w=1.0
    for d in days:
        s=d.get("state")
        if s not in IDX: continue
        if last is not None: C[IDX[last],IDX[s]]+=w
        w*=decay; last=s
    M=C/C.sum(axis=1,keepdims=True); U=np.ones((3,3))/3.0
    M=(1-BLEND)*M+BLEND*U
    return M/M.sum(axis=1,keepdims=True)

def simulate(M,start_state,days=30,sims=2000):
    start=IDX.get(start_state,1)
    counts=np.zeros((days,3))
    for _ in range(sims):
        s=start
        for t in range(days):
            counts[t,s]+=1
            s=np.random.choice([0,1,2],p=M[s])
    probs=counts/sims
    return probs,float(probs[:,0].sum())

def double_slit_forecast(steps=30):
    # optional alternative perspective (no quantum lens text)
    psi=np.array([1/np.sqrt(2),1/np.sqrt(2)],dtype=complex)
    U=np.array([[1,1],[1,-1]],dtype=complex)/np.sqrt(2)
    out=[]
    for _ in range(steps):
        psi=U@psi
        out.append(np.abs(psi)**2) # [focus, drift]
    probs=np.zeros((steps,3))
    probs[:,0]=np.array(out)[:,0]
    probs[:,2]=np.array(out)[:,1]
    probs[:,1]=1-probs[:,0]-probs[:,2]
    return probs

# ================= UI (styles) =================
st.set_page_config(page_title="TimeSculpt", layout="wide")
st.markdown("""
<style>
:root { --bg:#0d0d0d; --pane:#111; --fg:#eaeaea; --muted:#9aa0a6; --gold:#E0C36D; }
html, body, [class*="block-container"] { color: var(--fg); background: var(--bg); }
h1,h2,h3 { color: var(--gold) !important; }
.card{background:#131313;border:1px solid #222;border-radius:14px;padding:16px;margin:8px 0;}
.badge{background:#222;border:1px solid #333;border-radius:999px;padding:2px 8px;color:#bbb;font-size:.8rem}
.pill{background:#1a1a1a;border:1px solid #333;border-radius:999px;padding:2px 8px;color:#bbb;margin-right:6px}
label, .stTextInput label, .stNumberInput label, .stSelectbox label, .stTextArea label { color:#e9eef3 !important; font-weight:600; }
small.hint { color:#b7bdc3; font-size:0.85rem; display:block; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.checkbox("AI narration", value=False, key="use_ai")
st.sidebar.markdown("**Forecast mode**")
st.sidebar.radio(" ", ["Classical","Quantum (Double Slit)"], key="forecast_mode", label_visibility="collapsed")
tab = st.sidebar.radio("Go to", ["Input","Forecast","Interventions","Diagnostics","Lens","Lens Memory","Export","Help"], key="nav")

# Sticky header
days_all = load_days(365)
st.markdown("<div class='card' style='position:sticky;top:0;z-index:10;'>", unsafe_allow_html=True)
if days_all:
    t=days_all[-1]
    st.markdown(f"**Today** · {t['state']} · Focus {t['focus']:.0f} · Energy {t['energy']:.0f} · Progress {t['progress']:.0f}")
else:
    st.markdown("**Today** · no data yet")
st.markdown("</div>", unsafe_allow_html=True)

# ================= Tabs =================
if tab=="Input":
    st.header("Input")
    date = st.date_input("Date", value=dt.date.today()).isoformat()
    goal = st.text_input("Goal / Desire", value=settings_get("goal",""))
    if st.button("Save goal"): settings_set("goal",goal); st.success("Saved.")

    units_cfg = get_units_cfg()

    FIELDS = [
        ("creation:writing","Writing"),
        ("creation:project","Project"),
        ("mind:planning","Planning"),
        ("mind:reading","Reading"),
        ("mind:meditation","Meditation"),
        ("body:walk","Walk"),
        ("body:exercise","Exercise"),
        ("body:sleep_good","Good sleep"),
        ("body:late_sleep","Late sleep"),
        ("consumption:scroll","Scroll"),
        ("consumption:youtube","YouTube"),
        ("food:junk","Junk food"),
        ("finance:save_invest","Save/Invest"),
        ("finance:budget_check","Budget check"),
        ("finance:impulse_spend","Impulse spend"),
    ]

    # add custom loops into the grid
    custom_rows = custom_loops_all()
    custom_fields = []
    for r in custom_rows:
        key = f"{r['category']}:{r['name']}"
        label = f"{r['name']} ({r['category']})"
        custom_fields.append((key, label))

    ALL_FIELDS = FIELDS + custom_fields

    loops_today={}
    cols = st.columns(4)
    for i,(k,label) in enumerate(ALL_FIELDS):
        with cols[i%4]:
            allowed = units_cfg.get(k,{}).get("units",["minutes"])
            unit = st.selectbox(f"{label} — unit", allowed, key=f"unit_{k}")
            step = 1.0 if unit in ("pages","reps","sets","items","steps","%","sessions") else 5.0
            maxv = 10000.0 if unit in ("steps","£") else 600.0
            if k=="body:sleep_good" and unit=="hours":
                step, maxv = 0.5, 14.0
            val = st.number_input(label, min_value=0.0, step=step, max_value=maxv, value=0.0, key=f"val_{k}")
            eff = normalize_amount(k, val, unit, units_cfg)
            loops_today[k] = {"minutes": float(eff), "unit": unit}
            st.caption(f"{val:g} {unit} → **{eff:.1f} effective min**")

    s,F,E,P = label_state(loops_today)
    st.info(f"{s} · Focus {F} · Energy {E} · Progress {P}")

    note = st.text_area("Note (optional)")
    if st.button("Commit today"):
        save_day(date, note, loops_today, s, F, E, P)
        st.success("Saved.")

    st.markdown("### Add a custom loop")
    cc1,cc2,cc3 = st.columns(3)
    with cc1: cname = st.text_input("Name")
    with cc2: ccat  = st.selectbox("Category",["creation","mind","body","consumption","food","finance"])
    with cc3: cpol  = st.selectbox("Polarity",[+1,-1])
    if st.button("Add custom"):
        custom_loop_add(cname,ccat,cpol); st.experimental_rerun()

    with st.expander("🗂 Manage custom loops"):
        rows = custom_loops_all()
        if not rows:
            st.caption("No custom loops yet.")
        else:
            for r in rows:
                cc1,cc2,cc3,cc4 = st.columns([3,2,2,1])
                with cc1: st.write(f"**{r['name']}**")
                with cc2: st.write(r['category'])
                with cc3: st.write("Positive" if int(r['polarity'])>0 else "Negative")
                with cc4:
                    if st.button("Delete", key=f"del_{r['name']}"):
                        with _conn() as c:
                            c.execute("DELETE FROM custom_loops WHERE name=?", (r["name"],))
                        st.experimental_rerun()

    with st.expander("⚙️ Units & Conversions"):
        st.write("Tune how your inputs convert into effective minutes.")
        new_cfg = get_units_cfg()

        rate_fields = []
        for k,label in FIELDS:
            rate_fields.append((k,label))
        for r in custom_loops_all():
            key = f"{r['category']}:{r['name']}"
            label = f"{r['name']} ({r['category']})"
            rate_fields.append((key,label))

        for k,label in rate_fields:
            rates = new_cfg.get(k,{}).get("rate",{})
            st.markdown(f"**{label}**")
            if "sessions_to_minutes" in rates:
                rates["sessions_to_minutes"] = st.number_input(f"{label}: minutes per session", min_value=1.0, max_value=240.0, step=1.0, value=float(rates["sessions_to_minutes"]), key=f"rate_{k}_sess")
            if "pages_to_minutes" in rates:
                rates["pages_to_minutes"] = st.number_input(f"{label}: minutes per page", min_value=0.1, max_value=10.0, step=0.1, value=float(rates["pages_to_minutes"]), key=f"rate_{k}_page")
            if "chapters_to_pages" in rates:
                rates["chapters_to_pages"] = st.number_input(f"{label}: pages per chapter", min_value=1.0, max_value=200.0, step=1.0, value=float(rates["chapters_to_pages"]), key=f"rate_{k}_chap")
            if "hours_to_minutes" in rates:
                rates["hours_to_minutes"] = st.number_input(f"{label}: minutes per hour", min_value=30.0, max_value=120.0, step=5.0, value=float(rates["hours_to_minutes"]), key=f"rate_{k}_hour")
            if "reps_to_minutes" in rates:
                rates["reps_to_minutes"] = st.number_input(f"{label}: minutes per rep", min_value=0.01, max_value=5.0, step=0.01, value=float(rates["reps_to_minutes"]), key=f"rate_{k}_rep")
            if "sets_to_minutes" in rates:
                rates["sets_to_minutes"] = st.number_input(f"{label}: minutes per set", min_value=0.1, max_value=60.0, step=0.1, value=float(rates["sets_to_minutes"]), key=f"rate_{k}_set")
            if "steps_to_minutes" in rates:
                rates["steps_to_minutes"] = st.number_input(f"{label}: minutes per step", min_value=0.001, max_value=0.1, step=0.001, value=float(rates["steps_to_minutes"]), key=f"rate_{k}_steps")
            if "items_to_minutes" in rates:
                rates["items_to_minutes"] = st.number_input(f"{label}: minutes per item", min_value=0.1, max_value=60.0, step=0.1, value=float(rates["items_to_minutes"]), key=f"rate_{k}_item")
            if "pounds_to_minutes" in rates:
                rates["pounds_to_minutes"] = st.number_input(f"{label}: minutes per £", min_value=0.001, max_value=5.0, step=0.001, value=float(rates["pounds_to_minutes"]), key=f"rate_{k}_gbp")
            if "percent_to_minutes" in rates:
                rates["percent_to_minutes"] = st.number_input(f"{label}: minutes per %", min_value=0.01, max_value=10.0, step=0.01, value=float(rates["percent_to_minutes"]), key=f"rate_{k}_pct")
            new_cfg[k]["rate"] = rates

        if st.button("Save conversions"):
            save_units_cfg(new_cfg); st.success("Saved."); st.experimental_rerun()

elif tab=="Forecast":
    st.header("Forecast")
    days = load_days(120)
    if not days:
        st.info("Log at least 1 day to unlock the forecast.")
    else:
        start = days[-1]["state"]
        if st.session_state.get("forecast_mode")=="Quantum (Double Slit)":
            probs = double_slit_forecast(30)
            df = pd.DataFrame({"day":range(1,31),
                               "Focused":probs[:,0],"Mixed":probs[:,1],"Drift":probs[:,2]})
            st.subheader("Quantum Forecast (Double Slit)")
        else:
            M = learn_matrix(days)
            probs, expF = simulate(M, start, 30, 2000)
            df = pd.DataFrame({"day":range(1,31),
                               "Focused":probs[:,0],"Mixed":probs[:,1],"Drift":probs[:,2]})
            st.markdown(f"**Expected focused days:** {df['Focused'].sum():.1f}/30")

        dfm=df.melt("day",var_name="state",value_name="p")
        st.altair_chart(
            alt.Chart(dfm).mark_area(opacity=0.85).encode(
                x="day", y=alt.Y("p", stack="normalize", axis=alt.Axis(format="%")),
                color=alt.Color("state", scale=alt.Scale(
                    domain=["Focused","Mixed","Drift"],
                    range=["#E0C36D","#777","#B91C1C"]))
            ).properties(height=260),
            use_container_width=True
        )

        ctx={"goal": settings_get("goal",""), "state": start}
        phrase = smart_lens_line("emergence" if df['Focused'].sum()>=15 else "collapse", ctx)
        st.markdown(f"> _{phrase}_")

elif tab=="Interventions":
    st.header("Interventions — smallest honest moves")
    days = load_days(120)
    if not days:
        st.info("Need some days.")
    else:
        start=days[-1]["state"]; M=learn_matrix(days); _,baseF=simulate(M,start,30,2000)
        INTERVENTIONS=[
            {"title":"7-min starter","how":"Start badly. Stop after 7.","tweak":{"m_to_f":+0.06}},
            {"title":"15-min walk","how":"Swap one scroll for a walk.","tweak":{"m_to_f":+0.05,"d_self":-0.03}},
            {"title":"Sleep before midnight","how":"Shut down 30 min earlier.","tweak":{"d_self":-0.06}},
            {"title":"10% pay-yourself-first","how":"Automate on payday.","tweak":{"m_to_f":+0.05}},
        ]
        def apply_tweak(M, **kw):
            A=M.copy()
            def adj_row(i,delta):
                A[i]=np.maximum(1e-6, A[i]+delta); A[i]/=A[i].sum()
            IDXm={"Focused":0,"Mixed":1,"Drift":2}
            if kw.get("d_self"): adj_row(IDXm["Drift"], np.array([0,0,kw["d_self"]]))
            if kw.get("d_to_m"): adj_row(IDXm["Drift"], np.array([0,kw["d_to_m"],0]))
            if kw.get("m_to_f"): adj_row(IDXm["Mixed"], np.array([kw["m_to_f"],0,0]))
            if kw.get("f_self"): adj_row(IDXm["Focused"], np.array([kw["f_self"],0,0]))
            return A

        results=[]
        for iv in INTERVENTIONS:
            M2=apply_tweak(M,**iv["tweak"]); _,f2=simulate(M2,start,30,1500)
            results.append({**iv,"delta":min(2.0,f2-baseF)})
        results.sort(key=lambda r:-r["delta"])
        lens=get_active_lens()
        one=results[0]
        st.markdown(f"<div class='card'><span class='badge'>One smallest move</span><br><b>{one['title']}</b> — {smart_lens_line('recursion', {'goal':settings_get('goal','')})}<br><br>{one['how']}<br><br>Δ Focused days ≈ +{one['delta']:.2f}</div>", unsafe_allow_html=True)

        st.subheader("All options")
        for r in results:
            st.markdown(f"- **{r['title']}** — {r['how']}  \n  Δ focused days ≈ **+{r['delta']:.2f}**")

        st.markdown("---")
        st.markdown("**Log an outcome**  \nRecord if it helped; the system will adapt.")
        colA,colB=st.columns([2,1])
        with colA: iv_title = st.text_input("Intervention title", value=one["title"])
        with colB: helped = st.selectbox("Did it help?", ["", "Yes", "No"])
        if st.button("Save outcome"):
            with _conn() as c:
                c.execute("INSERT INTO interventions_log(at,title,accepted,helped) VALUES(?,?,?,?)",
                          (dt.datetime.now().isoformat(), iv_title, 1, 1 if helped=="Yes" else 0))
            st.success("Outcome saved.")

        with _conn() as c:
            rows=c.execute("""
                SELECT title,
                       COUNT(*) AS trials,
                       SUM(helped) AS helped_sum,
                       ROUND(AVG(helped)*100.0,1) AS success_rate
                FROM interventions_log
                GROUP BY title
                ORDER BY success_rate DESC, trials DESC
            """).fetchall()
        if rows:
            df=pd.DataFrame(rows)
            st.subheader("Proven for you (so far)")
            st.dataframe(df, use_container_width=True)

elif tab=="Diagnostics":
    st.header("Diagnostics")

    days = load_days(120)
    if not days:
        st.info("Log days to see drivers.")
    else:
        # ---- flatten day -> loop rows
        rows = []
        for d in days:
            loops = d.get("loops") or {}
            for k, v in loops.items():
                rows.append({
                    "d": d["d"],
                    "k": k,
                    "minutes": v.get("minutes", 0.0),
                    "state": d["state"]
                })

        if not rows:
            st.info("No loop data yet.")
        else:
            df = pd.DataFrame(rows)

            # ---- pivot means and counts
            pivot_mean = df.pivot_table(index="k", columns="state", values="minutes", aggfunc="mean", fill_value=0)
            pivot_cnt  = df.pivot_table(index="k", columns="state", values="minutes", aggfunc="count", fill_value=0)

            # pad missing state columns
            for state in ["Focused","Mixed","Drift"]:
                if state not in pivot_mean.columns: pivot_mean[state] = 0.0
                if state not in pivot_cnt.columns:  pivot_cnt[state]  = 0

            # ---- lift & total sample size
            out = pivot_mean.copy()
            out["lift"] = out["Focused"] - out["Drift"]
            out["n_total"] = (pivot_cnt["Focused"] + pivot_cnt["Mixed"] + pivot_cnt["Drift"]).astype(int)
            out = out.reset_index().rename(columns={"k":"loop"})

            # split best/worst
            best  = out.sort_values("lift", ascending=False).head(5).copy()
            worst = out.sort_values("lift", ascending=True).head(5).copy()

            # small badges
            st.markdown(
                f"<div class='card'>"
                f"<span class='pill'>Days analyzed: {len(days)}</span> "
                f"<span class='pill'>Loops observed: {out['loop'].nunique()}</span> "
                f"<span class='pill'>Records: {len(df)}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

            c1, c2 = st.columns(2)

            # ---- Force (+)
            with c1:
                st.subheader("Force (+)")
                if best.empty:
                    st.caption("No positive drivers yet.")
                else:
                    bdf = best.sort_values("lift", ascending=True)
                    chart_best = (
                        alt.Chart(bdf)
                        .mark_bar()
                        .encode(
                            x=alt.X("lift:Q", title="Lift (Focused − Drift, avg minutes)"),
                            y=alt.Y("loop:N", sort="-x", title=None),
                            tooltip=[
                                alt.Tooltip("loop:N", title="Loop"),
                                alt.Tooltip("lift:Q", title="Lift", format=".2f"),
                                alt.Tooltip("Focused:Q", title="Focused avg (min)", format=".1f"),
                                alt.Tooltip("Drift:Q", title="Drift avg (min)", format=".1f"),
                                alt.Tooltip("n_total:Q", title="Samples")
                            ],
                            color=alt.value("#E0C36D")
                        ).properties(height=220)
                    )
                    st.altair_chart(chart_best, use_container_width=True)
                    st.caption("Top positive drivers (higher = more time on Focused days)")
                    st.dataframe(
                        bdf[["loop","lift","Focused","Drift","n_total"]]
                        .rename(columns={"n_total":"samples"})
                        .reset_index(drop=True),
                        use_container_width=True
                    )

            # ---- Drift (−)
            with c2:
                st.subheader("Drift (−)")
                if worst.empty:
                    st.caption("No negative drivers yet.")
                else:
                    wdf = worst.sort_values("lift", ascending=True)
                    chart_worst = (
                        alt.Chart(wdf)
                        .mark_bar()
                        .encode(
                            x=alt.X("lift:Q", title="Lift (Focused − Drift, avg minutes)"),
                            y=alt.Y("loop:N", sort=None, title=None),
                            tooltip=[
                                alt.Tooltip("loop:N", title="Loop"),
                                alt.Tooltip("lift:Q", title="Lift", format=".2f"),
                                alt.Tooltip("Focused:Q", title="Focused avg (min)", format=".1f"),
                                alt.Tooltip("Drift:Q", title="Drift avg (min)", format=".1f"),
                                alt.Tooltip("n_total:Q", title="Samples")
                            ],
                            color=alt.value("#B91C1C")
                        ).properties(height=220)
                    )
                    st.altair_chart(chart_worst, use_container_width=True)
                    st.caption("Top negative drivers (lower = more time on Drift days)")
                    st.dataframe(
                        wdf[["loop","lift","Focused","Drift","n_total"]]
                        .rename(columns={"n_total":"samples"})
                        .reset_index(drop=True),
                        use_container_width=True
                    )

            # Export diagnostics
            with st.expander("Export diagnostics data"):
                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", data=csv, file_name="diagnostics.csv", mime="text/csv")

elif tab=="Lens":
    st.header("Lens")
    active=settings_get("active_lens","Core")
    names=["Core"]+[r["name"] for r in lens_all()]
    sel=st.selectbox("Active lens", names, index=max(0,names.index(active) if active in names else 0))
    if st.button("Use selected"): settings_set("active_lens", sel); st.success("Active lens set.")
    st.markdown("---")
    up=st.file_uploader("Upload .txt/.docx/.pdf", type=["txt","docx","pdf"])
    lname=st.text_input("Name new lens")
    if st.button("Add Lens") and up and lname:
        text=""
        try:
            if up.name.endswith(".txt"): text=up.read().decode("utf-8","ignore")
            elif up.name.endswith(".docx"):
                import docx
                d=docx.Document(up); text="\n".join(p.text for p in d.paragraphs)
            elif up.name.endswith(".pdf"):
                import PyPDF2
                pdf=PyPDF2.PdfReader(up); text="\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception: text=""
        def clean_passages(t):
            chunks=[p.strip() for p in (t or "").replace("\r","").split("\n") if len(p.strip())>=40] or [(t or "")[:280]]
            cats={"collapse":[],"recursion":[],"emergence":[],"neutral":[]}
            KW={"collapse":["release","end","close","let go","quit","stop"],
                "recursion":["repeat","again","habit","loop","daily","consistency"],
                "emergence":["begin","start","spark","new","future","grow","transform"]}
            seen=set()
            for raw in chunks:
                s=" ".join(raw.split())
                if s in seen: continue
                seen.add(s)
                low=s.lower()
                cat="neutral"
                for k,keys in KW.items():
                    if any(x in low for x in keys): cat=k; break
                cats[cat].append(s[:400])
            return cats
        lens_put(lname, clean_passages(text))
        st.success("Lens added."); st.experimental_rerun()

elif tab=="Lens Memory":
    st.header("Lens Memory")
    rows=last_lens_memory(50)
    if not rows: st.info("No narration logged yet.")
    for r in rows:
        st.markdown(f"<div class='card'><span class='pill'>{r.get('lens','')}</span> <span class='pill'>{r.get('kind','')}</span> <b>{r.get('phrase','')}</b><br><span class='badge'>{r.get('at','')}</span></div>", unsafe_allow_html=True)

elif tab=="Export":
    st.header("Export")
    days=load_days(0)
    data={"days":days,"settings":[{"key":k,"val":settings_get(k,"")} for k in ["goal","active_lens","units_cfg"]]}
    st.download_button("Download JSON", data=json.dumps(data,indent=2), file_name="timesculpt_export.json")

elif tab=="Help":
    st.header("How to use TimeSculpt")
    st.markdown("""
**Daily logging**
- Choose a **unit** for each field (e.g., reading in *pages*, sleep in *hours*, exercise in *reps* or *sets*).
- The line under each input shows how it converts to **effective minutes** (what the engine uses).

**States**
- Days classify into **Focused / Mixed / Drift** from your loops.  
- **Focus** = positives − negatives + boosts from Energy (walk/exercise/sleep) and Progress (writing/project/finance).

**Forecast**
- 30-day outlook. “Expected focused days” is the sum of Focus probabilities.
- You can also switch to **Quantum (Double Slit)** for an alternative perspective (optional).

**Interventions**
- Tiny moves scored by **Δ focused days**. Log whether it helped — the **Proven for you** table builds over time.

**Lens**
- Pick a voice (or upload text). The app remembers lines it used in **Lens Memory**.

**AI Activation**
- Put your key in **.streamlit/secrets.toml**:  
  `OPENAI_API_KEY = "sk-..."`  
  or set an env var in CMD/Terminal:  
  Windows: `setx OPENAI_API_KEY "sk-..."` (restart terminal)  
  macOS/Linux: `export OPENAI_API_KEY="sk-..."` (then run Streamlit)

**Export**
- Download your full dataset as JSON any time.
    """)
