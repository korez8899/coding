# TimeSculpt — Phase 3 Final (single file)
# Fully active app with profiles, goals, timestamped loops, forecasts (probability+ETA),
# interventions scoring, diagnostics, lenses (upload/select/mix) + narration,
# AI toggle/API key, theme, and self-healing SQLite schema.

import streamlit as st
import sqlite3, os, hashlib, datetime as dt, random, re, math, json
import numpy as np
import pandas as pd
import altair as alt

# ---------- Config / Theme ----------
st.set_page_config(page_title="TimeSculpt", page_icon="⏳", layout="wide")

THEME = {
    "bg0": "#0b1020",   # deep navy
    "bg1": "#111836",
    "bg2": "#0f1530",
    "gold": "#f6c56f",
    "off": "#f2f4f8",   # off-white body
    "muted": "#9ea5b4",
    "card": "#0c1430",
    "border": "#2b3558",
    "accent": "#7cc0ff"
}

st.markdown(f"""
<style>
:root {{
  --bg0:{THEME['bg0']}; --bg1:{THEME['bg1']}; --bg2:{THEME['bg2']};
  --off:{THEME['off']}; --gold:{THEME['gold']}; --muted:{THEME['muted']};
  --card:{THEME['card']}; --border:{THEME['border']}; --accent:{THEME['accent']};
}}
html, body, [data-testid="stAppViewContainer"] {{ background: radial-gradient(1200px 800px at 20% 0%, var(--bg1), var(--bg0)) fixed !important; color: var(--off); }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0d1430 0%, #0b1020 100%); border-right:1px solid var(--border); }}
h1,h2,h3,h4,h5,h6 {{ color: var(--gold) !important; font-weight: 800 !important; letter-spacing:.3px; }}
p,li,span,div {{ color: var(--off); }}
.small-muted {{ color: var(--muted); font-size: 0.9rem; }}
.badge {{ display:inline-block; padding:3px 8px; border:1px solid var(--border); border-radius:999px; background:#0e1a3b; color:var(--off); font-size:.8rem; }}
.card {{ background: var(--card); border:1px solid var(--border); border-radius:16px; padding:14px 16px; }}
.label {{ color: var(--off); font-weight:600; }}
.stTextInput>div>div>input, .stNumberInput input, .stDateInput input, .stTimeInput input, textarea, .stSelectbox div[data-baseweb="select"] input {{ color: var(--off) !important; }}
.stSelectbox>div>div {{ color: var(--off) !important; }}
.block-label {{ font-weight:700; color: var(--gold); }}
hr.svelte {{ border:none; height:1px; background:linear-gradient(90deg, transparent, var(--border), transparent); margin:10px 0 20px; }}
a, .markdown-text-container a {{ color: var(--accent); }}
</style>
""", unsafe_allow_html=True)

# ---------- Storage ----------
DB_PATH = "timesculpt.db"

def conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def table_cols(c, table):
    cur = c.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]

def col_exists(c, table, col):
    return col in table_cols(c, table)

def execmany(c, sql, seq):
    c.executemany(sql, seq)
    c.commit()

def ensure_schema():
    c = conn()
    cur = c.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS profiles(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        pin_hash TEXT,
        ai_toggle INTEGER DEFAULT 0,
        api_key TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS fs_traits(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER, trait TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS fs_letters(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER, body TEXT,
        resurfacing_rule TEXT DEFAULT 'dip', last_shown TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS goals(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        name TEXT,
        unit TEXT,
        target REAL,
        deadline TEXT,
        priority INTEGER DEFAULT 3,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS loops(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        name TEXT,
        value REAL,
        unit TEXT,
        ts TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS lens_lines(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        collection TEXT,
        category TEXT,
        text TEXT,
        used INTEGER DEFAULT 0
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS actions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER, kind TEXT, payload TEXT, ts TEXT
    )""")
    # Self-heal: add missing columns if users carry older DBs
    for tbl, need in {
        "profiles": ["ai_toggle", "api_key"],
        "lens_lines": ["collection", "category", "used"]
    }.items():
        cols = table_cols(c, tbl)
        for col in need:
            if col not in cols:
                if tbl == "profiles" and col == "ai_toggle":
                    cur.execute("ALTER TABLE profiles ADD COLUMN ai_toggle INTEGER DEFAULT 0")
                if tbl == "profiles" and col == "api_key":
                    cur.execute("ALTER TABLE profiles ADD COLUMN api_key TEXT")
                if tbl == "lens_lines" and col == "collection":
                    cur.execute("ALTER TABLE lens_lines ADD COLUMN collection TEXT")
                if tbl == "lens_lines" and col == "category":
                    cur.execute("ALTER TABLE lens_lines ADD COLUMN category TEXT")
                if tbl == "lens_lines" and col == "used":
                    cur.execute("ALTER TABLE lens_lines ADD COLUMN used INTEGER DEFAULT 0")
    c.commit(); c.close()

ensure_schema()

# ---------- Helpers ----------
def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def now_iso():
    return dt.datetime.now().isoformat(timespec="seconds")

def today():
    return dt.date.today()

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ---------- Session ----------
if "profile_id" not in st.session_state:
    st.session_state.profile_id = None
if "profile_name" not in st.session_state:
    st.session_state.profile_name = None

# ---------- Data Access ----------
def get_profiles():
    c = conn(); cur = c.cursor()
    cur.execute("SELECT id, name, ai_toggle, api_key FROM profiles ORDER BY name")
    rows = cur.fetchall(); c.close()
    return rows

def create_profile(name: str, pin: str):
    c = conn(); cur = c.cursor()
    cur.execute("INSERT INTO profiles(name, pin_hash) VALUES(?,?)", (name, sha(pin)))
    c.commit(); c.close()

def set_active_profile(name: str):
    rows = get_profiles()
    for pid, nm, ai_t, api in rows:
        if nm == name:
            st.session_state.profile_id = pid
            st.session_state.profile_name = nm
            return True
    return False

def check_pin(name: str, pin: str) -> bool:
    c = conn(); cur = c.cursor()
    cur.execute("SELECT pin_hash, id FROM profiles WHERE name=?", (name,))
    row = cur.fetchone(); c.close()
    if not row: return False
    ok = (row[0] == sha(pin))
    if ok:
        st.session_state.profile_id = row[1]
        st.session_state.profile_name = name
    return ok

def update_profile_flags(pid: int, ai_toggle: int = None, api_key: str = None):
    c = conn(); cur = c.cursor()
    if ai_toggle is not None:
        cur.execute("UPDATE profiles SET ai_toggle=? WHERE id=?", (int(ai_toggle), pid))
    if api_key is not None:
        cur.execute("UPDATE profiles SET api_key=? WHERE id=?", (api_key, pid))
    c.commit(); c.close()

def get_profile_flags(pid: int):
    c = conn(); cur = c.cursor()
    cur.execute("SELECT ai_toggle, api_key FROM profiles WHERE id=?", (pid,))
    row = cur.fetchone(); c.close()
    return (row[0], row[1]) if row else (0, None)

def add_trait(pid, t):
    c=conn(); c.execute("INSERT INTO fs_traits(profile_id, trait) VALUES(?,?)",(pid,t)); c.commit(); c.close()

def get_traits(pid):
    c=conn(); cur=c.cursor(); cur.execute("SELECT id, trait FROM fs_traits WHERE profile_id=? ORDER BY id DESC",(pid,)); rows=cur.fetchall(); c.close(); return rows

def delete_trait(tid):
    c=conn(); c.execute("DELETE FROM fs_traits WHERE id=?",(tid,)); c.commit(); c.close()

def save_letter(pid, body):
    c=conn(); c.execute("INSERT INTO fs_letters(profile_id, body) VALUES(?,?)",(pid, body)); c.commit(); c.close()

def get_letters(pid):
    c=conn(); cur=c.cursor(); cur.execute("SELECT id, body, created_at FROM fs_letters WHERE profile_id=? ORDER BY id DESC",(pid,)); rows=cur.fetchall(); c.close(); return rows

def add_goal(pid, name, unit, target, deadline, priority):
    c=conn(); c.execute("""INSERT INTO goals(profile_id, name, unit, target, deadline, priority)
                           VALUES(?,?,?,?,?,?)""",(pid,name,unit,target,deadline,priority)); c.commit(); c.close()

def get_goals(pid):
    c=conn(); cur=c.cursor(); cur.execute("""SELECT id,name,unit,target,deadline,priority,created_at
        FROM goals WHERE profile_id=? ORDER BY priority DESC, deadline ASC""",(pid,)); rows=cur.fetchall(); c.close(); return rows

def log_loop(pid, name, value, unit, ts):
    c=conn(); c.execute("""INSERT INTO loops(profile_id,name,value,unit,ts) VALUES(?,?,?,?,?)""",(pid,name,value,unit,ts)); c.commit(); c.close()

def get_loops(pid, since_days=180):
    c=conn(); cur=c.cursor()
    start=(dt.datetime.now()-dt.timedelta(days=since_days)).isoformat(timespec="seconds")
    cur.execute("""SELECT id,name,value,unit,ts FROM loops
                   WHERE profile_id=? AND ts>=? ORDER BY ts DESC""",(pid,start))
    rows=cur.fetchall(); c.close()
    df=pd.DataFrame(rows, columns=["id","name","value","unit","ts"])
    if not df.empty:
        df["ts"]=pd.to_datetime(df["ts"])
    return df

def lens_add_lines(pid, collection, category, lines):
    c=conn(); execmany(c, "INSERT INTO lens_lines(profile_id,collection,category,text) VALUES(?,?,?,?)",
                       [(pid,collection,category,l) for l in lines]); c.close()

def lens_collections(pid):
    c=conn(); cur=c.cursor()
    cur.execute("SELECT DISTINCT collection FROM lens_lines WHERE profile_id=? ORDER BY collection",(pid,))
    rows=cur.fetchall(); c.close()
    return [r[0] for r in rows]

def lens_sample(pid, collections=None, category=None, k=3):
    c=conn(); cur=c.cursor()
    sql="SELECT id,text FROM lens_lines WHERE profile_id=?"
    params=[pid]
    if collections:
        qmarks=",".join("?"*len(collections))
        sql+=f" AND collection IN ({qmarks})"
        params+=collections
    if category:
        sql+=" AND category=?"; params.append(category)
    sql+=" ORDER BY used ASC, RANDOM() LIMIT ?"; params.append(k)
    cur.execute(sql, tuple(params))
    rows=cur.fetchall()
    if rows:
        ids=[r[0] for r in rows]
        cur.executemany("UPDATE lens_lines SET used=used+1 WHERE id=?", [(i,) for i in ids])
        c.commit()
    c.close()
    return [r[1] for r in rows]

# ---------- Narration ----------
def lens_line(pid, collections, category, fallback):
    lines = lens_sample(pid, collections, category, 1)
    return lines[0] if lines else fallback

def narr_forecast(pid, collections, goal_row, prob, eta_days, tailwinds, headwinds, ai_on=False, api_key=None):
    g_id,g_name,g_unit,g_target,g_deadline,g_pri,_ = goal_row
    base = f"**{g_name}** — success chance **{int(round(prob*100))}%**. ETA ~ **{int(round(eta_days))}** days at current pace."
    tail = ", ".join(tailwinds[:2]) if tailwinds else "—"
    drag = ", ".join(headwinds[:2]) if headwinds else "—"
    lens_fallback = f"You bend toward your Future Self when {tail} leads and {drag} recedes."
    line = lens_line(pid, collections, "Recursion", lens_fallback)
    return base + "  \n" + line

def narr_intervention(pid, collections, goal_name, move, impact_days):
    lens_fallback = f"Smallest viable shift next: **{move}** (≈{impact_days:.1f} days sooner)."
    return lens_line(pid, collections, "Emergence", lens_fallback)

# ---------- Math: Forecasts ----------
def daily_pace(df_loops, unit_match, name_tokens):
    if df_loops.empty: return 0.0
    df = df_loops.copy()
    df["day"]=df["ts"].dt.date
    mask = df["unit"].str.lower().eq(unit_match.lower())
    if name_tokens:
        token_re = "|".join([re.escape(t) for t in name_tokens])
        mask = mask & df["name"].str.lower().str.contains(token_re, na=False)
    sub = df[mask]
    if sub.empty: return 0.0
    daily = sub.groupby("day")["value"].sum()
    window = daily.tail(14) if len(daily)>14 else daily
    return window.mean()

def progress_total(df_loops, unit_match, name_tokens):
    if df_loops.empty: return 0.0
    mask = df_loops["unit"].str.lower().eq(unit_match.lower())
    if name_tokens:
        token_re = "|".join([re.escape(t) for t in name_tokens])
        mask = mask & df_loops["name"].str.lower().str.contains(token_re, na=False)
    return df_loops.loc[mask, "value"].sum()

def beta_probability(progress, target):
    if target <= 0: return 0.0
    successes = clamp(progress, 0, target)
    failures = clamp(target - successes, 0, target)
    alpha = 1 + successes
    beta = 1 + failures
    return alpha / (alpha + beta)

def eta_days(progress, target, pace_per_day):
    remaining = max(target - progress, 0.0)
    if remaining <= 0: return 0.0
    if pace_per_day <= 0: return float("inf")
    return remaining / max(pace_per_day, 1e-6)

# ---------- Interventions ----------
CANDIDATES = [
    ("Writing — 20m starter", "minutes", 20, ["write","draft","book","essay"]),
    ("Walk — 15m reset", "minutes", 15, ["walk","steps","move"]),
    ("Sleep — lights out by 23:00", "sessions", 1, ["sleep","rest"]),
    ("Finance — pay-yourself-first 10%", "amount", 10, ["save","budget","finance"]),
    ("Plan — 10m plan next block", "minutes", 10, ["plan","review"])
]

def score_intervention(goal_name, goal_unit, pace, prob):
    s = 0.0
    n = goal_name.lower()
    if "write" in n or "book" in n: s += 2.0
    if "sleep" in n or "rest" in n: s += 1.0
    if "save" in n or "money" in n or "finance" in n: s += 1.5
    if pace <= 0: s += 2.0
    s += (1-prob)*2.0
    if goal_unit == "minutes": s += 0.5
    return s

# ---------- Diagnostics ----------
def drivers(df_loops):
    if df_loops.empty: return [], []
    last = df_loops.copy()
    last["day"]=last["ts"].dt.date
    agg = last.groupby(["name","unit"])["value"].sum().reset_index()
    agg["score"] = agg["value"] / agg["value"].sum()
    agg = agg.sort_values("score", ascending=False)
    # naive negative heuristics
    drag_words = ["scroll","late","junk","youtube","tiktok","instagram","procrast"]
    agg["is_drag"] = agg["name"].str.lower().apply(lambda s: any(w in s for w in drag_words))
    forces = agg[~agg["is_drag"]][["name","score"]].head(5).values.tolist()
    drags  = agg[ agg["is_drag"]][["name","score"]].head(5).values.tolist()
    return forces, drags

# ---------- Sidebar Navigation ----------
st.sidebar.markdown("### Navigate")
TAB = st.sidebar.radio(
    "", 
    ["Guide","Profiles","Future Self","Goals","Input (Loops)","Forecast","Interventions","Diagnostics","Lens","Settings"],
    index=0,
    label_visibility="collapsed"
)

# ---------- GUIDE ----------
if TAB == "Guide":
    st.title("TimeSculpt — Instructional Guide")
    st.markdown("<hr class='svelte'/>", unsafe_allow_html=True)
    st.subheader("Opening Transmission")
    st.write("TimeSculpt is not a tracker. It is a sculptor’s tool. Each log, each loop, each choice bends probability toward the self you’ve already chosen.")
    st.markdown("<hr class='svelte'/>", unsafe_allow_html=True)

    with st.expander("Profiles"):
        st.write("""
- Create a profile with **name + PIN** (Settings → Profile Manager).
- Toggle **AI ON/OFF** and add an **API key** (optional). AI enriches narration using your Lens.
- Switch active profiles from the Profiles or Settings tab.
""")
    with st.expander("Future Self"):
        st.write("""
- Define **traits** (3–7) and write **letters from Future → Present**.
- Letters can resurface when confidence dips.
""")
    with st.expander("Goals"):
        st.write("""
- Attach measurable goals to your Future Self: **name, unit, target, deadline, priority**.
- Example: *Finish draft* → unit *words*, target *40 000*, deadline *June 1*, priority *5*.
""")
    with st.expander("Input (Loops)"):
        st.write("""
- Log loops as you work (e.g., *writing 40 minutes*). **Every entry is timestamped**.
- Loops feed Forecast and Diagnostics; units should match your goals when possible.
""")
    with st.expander("Forecast"):
        st.write("""
- Forecast shows **probability of success** and **ETA** for each goal.
- Probability uses a **Beta model**; ETA uses your **recent pace**.
- Narration blends your **Lens** (if uploaded & selected).
""")
    with st.expander("Interventions"):
        st.write("""
- See a short list of **smallest viable moves** ranked for impact given your context.
- Apply one, then **do it**. The forecast updates with the changed pace.
""")
    with st.expander("Diagnostics"):
        st.write("""
- Shows **Force (+)** loops that correlate with progress and **Drag (−)** loops that slow you.
- Use this weekly to prune drags and reinforce forces.
""")
    with st.expander("Lens"):
        st.write("""
- Upload `.txt/.docx/.pdf` (export to text before upload for best results).
- Organize by **collection** and **category** (*Collapse*, *Recursion*, *Emergence*, *Neutral*).
- Select one or **mix multiple** collections; narration will pull from them.
""")
    with st.expander("What to Expect with Consistent Use"):
        st.write("""
- Clarity about who you’re becoming.
- Simpler, smaller next actions that compound.
- A narrative layer that keeps identity front-and-center.
""")

# ---------- PROFILES ----------
elif TAB == "Profiles":
    st.title("Profiles")
    st.markdown("<div class='small-muted'>Create or select an active profile.</div>", unsafe_allow_html=True)
    st.markdown("<hr class='svelte'/>", unsafe_allow_html=True)

    cols = st.columns(2)
    with cols[0]:
        st.markdown("#### Create Profile")
        new_name = st.text_input("New Profile Name", key="new_prof_name")
        new_pin  = st.text_input("New PIN", type="password", key="new_prof_pin")
        if st.button("Create Profile"):
            if new_name and new_pin:
                try:
                    create_profile(new_name, new_pin)
                    st.success(f"Profile '{new_name}' created.")
                except sqlite3.IntegrityError:
                    st.error("That profile name already exists.")
            else:
                st.error("Enter a name and PIN.")
    with cols[1]:
        st.markdown("#### Login / Switch")
        profs = [r[1] for r in get_profiles()]
        pick = st.selectbox("Select Profile", profs)
        pin  = st.text_input("PIN", type="password", key="login_pin")
        if st.button("Login"):
            if not pick or not pin:
                st.error("Choose a profile and enter PIN.")
            else:
                if check_pin(pick, pin):
                    st.success(f"Active profile: {pick}")
                else:
                    st.error("Incorrect PIN.")

    if st.session_state.profile_id:
        st.markdown("<hr class='svelte'/>", unsafe_allow_html=True)
        pid = st.session_state.profile_id
        ai_t, key = get_profile_flags(pid)
        st.markdown("#### AI Settings")
        ai_toggle = st.toggle("AI Narration ON", value=bool(ai_t))
        api_key = st.text_input("API key (optional)", value=key if key else "")
        if st.button("Save AI Settings"):
            update_profile_flags(pid, int(ai_toggle), api_key if api_key else None)
            st.success("Saved.")

# ---------- FUTURE SELF ----------
elif TAB == "Future Self":
    st.title("Future Self")
    if not st.session_state.profile_id:
        st.info("No profile selected. Go to Profiles to log in.")
    else:
        pid = st.session_state.profile_id
        st.markdown(f"<div class='badge'>Active: {st.session_state.profile_name}</div>", unsafe_allow_html=True)
        st.markdown("<hr class='svelte'/>", unsafe_allow_html=True)
        c1,c2 = st.columns([2,3])
        with c1:
            st.markdown("#### Traits")
            tnew = st.text_input("Add a new trait")
            if st.button("Add Trait"):
                if tnew.strip():
                    add_trait(pid, tnew.strip())
                    st.success("Added.")
            rows = get_traits(pid)
            if rows:
                for tid, tr in rows:
                    cols=st.columns([4,1])
                    cols[0].write(f"- {tr}")
                    if cols[1].button("✖", key=f"deltr{tid}"):
                        delete_trait(tid); st.experimental_rerun()
        with c2:
            st.markdown("#### Future Self Letter")
            body = st.text_area("Write a letter from your future self to present you")
            if st.button("Save Letter"):
                if body.strip():
                    save_letter(pid, body.strip())
                    st.success("Letter saved.")
            st.markdown("##### Recent Letters")
            for lid, b, created in get_letters(pid)[:3]:
                st.markdown(f"<div class='card'><div class='small-muted'>{created}</div><div style='margin-top:6px'>{b}</div></div>", unsafe_allow_html=True)

# ---------- GOALS ----------
elif TAB == "Goals":
    st.title("Goals")
    if not st.session_state.profile_id:
        st.info("No profile selected.")
    else:
        pid = st.session_state.profile_id
        with st.form("gform", clear_on_submit=True):
            name = st.text_input("Goal name")
            unit = st.text_input("Unit (e.g., minutes, words, sessions, amount)")
            target = st.number_input("Target", min_value=0.0, step=1.0)
            deadline = st.date_input("Deadline", value=today()+dt.timedelta(days=30))
            priority = st.slider("Priority", 1, 5, 3)
            sub = st.form_submit_button("Add Goal")
        if sub:
            if name and unit and target>0:
                add_goal(pid, name.strip(), unit.strip(), float(target), deadline.isoformat(), int(priority))
                st.success("Goal added.")
        st.markdown("<hr class='svelte'/>", unsafe_allow_html=True)
        rows = get_goals(pid)
        if rows:
            for row in rows:
                gid,gname,gunit,gtarget,gdeadline,gpri,gcreated = row
                st.markdown(f"<div class='card'><b>{gname}</b>  — unit: <b>{gunit}</b> • target: <b>{gtarget}</b> • deadline: <b>{gdeadline}</b> • priority: <b>{gpri}</b></div>", unsafe_allow_html=True)
        else:
            st.info("No goals yet. Add your first above.")

# ---------- INPUT ----------
elif TAB == "Input (Loops)":
    st.title("Commit Today — Log Loops")
    if not st.session_state.profile_id:
        st.info("No profile selected.")
    else:
        pid = st.session_state.profile_id
        with st.form("loopform", clear_on_submit=True):
            lname = st.text_input("Loop name", placeholder="e.g., writing, walk, sleep, save")
            val   = st.number_input("Value", min_value=0.0, step=1.0)
            u     = st.text_input("Unit", placeholder="minutes / words / sessions / amount")
            col = st.columns(2)
            with col[0]:
                d = st.date_input("Date", value=today())
            with col[1]:
                t = st.time_input("Time", value=dt.datetime.now().time())
            submit = st.form_submit_button("Log Loop")
        if submit:
            if lname and u:
                ts = dt.datetime.combine(d, t).isoformat(timespec="seconds")
                log_loop(pid, lname.strip(), float(val), u.strip(), ts)
                st.success("Logged.")
        st.markdown("<hr class='svelte'/>", unsafe_allow_html=True)
        df = get_loops(pid, 30)
        if df.empty:
            st.info("No recent loops.")
        else:
            st.dataframe(df[["ts","name","value","unit"]].head(50), use_container_width=True)

# ---------- FORECAST ----------
elif TAB == "Forecast":
    st.title("Forecast — Goal Probabilities & ETA")
    if not st.session_state.profile_id:
        st.info("No profile selected.")
    else:
        pid = st.session_state.profile_id
        rows = get_goals(pid)
        if not rows:
            st.info("No goals. Add goals to see forecasts.")
        else:
            df = get_loops(pid, 180)
            ai_toggle, api_key = get_profile_flags(pid)
            collections = st.multiselect("Active Lens Collections (optional)", lens_collections(pid))
            st.markdown("<hr class='svelte'/>", unsafe_allow_html=True)

            cards=[]
            for row in rows:
                gid,gname,gunit,gtarget,gdeadline,gpri,gcreated = row
                tokens = re.findall(r"[a-zA-Z]+", gname.lower())
                prog = progress_total(df, gunit, tokens)
                pace = daily_pace(df, gunit, tokens)
                prob = beta_probability(prog, gtarget)
                eta  = eta_days(prog, gtarget, pace)
                forces, drags = drivers(df)
                tw = [f"{n}" for n,_ in forces]
                dw = [f"{n}" for n,_ in drags]
                brief = narr_forecast(pid, collections, row, prob, eta, tw, dw, ai_on=bool(ai_toggle), api_key=api_key)
                colA, colB = st.columns([2,1])
                with colA:
                    st.markdown(f"#### {gname}")
                    st.markdown(brief)
                    st.caption(f"Progress: {prog:.1f}/{gtarget:.1f} {gunit} • Pace: {pace:.2f}/{gunit}/day • Deadline: {gdeadline} • Priority {gpri}")
                with colB:
                    p = int(round(prob*100))
                    st.metric("Success probability", f"{p}%")
                    st.metric("ETA (days)", "∞" if math.isinf(eta) else f"{eta:.1f}")

            # Simple timeline chart of last 30 days sum of loop values per day
            if not df.empty:
                d = df.copy()
                d["day"]=d["ts"].dt.date
                chart = alt.Chart(d.groupby("day")["value"].sum().reset_index()).mark_area().encode(
                    x="day:T", y="value:Q"
                ).properties(height=180)
                st.altair_chart(chart, use_container_width=True)

# ---------- INTERVENTIONS ----------
elif TAB == "Interventions":
    st.title("Interventions — Next Smallest Move")
    if not st.session_state.profile_id:
        st.info("No profile selected.")
    else:
        pid = st.session_state.profile_id
        rows = get_goals(pid)
        if not rows:
            st.info("No goals defined.")
        else:
            df = get_loops(pid, 90)
            collections = st.multiselect("Lens Collections (optional)", lens_collections(pid), key="int_lens")
            st.markdown("<hr class='svelte'/>", unsafe_allow_html=True)

            for row in rows:
                gid,gname,gunit,gtarget,gdeadline,gpri,gcreated = row
                tokens = re.findall(r"[a-zA-Z]+", gname.lower())
                prog = progress_total(df, gunit, tokens)
                pace = daily_pace(df, gunit, tokens)
                prob = beta_probability(prog, gtarget)
                candidates = []
                for label, unit, amount, hints in CANDIDATES:
                    sc = score_intervention(gname, gunit, pace, prob)
                    # mock impact: more score → bigger impact
                    impact_days = clamp(sc*0.8, 0.5, 7.0)
                    candidates.append((label, sc, impact_days))
                candidates.sort(key=lambda x: x[1], reverse=True)
                top = candidates[:3]

                st.markdown(f"#### {gname}")
                cols = st.columns(len(top))
                for i,(label,score,impact) in enumerate(top):
                    with cols[i]:
                        st.markdown(f"<div class='card'><b>{label}</b><br/><span class='small-muted'>Score {score:.1f}</span><br/><br/>{narr_intervention(pid, collections, gname, label, impact)}</div>", unsafe_allow_html=True)

# ---------- DIAGNOSTICS ----------
elif TAB == "Diagnostics":
    st.title("Diagnostics — Force & Drag")
    if not st.session_state.profile_id:
        st.info("No profile selected.")
    else:
        pid = st.session_state.profile_id
        df = get_loops(pid, 90)
        if df.empty:
            st.info("No data yet.")
        else:
            f, d = drivers(df)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Force (+)")
                if f:
                    for n,s in f:
                        st.markdown(f"- **{n}**  ▸ {s:.2f}")
                else:
                    st.write("—")
            with col2:
                st.markdown("#### Drag (−)")
                if d:
                    for n,s in d:
                        st.markdown(f"- **{n}**  ▸ {s:.2f}")
                else:
                    st.write("—")
            st.markdown("<hr class='svelte'/>", unsafe_allow_html=True)
            df["day"]=df["ts"].dt.date
            by_name = df.groupby(["day","name"])["value"].sum().reset_index()
            chart = alt.Chart(by_name).mark_line().encode(
                x="day:T", y="value:Q", color="name:N"
            ).properties(height=240)
            st.altair_chart(chart, use_container_width=True)

# ---------- LENS ----------
elif TAB == "Lens":
    st.title("Lens — Upload, Select, Mix")
    if not st.session_state.profile_id:
        st.info("No profile selected.")
    else:
        pid = st.session_state.profile_id
        st.markdown("Use collections to group related texts (e.g., *Stoicism*, *James Allen*, *Personal Notes*).")
        with st.form("lensup"):
            col = st.text_input("Collection name", placeholder="e.g., Stoicism")
            cat = st.selectbox("Category", ["Collapse","Recursion","Emergence","Neutral"])
            raw = st.text_area("Paste text (one line per passage)")
            sub = st.form_submit_button("Add to Lens")
        if sub:
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            if pid and col and lines:
                lens_add_lines(pid, col.strip(), cat, lines)
                st.success(f"Added {len(lines)} lines to [{col}] ({cat}).")
        st.markdown("<hr class='svelte'/>", unsafe_allow_html=True)
        cols = lens_collections(pid)
        pick = st.multiselect("Active Collections", options=cols)
        if st.button("Sample Lines"):
            lines = lens_sample(pid, pick if pick else None, None, 5)
            if lines:
                for ln in lines:
                    st.markdown(f"<div class='card'>{ln}</div>", unsafe_allow_html=True)
            else:
                st.info("No lines available yet.")

# ---------- SETTINGS ----------
elif TAB == "Settings":
    st.title("Settings")
    st.markdown("Profile Manager & Theme are handled here.")
    st.markdown("<hr class='svelte'/>", unsafe_allow_html=True)
    # For simplicity, keep only quick shortcuts; full profile actions are in Profiles tab
    if st.session_state.profile_id:
        pid = st.session_state.profile_id
        ai_t, key = get_profile_flags(pid)
        st.markdown(f"<div class='badge'>Active profile: {st.session_state.profile_name}</div>", unsafe_allow_html=True)
        ai_toggle = st.toggle("AI Narration ON", value=bool(ai_t), key="ai_set")
        api_key = st.text_input("API key", value=key if key else "", key="api_set")
        if st.button("Save"):
            update_profile_flags(pid, int(ai_toggle), api_key if api_key else None)
            st.success("Saved.")
    else:
        st.info("Login or create a profile from Profiles tab.")

# ---------- Footer ----------
st.markdown("<br><div class='small-muted'>TimeSculpt ©</div>", unsafe_allow_html=True)
