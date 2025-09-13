# ============================================================
# TimeSculpt — Phase 6.3 Final Production (Corrected, Complete)
# Single-file Streamlit app with full features & migrations
# ============================================================

import streamlit as st
import sqlite3, bcrypt, random, json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, time as dtime
from openai import OpenAI

# =========================
# CONFIG & STYLING
# =========================
st.set_page_config(page_title="TimeSculpt", layout="wide")

st.markdown("""
<style>
:root {
  --bg: #0b0f19;
  --card: #111729;
  --fg: #e7eaf3;
  --muted: #a7b0c0;
  --gold: #f6c453;
  --accent: #7aa2ff;
}
html, body, .stApp {
  background: radial-gradient(900px 700px at 95% 5%, #0f1630 0%, var(--bg) 55%);
  color: var(--fg);
  font-size: 18px;
}
h1, h2, h3, h4, h5 { color: var(--fg); }
.small { color: var(--muted); font-size: 0.92rem; }
.card { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 14px 16px; margin: 8px 0; }
.narr { border-left: 4px solid var(--gold); background: rgba(246,196,83,0.08); padding: 12px 14px; border-radius: 8px; color:#ffe9b7; font-style: italic;}
div[data-testid="stMetricValue"] { color: var(--gold) !important; font-weight: 800 !important; }

/* Inputs / labels high contrast */
label, .stMarkdown p, .stMarkdown span, .stMarkdown li, .stText, .stSelectbox, .stDateInput, .stTimeInput {
  color: var(--fg) !important;
}
.stTextInput > div > div > input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div,
.stDateInput input,
.stTimeInput input {
  background: var(--card) !important;
  color: var(--fg) !important;
  border-radius: 10px !important;
  border: 1px solid #2a2f45 !important;
}
.stSlider, .stCheckbox { color: var(--fg) !important; }

/* Tabs: sticky, visible */
.stTabs [role="tablist"] {
  display: flex; flex-wrap: wrap; gap: 0.8rem;
  background: #0f1630; padding: 0.6rem 0.6rem;
  border-bottom: 2px solid #2a2f45;
  position: sticky; top: 0; z-index: 999;
}
.stTabs [role="tab"] {
  color: #cdd3e2 !important; font-weight: 700 !important; padding: 8px 12px !important;
  border-radius: 8px 8px 0 0; background: #111729;
}
.stTabs [role="tab"][aria-selected="true"] {
  color: #f6c453 !important; background: #1a2139;
  border: 1px solid #f6c453; border-bottom: none;
}
</style>
""", unsafe_allow_html=True)

# =========================
# DB INIT + MIGRATIONS
# =========================
DB = "timesculpt.db"

def _exec(conn, q, params=()):
    cur = conn.cursor()
    cur.execute(q, params)
    return cur

def init_db():
    with sqlite3.connect(DB) as conn:
        _exec(conn, """CREATE TABLE IF NOT EXISTS profiles(
            id INTEGER PRIMARY KEY, name TEXT UNIQUE, pin_hash TEXT,
            api_key TEXT, ai_enabled INT DEFAULT 0, ai_model TEXT DEFAULT 'gpt-4o-mini',
            demo INT DEFAULT 0, thresholds TEXT DEFAULT '{}',
            traits TEXT DEFAULT '', rituals TEXT DEFAULT '', letter TEXT DEFAULT ''
        )""")
        _exec(conn, """CREATE TABLE IF NOT EXISTS goals(
            id INTEGER PRIMARY KEY, profile_id INT, name TEXT, target REAL,
            unit TEXT, deadline TEXT, priority INT
        )""")
        _exec(conn, """CREATE TABLE IF NOT EXISTS loops(
            id INTEGER PRIMARY KEY, profile_id INT, category TEXT,
            value REAL, date TEXT, time TEXT
        )""")
        _exec(conn, """CREATE TABLE IF NOT EXISTS future_self(
            id INTEGER PRIMARY KEY, profile_id INT, title TEXT,
            traits TEXT, rituals TEXT, letter TEXT
        )""")
        _exec(conn, """CREATE TABLE IF NOT EXISTS interventions(
            id INTEGER PRIMARY KEY, profile_id INT, description TEXT,
            status TEXT, completed_date TEXT, helpful TEXT, reflection TEXT
        )""")
        _exec(conn, """CREATE TABLE IF NOT EXISTS lens(
            id INTEGER PRIMARY KEY, profile_id INT, passage TEXT,
            category TEXT, active INT DEFAULT 1
        )""")
        conn.commit()

def column_exists(table, column):
    with sqlite3.connect(DB) as conn:
        cols = [r[1] for r in _exec(conn, f"PRAGMA table_info({table})").fetchall()]
        return column in cols

def add_column_if_missing(table, column, coltype, default=None):
    if not column_exists(table, column):
        with sqlite3.connect(DB) as conn:
            _exec(conn, f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
            if default is not None:
                _exec(conn, f"UPDATE {table} SET {column}=?", (default,))
            conn.commit()

def migrate_db():
    init_db()
    add_column_if_missing("loops","time","TEXT","08:00:00")
    add_column_if_missing("profiles","api_key","TEXT","")
    add_column_if_missing("profiles","ai_enabled","INT",0)
    add_column_if_missing("profiles","ai_model","TEXT","gpt-4o-mini")
    add_column_if_missing("profiles","demo","INT",0)
    add_column_if_missing("profiles","thresholds","TEXT","{}")
    add_column_if_missing("profiles","traits","TEXT","")
    add_column_if_missing("profiles","rituals","TEXT","")
    add_column_if_missing("profiles","letter","TEXT","")
    add_column_if_missing("interventions","helpful","TEXT","")
    add_column_if_missing("interventions","reflection","TEXT","")
    add_column_if_missing("lens","active","INT",1)

def save(q, params=()):
    with sqlite3.connect(DB) as conn:
        _exec(conn, q, params); conn.commit()

def fetch(q, params=()):
    with sqlite3.connect(DB) as conn:
        return _exec(conn, q, params).fetchall()

migrate_db()

# =========================
# SESSION HELPERS
# =========================
if "profile" not in st.session_state:
    st.session_state.profile = None
if "active_lens_cats" not in st.session_state:
    st.session_state.active_lens_cats = ["recursion","emergence"]

def current_profile():
    return st.session_state.profile

# =========================
# AI (OpenAI)
# =========================
def get_ai_settings(pid):
    row = fetch("SELECT api_key, ai_enabled, ai_model FROM profiles WHERE id=?", (pid,))
    if not row: return None, 0, "gpt-4o-mini"
    api, on, model = row[0]
    return api, on, model or "gpt-4o-mini"

def get_ai_client(pid):
    api, on, model = get_ai_settings(pid)
    if not api or not on: return None, model
    try:
        return OpenAI(api_key=api), model
    except Exception:
        return None, model

def ai_narration(pid, prompt):
    client, model = get_ai_client(pid)
    if not client: return None
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":"You are a concise, motivating coach. Be specific, actionable, warm."},
                {"role":"user","content":prompt}
            ],
            temperature=0.7
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"(AI error: {e})"

def blended_lens_line(pid, cats=None):
    if cats is None: cats = st.session_state.active_lens_cats or ["recursion","emergence"]
    if not cats: return None
    qmarks = ",".join("?"*len(cats))
    rows = fetch(f"SELECT passage FROM lens WHERE profile_id=? AND active=1 AND category IN ({qmarks})", (pid, *cats))
    return random.choice(rows)[0] if rows else None

# =========================
# DEMO DATA
# =========================
def seed_demo(pid, days=30):
    today = datetime.now().date()
    good = ["write","exercise","study","meditate","walk","sleep_good","save"]
    bad  = ["scroll","late","junk","skip","binge"]
    for i in range(days):
        day = today - timedelta(days=i)
        for _ in range(random.randint(2,4)):
            c = random.choice(good + bad)
            v = random.randint(1,3)
            t = dtime(hour=random.choice([6,8,12,18,21]), minute=random.choice([0,15,30,45]))
            save("INSERT INTO loops(profile_id,category,value,date,time) VALUES(?,?,?,?,?)",
                 (pid, f"[DEMO] {c}", v, day.isoformat(), t.strftime("%H:%M:%S")))
    dl = (datetime.now() + timedelta(days=30)).date().isoformat()
    save("INSERT INTO goals(profile_id,name,target,unit,deadline,priority) VALUES(?,?,?,?,?,?)",
         (pid,"[DEMO] Write 30 sessions",30,"sessions",dl,4))
    for text in ("[DEMO] 7-min writing starter","[DEMO] 15-min walk","[DEMO] Zero-scroll morning"):
        save("INSERT INTO interventions(profile_id,description,status) VALUES(?,?,?)",(pid,text,"pending"))
    save("""INSERT INTO future_self(profile_id,title,traits,rituals,letter)
            VALUES(?,?,?,?,?)""",
         (pid,"[DEMO] The Disciplined Architect",
          "Systems over moods; Consistency over intensity",
          "Morning walk; Evening shutdown",
          "Dear me, stay consistent; small steady steps tilt probability."))
    for p,c in [
        ("Clarity precedes power; mornings are leverage.","recursion"),
        ("Guard what guards you back: sleep and focus.","emergence"),
        ("Return to the smallest next move you can keep.","neutral")
    ]:
        save("INSERT INTO lens(profile_id,passage,category,active) VALUES(?,?,?,1)", (pid, f"[DEMO] {p}", c))

def delete_demo(pid):
    save("DELETE FROM loops WHERE profile_id=? AND category LIKE '[DEMO]%%'", (pid,))
    save("DELETE FROM goals WHERE profile_id=? AND name LIKE '[DEMO]%%'", (pid,))
    save("DELETE FROM interventions WHERE profile_id=? AND description LIKE '[DEMO]%%'", (pid,))
    save("DELETE FROM lens WHERE profile_id=? AND passage LIKE '[DEMO]%%'", (pid,))
    save("DELETE FROM future_self WHERE profile_id=? AND title LIKE '[DEMO]%%'", (pid,))

# =========================
# GUIDE
# =========================
def show_guide():
    st.header("📖 TimeSculpt — In-Depth Guide")
    st.markdown("""
**What TimeSculpt is:** a multi-goal, identity-first system that converts small daily loops into **forecasts**, **interventions**, and **narration** in your voice (Lens or AI).

### Daily
1. **Log Loops** — category, value, date, time. Keep it honest, lightweight.  
2. **Forecast** — gauges + ribbons estimate success by deadline; narration explains *why*.  
3. **Pick One Intervention** — accept a smallest-move (e.g., 7-min starter), complete, reflect.  
4. **Letters Resurface** — if alignment dips, your Future Self nudges you.

### Weekly
- **Diagnostics** — Forces vs Drags; balance ratio.  
- Adjust **Goals** — target, unit, deadline, priority.  
- Refresh **Lens** — passages that keep your voice alive.

### Tabs
- **👤 Profiles** — create/login (PIN); enable AI; model; demo seed; profile traits/rituals/letter.  
- **🌠 Future Self** — identity title; traits; rituals; letter (used in dips).  
- **🎯 Goals** — name, target, unit, deadline, priority.  
- **🔁 Loops** — write/exercise/save/scroll/... + value + date + time.  
- **📈 Forecast** — progress, ETA feel, narration (AI or Lens).  
- **🛠 Interventions** — accept → complete → helpful? + reflection.  
- **📚 Lens** — passages (recursion/emergence/neutral), active blend; narration fallback.  
- **⚖️ Diagnostics** — forces/drags/neutral + ratio + narration.  
- **⚙️ Settings** — AI toggle + API key + model; demo on/off; reset profile data.

> You’re not tracking habits; you’re sculpting identity.
""")

# =========================
# PROFILES
# =========================
def show_profiles():
    st.header("👤 Profiles")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Create")
        name = st.text_input("Profile Name")
        pin  = st.text_input("PIN (4+ digits)", type="password")
        if st.button("Create Profile"):
            if not name or not pin:
                st.error("Name and PIN required.")
            else:
                hashed = bcrypt.hashpw(pin.encode(), bcrypt.gensalt()).decode()
                try:
                    save("INSERT INTO profiles(name,pin_hash) VALUES(?,?)",(name,hashed))
                    st.success("Created. Use Login on the right.")
                except sqlite3.IntegrityError:
                    st.error("Profile name exists.")

    with col2:
        st.subheader("Login / AI / Demo / Profile Info")
        profs = fetch("SELECT id,name FROM profiles ORDER BY name")
        if profs:
            names = [p[1] for p in profs]
            sel_name = st.selectbox("Select Profile", names)
            pin_try = st.text_input("PIN to login", type="password")
            if st.button("Login"):
                row = fetch("SELECT id,pin_hash FROM profiles WHERE name=?", (sel_name,))
                if row and bcrypt.checkpw(pin_try.encode(), row[0][1].encode()):
                    st.session_state.profile = row[0][0]
                    st.success(f"Logged in as {sel_name}")
                else:
                    st.error("Invalid PIN.")

        pid = current_profile()
        if pid:
            row = fetch("SELECT ai_enabled, api_key, ai_model, demo, traits, rituals, letter FROM profiles WHERE id=?", (pid,))
            ai_on, api_key, ai_model, demo_on, p_traits, p_rituals, p_letter = row[0]
            st.markdown("**AI Settings**")
            ai_on_new = st.toggle("Enable AI", bool(ai_on))
            api_key_new = st.text_input("OpenAI API Key", value=api_key or "", type="password")
            model_new = st.selectbox("AI Model", ["gpt-4o-mini","gpt-4o","gpt-3.5-turbo"],
                                     index=["gpt-4o-mini","gpt-4o","gpt-3.5-turbo"].index(ai_model or "gpt-4o-mini"))
            demo_new = st.toggle("Enable Demo Data (30 days)", bool(demo_on))
            st.markdown("**Profile Identity (optional)**")
            p_traits_new  = st.text_area("Profile Traits (one per line)", value=p_traits or "")
            p_rituals_new = st.text_area("Profile Rituals (one per line)", value=p_rituals or "")
            p_letter_new  = st.text_area("Profile Letter to Present Self", value=p_letter or "")

            if st.button("Save Profile Settings"):
                save("""UPDATE profiles SET ai_enabled=?, api_key=?, ai_model=?, demo=?, traits=?, rituals=?, letter=? WHERE id=?""",
                     (1 if ai_on_new else 0, api_key_new, model_new, 1 if demo_new else 0,
                      p_traits_new, p_rituals_new, p_letter_new, pid))
                if demo_new:
                    seed_demo(pid)
                st.success("Saved.")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Delete Demo Data"):
                    delete_demo(pid); st.success("Demo data removed.")
            with c2:
                if st.button("Reset Profile Data (keep profile)"):
                    save("DELETE FROM loops WHERE profile_id=?", (pid,))
                    save("DELETE FROM goals WHERE profile_id=?", (pid,))
                    save("DELETE FROM interventions WHERE profile_id=?", (pid,))
                    save("DELETE FROM lens WHERE profile_id=?", (pid,))
                    save("DELETE FROM future_self WHERE profile_id=?", (pid,))
                    st.success("Cleared profile data.")

# =========================
# FUTURE SELF
# =========================
def show_future():
    st.header("🌠 Future Self")
    pid = current_profile()
    if not pid: st.info("Select a profile."); return
    title  = st.text_input("Identity Title (e.g., The Disciplined Architect)")
    traits = st.text_area("Traits (one per line)")
    rituals= st.text_area("Rituals (one per line)")
    letter = st.text_area("Letter to Present Self (used when alignment dips)")
    if st.button("Save Future Self"):
        save("""INSERT INTO future_self(profile_id,title,traits,rituals,letter) VALUES(?,?,?,?,?)""",
             (pid,title,traits,rituals,letter))
        st.success("Saved.")
    rows = fetch("SELECT title,traits,rituals,letter FROM future_self WHERE profile_id=? ORDER BY id DESC LIMIT 5", (pid,))
    if rows:
        st.subheader("Recent")
        for t,tr,ri,le in rows:
            with st.expander(t or "Entry"):
                st.write("**Traits**"); st.code(tr or "")
                st.write("**Rituals**"); st.code(ri or "")
                st.write("**Letter**"); st.write(le or "")

# =========================
# GOALS
# =========================
def show_goals():
    st.header("🎯 Goals")
    pid = current_profile()
    if not pid: st.info("Select a profile."); return
    name = st.text_input("Goal Name")
    target = st.number_input("Target", min_value=0.0, step=1.0)
    unit = st.text_input("Unit (sessions, pages, km, £, etc.)")
    deadline = st.date_input("Deadline").isoformat()
    priority = st.slider("Priority", 1, 5, 3)
    if st.button("Save Goal"):
        save("""INSERT INTO goals(profile_id,name,target,unit,deadline,priority) VALUES(?,?,?,?,?,?)""",
             (pid,name,target,unit,deadline,priority))
        st.success("Goal saved.")
    rows = fetch("SELECT id,name,target,unit,deadline,priority FROM goals WHERE profile_id=? ORDER BY deadline",(pid,))
    if rows:
        st.subheader("Your Goals")
        for gid,n,t,u,dl,pr in rows:
            st.write(f"• **{n}** — target {t} {u} by {dl} (priority {pr})")

# =========================
# LOOPS
# =========================
def show_loops():
    st.header("🔁 Loops")
    pid = current_profile()
    if not pid: st.info("Select a profile."); return
    c = st.text_input("Category / Tag (e.g., write, exercise, save, scroll)")
    v = st.number_input("Value", min_value=0.0, step=1.0)
    d = st.date_input("Date").isoformat()
    t = st.time_input("Time", value=dtime(8,0)).strftime("%H:%M:%S")
    if st.button("Log Loop"):
        save("INSERT INTO loops(profile_id,category,value,date,time) VALUES(?,?,?,?,?)",
             (pid,c,v,d,t))
        st.success("Loop saved.")
    rows = fetch("""SELECT category,value,date,time FROM loops WHERE profile_id=?
                    ORDER BY date DESC, time DESC LIMIT 30""",(pid,))
    if rows:
        st.subheader("Recent")
        for c1,v1,d1,t1 in rows:
            st.write(f"• {d1} {t1} — **{c1}** (+{v1})")

# =========================
# FORECAST
# =========================
def show_forecast():
    st.header("📈 Forecast")
    pid = current_profile()
    if not pid: st.info("Select a profile."); return
    goals = fetch("SELECT id,name,target,unit,deadline,priority FROM goals WHERE profile_id=?", (pid,))
    if not goals: st.info("Add a goal first."); return

    since = (datetime.now().date() - timedelta(days=30)).isoformat()
    loop_sums = fetch("""SELECT category, SUM(value) FROM loops
                         WHERE profile_id=? AND date>=? GROUP BY category""",(pid,since))
    total_effort = sum([r[1] or 0 for r in loop_sums]) or 0

    for gid, name, target, unit, deadline, pr in goals:
        pct = 0.0
        if target and target > 0:
            pct = min(1.0, total_effort / target)
        # Gauge
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct*100.0,
            number={'suffix': "%"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': '#f6c453'},
                   'bgcolor': "#1a2139",
                   'borderwidth': 1,
                   'bordercolor': "#2a2f45"}
        ))
        fig_g.update_layout(title=f"{name} — Success Likelihood", paper_bgcolor="#0f1630", font={'color': '#e7eaf3'})
        st.plotly_chart(fig_g, use_container_width=True)
        # Trend (dummy sim based on current pct)
        x = list(range(14))
        y = [max(0, min(1, pct + random.uniform(-0.05, 0.05))) for _ in x]
        fig_l = px.line(x=x,y=y,labels={'x':'Days','y':'Probability'}, title=f"{name} — 2-week Trajectory")
        st.plotly_chart(fig_l, use_container_width=True)
        st.metric(f"{name} Progress toward {target} {unit}", f"{pct*100:.1f}%")

        # Narration: AI or Lens
        fs_letter = fetch("""SELECT letter FROM future_self WHERE profile_id=? ORDER BY id DESC LIMIT 1""",(pid,))
        base = f"Goal: {name}. 30-day effort={int(total_effort)}/{int(target)} {unit}. Estimated success ~{int(pct*100)}% by {deadline}."
        if fs_letter and fs_letter[0][0]:
            base += " If alignment dips, echo: " + (fs_letter[0][0][:120] + ("…" if len(fs_letter[0][0])>120 else ""))
        narr = ai_narration(pid, base)
        if not narr:
            narr = blended_lens_line(pid, None) or "Return to the smallest next move. Momentum compounds."
        st.markdown(f"<div class='narr'>{narr}</div>", unsafe_allow_html=True)

# =========================
# INTERVENTIONS
# =========================
def show_interventions():
    st.header("🛠 Interventions")
    pid = current_profile()
    if not pid: st.info("Select a profile."); return
    desc = st.text_input("Intervention Description (the smallest next move)")
    if st.button("Accept / Add"):
        if desc:
            save("INSERT INTO interventions(profile_id,description,status) VALUES(?,?,?)",(pid,desc,"accepted"))
            st.success("Intervention added.")

    rows = fetch("""SELECT id,description,status,completed_date,helpful,reflection
                    FROM interventions WHERE profile_id=? ORDER BY id DESC""",(pid,))
    if not rows:
        st.info("No interventions yet."); return

    for iid, d, status, cdate, helpful, reflection in rows:
        st.write(f"**{d}** — {status if status else 'pending'}")
        cols = st.columns([1,1,3])
        with cols[0]:
            if status != "completed":
                if st.button("Complete", key=f"complete_{iid}"):
                    save("UPDATE interventions SET status=?, completed_date=? WHERE id=?",
                         ("completed", datetime.now().isoformat(), iid))
                    st.rerun()
        with cols[1]:
            if status == "completed":
                idx = 0 if (helpful or "Yes") == "Yes" else 1
                helpful_sel = st.selectbox("Helpful?", ["Yes","No"], index=idx, key=f"h{iid}")
        with cols[2]:
            if status == "completed":
                refl_txt = st.text_input("Reflection", value=reflection or "", key=f"r{iid}")
                if st.button("Save Feedback", key=f"save_{iid}"):
                    save("UPDATE interventions SET helpful=?, reflection=? WHERE id=?",
                         (helpful_sel, refl_txt, iid))
                    st.success("Saved.")
    hint = blended_lens_line(pid, ["recursion","neutral"]) or "Try a 7-minute starter; finish with a one-line summary."
    st.markdown(f"<div class='small'>Hint: {hint}</div>", unsafe_allow_html=True)

# =========================
# LENS
# =========================
def show_lens():
    st.header("📚 Lens")
    pid = current_profile()
    if not pid: st.info("Select a profile."); return
    passage = st.text_area("Add Passage")
    cat = st.selectbox("Category", ["recursion","emergence","neutral"])
    active = st.checkbox("Active", True)
    if st.button("Save Passage"):
        if passage:
            save("INSERT INTO lens(profile_id,passage,category,active) VALUES(?,?,?,?)",
                 (pid, passage, cat, 1 if active else 0))
            st.success("Saved.")
    rows = fetch("SELECT id,passage,category,active FROM lens WHERE profile_id=? ORDER BY id DESC",(pid,))
    if rows:
        st.subheader("Passages")
        for lid,p,c,a in rows:
            st.markdown(f"**[{c}]** {'🟢' if a else '⚪'} — {p}")
    st.subheader("Active Blend")
    cats = st.multiselect("Blend categories", ["recursion","emergence","neutral"], default=st.session_state.active_lens_cats)
    st.session_state.active_lens_cats = cats
    line = blended_lens_line(pid, cats)
    if line:
        st.markdown(f"<div class='narr'>{line}</div>", unsafe_allow_html=True)

# =========================
# DIAGNOSTICS
# =========================
def show_diag():
    st.header("⚖️ Diagnostics")
    pid = current_profile()
    if not pid: st.info("Select a profile."); return
    window = st.selectbox("Window", ["Last 7 days","Last 30 days"], index=1)
    days = 7 if window.startswith("Last 7") else 30
    since = (datetime.now().date() - timedelta(days=days)).isoformat()
    rows = fetch("SELECT category,value FROM loops WHERE profile_id=? AND date>=?", (pid, since))
    if not rows:
        st.info("No loops yet."); return

    forces, drags, neutral = {}, {}, {}
    force_kw = ["write","exercise","save","sleep","study","meditate","walk","water","focus","plan"]
    drag_kw  = ["scroll","late","junk","skip","procrastinate","smoke","drink","binge","doom"]
    for cat, val in rows:
        c = cat.lower()
        if any(k in c for k in force_kw): forces[cat] = forces.get(cat,0)+val
        elif any(k in c for k in drag_kw): drags[cat] = drags.get(cat,0)+val
        else: neutral[cat] = neutral.get(cat,0)+val

    if forces:
        st.plotly_chart(px.bar(x=list(forces.keys()),y=list(forces.values()),
                               title="Forces (+)", labels={'x':'Loop','y':'Value'}),
                        use_container_width=True)
    if drags:
        st.plotly_chart(px.bar(x=list(drags.keys()),y=list(drags.values()),
                               title="Drags (–)", labels={'x':'Loop','y':'Value'}),
                        use_container_width=True)

    tf, td = sum(forces.values()), sum(drags.values())
    if tf+td > 0:
        ratio = tf/(tf+td)
        st.metric("Forces/Drags Balance", f"{ratio:.2f}")
        narr = ai_narration(pid, f"Diagnostics: Forces={tf}, Drags={td}, Ratio={ratio:.2f}. Offer one sentence of coaching.")
        if not narr:
            narr = blended_lens_line(pid, ["emergence","neutral"]) or "Trim one drag; double down on a force."
        st.markdown(f"<div class='narr'>{narr}</div>", unsafe_allow_html=True)

# =========================
# SETTINGS
# =========================
def show_settings():
    st.header("⚙️ Settings")
    pid = current_profile()
    if not pid:
        st.info("Select a profile."); return

    row = fetch("SELECT ai_enabled, api_key, ai_model, demo FROM profiles WHERE id=?", (pid,))
    ai_on, api_key, ai_model, demo_on = row[0]

    st.subheader("AI")
    ai_on_new = st.toggle("Enable AI", bool(ai_on))
    api_key_new = st.text_input("OpenAI API Key", value=api_key or "", type="password")
    model_new = st.selectbox("AI Model", ["gpt-4o-mini","gpt-4o","gpt-3.5-turbo"],
                             index=["gpt-4o-mini","gpt-4o","gpt-3.5-turbo"].index(ai_model or "gpt-4o-mini"))

    st.subheader("Demo Mode")
    demo_new = st.toggle("Enable Demo Data (30 days)", bool(demo_on))

    st.subheader("Maintenance")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save Settings"):
            save("UPDATE profiles SET ai_enabled=?, api_key=?, ai_model=?, demo=? WHERE id=?",
                 (1 if ai_on_new else 0, api_key_new, model_new, 1 if demo_new else 0, pid))
            if demo_new:
                seed_demo(pid)
            st.success("Saved.")
    with c2:
        if st.button("Delete Demo Data"):
            delete_demo(pid); st.success("Demo data removed.")

    st.markdown("<div class='small'>Tabs are sticky at the top. If inputs ever look invisible, reload the page to refresh the theme.</div>", unsafe_allow_html=True)

# =========================
# MAIN (ALL TABS VISIBLE)
# =========================
tabs = st.tabs([
    "📖 Guide",
    "👤 Profiles",
    "🌠 Future Self",
    "🎯 Goals",
    "🔁 Loops",
    "📈 Forecast",
    "🛠 Interventions",
    "📚 Lens",
    "⚖️ Diagnostics",
    "⚙️ Settings"   # ensure Settings is in the list
])

with tabs[0]: show_guide()
with tabs[1]: show_profiles()
with tabs[2]: show_future()
with tabs[3]: show_goals()
with tabs[4]: show_loops()
with tabs[5]: show_forecast()
with tabs[6]: show_interventions()
with tabs[7]: show_lens()
with tabs[8]: show_diag()
with tabs[9]: show_settings()

# Sanity ping
st.write("✅ App loaded — base render OK.")
