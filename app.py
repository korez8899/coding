# ================================================
# TimeSculpt – Phase 6.2 (Production, Single File)
# ================================================
import os, sqlite3, bcrypt, random
from datetime import datetime, date, time as dtime, timedelta
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ---------------------------
# Theme / Global Page Config
# ---------------------------
st.set_page_config(page_title="TimeSculpt", layout="wide")
THEME_BG = "#0B0E14"
THEME_PANEL = "#121826"
THEME_TEXT = "#EDEDED"   # off-white
THEME_MUTED = "#C8CCD6"  # lighter gray labels
THEME_GOLD = "#FFD166"   # accents
THEME_EMPH = "#A5D6FF"   # cool accent

CSS = f"""
<style>
/* Base */
body, .stApp {{ background: {THEME_BG}; color: {THEME_TEXT}; }}
.block-container {{ padding-top: 2rem; }}
/* Tabs: ensure always visible */
[role="tablist"] {{
  display: flex !important;
  border-bottom: 2px solid #2A3244;
  margin-bottom: 1rem;
}}
[role="tablist"] [role="tab"] {{
  color: {THEME_TEXT}; font-weight: 700; padding: 0.5rem 1rem;
}}
[role="tablist"] [role="tab"][aria-selected="true"] {{
  border-bottom: 3px solid {THEME_GOLD};
  color: {THEME_GOLD};
}}
/* Inputs, buttons, narration, etc. keep same styles */
label, .stMarkdown p {{ color: {THEME_MUTED} !important; font-weight: 600; }}
.stTextInput > div > div input, .stTextArea textarea, .stNumberInput input,
.stDateInput input, .stTimeInput input {{
  color: {THEME_TEXT} !important; background: {THEME_PANEL} !important; border-radius: 10px;
  border: 1px solid #2A3244;
}}
.stButton > button {{
  background: linear-gradient(45deg, {THEME_GOLD}, #E8B85B);
  color: #121212; font-weight: 800; border: none; border-radius: 12px; padding: 0.5rem 1rem;
}}
.narration-box {{
  border: 2px solid {THEME_GOLD}; border-radius: 12px; padding: 12px; margin: 8px 0;
  background: rgba(255, 209, 102, 0.08); color: {THEME_TEXT};
  font-style: italic;
}}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------
# DB helpers + schema guard
# ---------------------------
DB_PATH = "timesculpt.db"

def conn():
    return sqlite3.connect(DB_PATH)

def fetch(q, args=()):
    c = conn(); cur = c.cursor()
    cur.execute(q, args)
    rows = cur.fetchall()
    c.close()
    return rows

def save(q, args=()):
    c = conn(); cur = c.cursor()
    cur.execute(q, args)
    c.commit(); c.close()

def init_db():
    c = conn(); cur = c.cursor()

    # profiles
    cur.execute("""
    CREATE TABLE IF NOT EXISTS profiles(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        pin_hash TEXT,
        ai_toggle INTEGER DEFAULT 0,
        api_key TEXT,
        model TEXT,
        thresholds TEXT,
        demo_enabled INTEGER DEFAULT 0
    )""")

    # future_self
    cur.execute("""
    CREATE TABLE IF NOT EXISTS future_self(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        title TEXT,
        traits TEXT,
        rituals TEXT,
        letter TEXT,
        created_at TEXT
    )""")
    try: cur.execute("ALTER TABLE future_self ADD COLUMN letter TEXT")
    except sqlite3.OperationalError: pass
    try: cur.execute("ALTER TABLE future_self ADD COLUMN created_at TEXT")
    except sqlite3.OperationalError: pass

    # goals
    cur.execute("""
    CREATE TABLE IF NOT EXISTS goals(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        name TEXT,
        target REAL,
        unit TEXT,
        deadline TEXT,
        priority INTEGER,
        loop_tags TEXT,
        created_at TEXT
    )""")
    for col in ("loop_tags", "created_at"):
        try: cur.execute(f"ALTER TABLE goals ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError: pass

    # loops
    cur.execute("""
    CREATE TABLE IF NOT EXISTS loops(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        category TEXT,
        value REAL,
        unit TEXT,
        timestamp TEXT
    )""")
    for col in ("unit", "timestamp"):
        try: cur.execute(f"ALTER TABLE loops ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError: pass

    # interventions
    cur.execute("""
    CREATE TABLE IF NOT EXISTS interventions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        goal_id INTEGER,
        description TEXT,
        status TEXT,
        helpful INTEGER,
        reflection TEXT,
        created_at TEXT,
        completed_at TEXT
    )""")
    for col in ("helpful","reflection","created_at","completed_at"):
        try: cur.execute(f"ALTER TABLE interventions ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError: pass

    # lens
    cur.execute("""
    CREATE TABLE IF NOT EXISTS lens(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        passage TEXT,
        category TEXT,
        collection TEXT,
        active INTEGER DEFAULT 1,
        created_at TEXT
    )""")
    for col in ("collection","active","created_at"):
        try: cur.execute(f"ALTER TABLE lens ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError: pass
    # normalize active to int if text
    try: cur.execute("UPDATE lens SET active = 1 WHERE active IS NULL")
    except: pass

    c.commit(); c.close()

init_db()

# ---------------------------
# Session & small utilities
# ---------------------------
if "profile_id" not in st.session_state: st.session_state.profile_id = None
def current_profile(): return st.session_state.profile_id

def ai_enabled(pid):
    row = fetch("SELECT ai_toggle FROM profiles WHERE id=?", (pid,))
    return bool(row and str(row[0][0]) == "1")

def get_threshold(pid, default=40):
    row = fetch("SELECT thresholds FROM profiles WHERE id=?", (pid,))
    try:
        return int(row[0][0]) if row and row[0][0] is not None else default
    except: return default

def get_demo_flag(pid):
    row = fetch("SELECT demo_enabled FROM profiles WHERE id=?", (pid,))
    return bool(row and str(row[0][0]) == "1")

def get_api_key_and_model(pid):
    row = fetch("SELECT api_key, model FROM profiles WHERE id=?", (pid,))
    if row:
        api = row[0][0]
        model = row[0][1] if row[0][1] else "gpt-4o-mini"
        return api, model
    return None, "gpt-4o-mini"

def ai_narration(pid, prompt):
    api_key, model = get_api_key_and_model(pid)
    if not (pid and ai_enabled(pid) and api_key and OpenAI):
        return None
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":"You are TimeSculpt, a crisp, empowering, grounded coach. Be concise, warm, and concrete."},
                {"role":"user","content": prompt}
            ],
            max_tokens=160, temperature=0.7
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None

def lens_line(pid, categories=None):
    # prefer active passages; then any
    if categories:
        cat_q = ",".join("?" for _ in categories)
        rows = fetch(f"SELECT passage FROM lens WHERE profile_id=? AND (active='1' OR active=1) AND LOWER(category) IN ({cat_q}) ORDER BY RANDOM() LIMIT 1",
                     tuple([pid] + [c.lower() for c in categories]))
        if rows: return rows[0][0]
    rows = fetch("SELECT passage FROM lens WHERE profile_id=? AND (active='1' OR active=1) ORDER BY RANDOM() LIMIT 1", (pid,))
    if rows: return rows[0][0]
    rows = fetch("SELECT passage FROM lens WHERE profile_id=? ORDER BY RANDOM() LIMIT 1", (pid,))
    return rows[0][0] if rows else ""

# ---------------------------
# Demo seeding (1 month)
# ---------------------------
def seed_demo():
    # if DemoUser exists, skip
    demo = fetch("SELECT id FROM profiles WHERE name=?", ("DemoUser",))
    if demo: return
    h = bcrypt.hashpw("demo".encode(), bcrypt.gensalt()).decode()
    save("INSERT INTO profiles(name,pin_hash,ai_toggle,model,thresholds,demo_enabled) VALUES (?,?,?,?,?,?)",
         ("DemoUser", h, 0, "gpt-4o-mini", "40", 1))
    pid = fetch("SELECT id FROM profiles WHERE name=?", ("DemoUser",))[0][0]
    # Future Self
    save("""INSERT INTO future_self(profile_id,title,traits,rituals,letter,created_at)
            VALUES(?,?,?,?,?,?)""",
         (pid, "The Disciplined Architect",
          "Focused, Patient, Resilient",
          "Morning writing, Evening walk",
          "Stay the course. You already are the one who finishes.",
          datetime.now().isoformat()))
    # Goals with loop tags mapping
    goals = [
        ("Finish Book", 10000, "words", (date.today()+timedelta(days=30)).isoformat(), 5, "writing,planning"),
        ("Lose Weight", 2, "kg",      (date.today()+timedelta(days=30)).isoformat(), 4, "exercise,walk,sleep_good,water"),
        ("Save Money", 500, "£",       (date.today()+timedelta(days=30)).isoformat(), 3, "save_invest,budget_check")
    ]
    for name, target, unit, dl, pr, tags in goals:
        save("""INSERT INTO goals(profile_id,name,target,unit,deadline,priority,loop_tags,created_at)
                VALUES(?,?,?,?,?,?,?,?)""",
             (pid, name, target, unit, dl, pr, tags, datetime.now().isoformat()))
    # Loops 30 days
    loop_pool = [
        ("writing","mins"), ("planning","mins"), ("exercise","sessions"),
        ("walk","mins"), ("sleep_good","hrs"), ("water","glasses"),
        ("save_invest","£"), ("budget_check","count"),
        ("scroll","mins"), ("late_sleep","hrs"), ("junk_food","units")
    ]
    for i in range(30):
        ts_day = datetime.now() - timedelta(days=i)
        for cat,u in loop_pool:
            val = max(0, int(random.gauss(3, 2)))
            if random.random()<0.2: val=0
            save("INSERT INTO loops(profile_id,category,value,unit,timestamp) VALUES(?,?,?,?,?)",
                 (pid, cat, float(val), u, (ts_day - timedelta(minutes=random.randint(0,120))).isoformat()))
    # Lens
    passages = [
        ("Small steps carve the stone.", "Recursion", "core"),
        ("Late nights erode clarity.", "Collapse", "core"),
        ("Momentum grows in silence.", "Emergence", "core"),
        ("Attention is the chisel of identity.", "Recursion", "core"),
        ("Guard the morning; it guards your future.", "Emergence", "core")
    ]
    for p,cat,col in passages:
        save("INSERT INTO lens(profile_id,passage,category,collection,active,created_at) VALUES(?,?,?,?,?,?)",
             (pid, p, cat, col, 1, datetime.now().isoformat()))
    # Interventions
    gi = fetch("SELECT id,name FROM goals WHERE profile_id=?", (pid,))
    gmap = {n:i for (i,n) in gi}
    def giid(name): return gmap.get(name, None)
    rows = [
        (giid("Finish Book"), "7-minute writing starter"),
        (giid("Lose Weight"), "15-minute walk after lunch"),
        (giid("Save Money"),  "Pay-yourself-first 10%")
    ]
    for gid,desc in rows:
        if gid:
            save("""INSERT INTO interventions(profile_id,goal_id,description,status,helpful,created_at)
                    VALUES(?,?,?,?,?,?)""",
                 (pid, gid, desc, "completed", 1, datetime.now().isoformat()))

# ---------------------------
# Tabs
# ---------------------------
tabs = st.tabs([
    "📖 Guide","👤 Profiles","🌠 Future Self","🎯 Goals","🔄 Loops",
    "📈 Forecast","🛠 Interventions","⚖️ Diagnostics","📚 Lens","⚙️ Settings"
])

# ---------------------------
# GUIDE
# ---------------------------
with tabs[0]:
    st.title("TimeSculpt — Quick Start & Deep Guide")
    st.markdown("""
**What TimeSculpt is:** a multi-goal, identity-first system that converts small daily loops into progress forecasts,
interventions that actually help, and narrative guidance in your voice (via Lens or AI).
""")
    st.markdown("---")
    st.subheader("Daily Flow")
    st.markdown("""
1. **Log Loops** (🔄): capture writing/exercise/save/etc. with date+time.  
2. **Check Forecast** (📈): see % to hit each goal + smallest next move.  
3. **Accept & Complete an Intervention** (🛠): then record if it helped.  
4. **Let Letters Resurface** (🌠→📈): if alignment dips, your Future Self nudges you.  
""")
    st.subheader("Weekly Flow")
    st.markdown("""
- **Diagnostics** (⚖️): review **Forces vs Drags** and the **balance ratio**.  
- **Refit goals** (🎯): adjust targets/loop tags if needed.  
- **Refresh Lens** (📚): add passages; keep narration alive.  
""")
    st.subheader("Tabs Overview")
    st.markdown("""
- **👤 Profiles**: create/select/delete.  
- **🌠 Future Self**: title, traits, rituals, letter (resurfaces when forecasts dip).  
- **🎯 Goals**: target/unit/deadline/priority + **loop tags** to map which loops count.  
- **🔄 Loops**: category/value/unit + date/time.  
- **📈 Forecast**: gauge + 30-day trend + narration (AI or Lens).  
- **🛠 Interventions**: offer → accept → complete; reflect & mark helpful.  
- **⚖️ Diagnostics**: expanded forces/drags/neutral, ratio metric, narration.  
- **📚 Lens**: passages (category/collection) with **active** toggle.  
- **⚙️ Settings**: AI toggle, API key, model (GPT-4o-mini/4o/4-turbo), threshold, demo toggle.  
""")
    if not current_profile():
        st.info("ℹ️ No profile is active. Create one in **Profiles**, or enable **Demo Data** in **Settings** to preload a sample.")

# ---------------------------
# PROFILES
# ---------------------------
with tabs[1]:
    st.subheader("Profiles")
    colL, colR = st.columns([2,1])
    with colL:
        name = st.text_input("Profile Name")
        pin = st.text_input("PIN (4+ digits)", type="password")
        if st.button("Create Profile"):
            if not name or not pin:
                st.warning("Name and PIN required.")
            else:
                h = bcrypt.hashpw(pin.encode(), bcrypt.gensalt()).decode()
                save("INSERT INTO profiles(name,pin_hash,ai_toggle,model,thresholds,demo_enabled) VALUES(?,?,?,?,?,?)",
                     (name, h, 0, "gpt-4o-mini", "40", 0))
                st.success(f"Profile '{name}' created.")
    with colR:
        profs = fetch("SELECT id,name FROM profiles ORDER BY id DESC")
        if profs:
            choice = st.selectbox("Select Profile", profs, format_func=lambda r: r[1])
            if st.button("Login"):
                st.session_state.profile_id = choice[0]
                st.success(f"Active profile: {choice[1]}")
            if st.button("Delete Selected"):
                save("DELETE FROM profiles WHERE id=?", (choice[0],))
                st.session_state.profile_id = None
                st.success("Profile deleted.")
        else:
            st.caption("No profiles yet.")

# ---------------------------
# FUTURE SELF
# ---------------------------
with tabs[2]:
    st.subheader("Future Self")
    pid = current_profile()
    if not pid:
        st.info("Select a profile.")
    else:
        with st.form("fs_form"):
            t = st.text_input("Title")
            traits = st.text_area("Traits (comma separated)")
            rituals = st.text_area("Rituals (comma separated)")
            letter = st.text_area("Letter from Future Self")
            if st.form_submit_button("Save Future Self"):
                save("""INSERT INTO future_self(profile_id,title,traits,rituals,letter,created_at)
                        VALUES(?,?,?,?,?,?)""",
                     (pid, t, traits, rituals, letter, datetime.now().isoformat()))
                st.success("Saved.")
        fs = fetch("""SELECT title,traits,rituals,letter,created_at
                      FROM future_self WHERE profile_id=? ORDER BY id DESC LIMIT 1""",(pid,))
        if fs:
            t, tr, r, l, ca = fs[0]
            st.markdown(f"**Title:** {t or '—'}")
            st.markdown(f"**Traits:** {tr or '—'}")
            st.markdown(f"**Rituals:** {r or '—'}")
            st.markdown("**Letter:**")
            st.markdown(f"<div class='narration-box'>{(l or '').strip()}</div>", unsafe_allow_html=True)
            st.caption(f"Created: {ca}")

# ---------------------------
# GOALS
# ---------------------------
with tabs[3]:
    st.subheader("Goals")
    pid = current_profile()
    if not pid:
        st.info("Select a profile.")
    else:
        with st.form("goal_form"):
            gname = st.text_input("Goal Name")
            unit = st.text_input("Unit (e.g., words, kg, £, mins)")
            target = st.number_input("Target", min_value=0.0, step=1.0)
            deadline = st.date_input("Deadline", value=date.today()+timedelta(days=30))
            priority = st.slider("Priority (1 low → 5 high)", 1, 5, 3)
            loop_tags = st.text_input("Loop Tags (comma separated, e.g., writing,planning)")
            if st.form_submit_button("Save Goal"):
                save("""INSERT INTO goals(profile_id,name,target,unit,deadline,priority,loop_tags,created_at)
                        VALUES(?,?,?,?,?,?,?,?)""",
                     (pid, gname, target, unit, str(deadline), priority, loop_tags, datetime.now().isoformat()))
                st.success("Goal saved.")
        gl = fetch("""SELECT id,name,target,unit,deadline,priority,loop_tags
                      FROM goals WHERE profile_id=? ORDER BY id DESC""",(pid,))
        if gl:
            st.markdown("### Current Goals")
            for gid, gn, tar, u, dl, pr, tags in gl:
                st.markdown(f"- **{gn}** → {tar} {u} by {dl} · Priority {pr} · Tags: `{tags or '—'}`")

# ---------------------------
# LOOPS
# ---------------------------
with tabs[4]:
    st.subheader("Loops")
    pid = current_profile()
    if not pid:
        st.info("Select a profile.")
    else:
        with st.form("loop_form"):
            lcat = st.text_input("Category (e.g., writing, exercise, save_invest, scroll)")
            lval = st.number_input("Value", min_value=0.0, step=1.0)
            lunit = st.text_input("Unit (mins, £, sessions, etc.)")
            ldate = st.date_input("Date", value=date.today())
            ltime = st.time_input("Time", value=(datetime.now().time()))
            if st.form_submit_button("Log Loop"):
                ts = datetime.combine(ldate, ltime).isoformat()
                save("INSERT INTO loops(profile_id,category,value,unit,timestamp) VALUES(?,?,?,?,?)",
                     (pid, lcat, lval, lunit, ts))
                st.success("Loop logged.")
        recent = fetch("""SELECT category,value,unit,timestamp
                          FROM loops WHERE profile_id=? ORDER BY id DESC LIMIT 12""",(pid,))
        if recent:
            st.markdown("### Recent")
            for c, v, u, ts in recent:
                st.markdown(f"- **{c}** → {v:g} {u or ''} · {ts}")

# ---------------------------
# FORECAST
# ---------------------------
with tabs[5]:
    st.subheader("Forecast")
    pid = current_profile()
    if not pid:
        st.info("Select a profile.")
    else:
        goals = fetch("""SELECT id,name,target,unit,deadline,priority,loop_tags
                         FROM goals WHERE profile_id=? ORDER BY priority DESC, id ASC""",(pid,))
        if not goals:
            st.info("Add a goal in the Goals tab.")
        else:
            threshold = get_threshold(pid, 40)
            for gid, gn, tar, u, dl, pr, tags in goals:
                tagset = [t.strip().lower() for t in (tags or "").split(",") if t.strip()]
                if tagset:
                    qs = ",".join("?"*len(tagset))
                    loops = fetch(f"""SELECT value, timestamp FROM loops
                                      WHERE profile_id=? AND LOWER(category) IN ({qs})
                                      ORDER BY timestamp ASC""", tuple([pid]+tagset))
                else:
                    # fallback: try matching goal name
                    loops = fetch("""SELECT value, timestamp FROM loops
                                     WHERE profile_id=? AND LOWER(category) LIKE ?
                                     ORDER BY timestamp ASC""", (pid, f"%{gn.lower()}%"))
                prog = sum(float(v or 0) for v,_ in loops) if loops else 0.0
                pct = max(0.0, min(100.0, (prog/float(tar))*100.0 if tar else 0.0))

                st.markdown(f"#### {gn}")
                colA,colB = st.columns([1,2])
                with colA:
                    st.metric("Progress", f"{pct:.1f}%")
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=pct,
                        title={"text": f"{gn}"},
                        gauge={'axis': {'range': [0,100]}, 'bar': {'color': THEME_GOLD}}
                    ))
                    st.plotly_chart(gauge, use_container_width=True, key=f"g-{gid}")
                with colB:
                    # build 30-day daily % trend (synthetic from cumulative loops)
                    today = date.today()
                    days = [today - timedelta(days=i) for i in range(29,-1,-1)]
                    # sum per day
                    per_day = {d.isoformat():0.0 for d in days}
                    for v, ts in loops:
                        dkey = datetime.fromisoformat(ts).date().isoformat()
                        if dkey in per_day: per_day[dkey] += float(v)
                    cum=0.0; trend=[]
                    for d in days:
                        cum += per_day[d.isoformat()]
                        trend.append( (cum/float(tar))*100.0 if tar else 0.0 )
                    line = px.line(x=days, y=[max(0,min(100,t)) for t in trend],
                                   labels={'x':'Date','y':'Success %'},
                                   title=f"{gn} — 30-day Trend")
                    st.plotly_chart(line, use_container_width=True, key=f"t-{gid}")

                base = f"Goal: {gn}. Progress {prog:g}/{tar:g} {u}. Deadline {dl}. Priority {pr}."
                ll = lens_line(pid, categories=["Recursion","Emergence","Neutral"])
                text = ai_narration(pid, base + (" " + ll if ll else "")) or (base + (" " + ll if ll else ""))
                st.markdown(f"<div class='narration-box'>{text}</div>", unsafe_allow_html=True)

                if pct < float(threshold or 40):
                    lf = fetch("""SELECT letter FROM future_self
                                  WHERE profile_id=? ORDER BY id DESC LIMIT 1""",(pid,))
                    if lf and lf[0][0]:
                        st.warning("📜 Future Self Letter resurfaces:")
                        st.markdown(f"<div class='narration-box'>{lf[0][0]}</div>", unsafe_allow_html=True)

# ---------------------------
# INTERVENTIONS
# ---------------------------
with tabs[6]:
    st.subheader("Interventions")
    pid = current_profile()
    if not pid:
        st.info("Select a profile.")
    else:
        gl = fetch("SELECT id,name FROM goals WHERE profile_id=? ORDER BY id DESC", (pid,))
        gmap = {n:i for (i,n) in gl}
        goal_sel = st.selectbox("Attach to Goal", ["—"] + [n for n in gmap.keys()])
        with st.form("int_form"):
            desc = st.text_input("Intervention Description")
            if st.form_submit_button("Offer / Add"):
                if goal_sel != "—" and desc.strip():
                    save("""INSERT INTO interventions(profile_id,goal_id,description,status,created_at)
                            VALUES(?,?,?,?,?)""",(pid, gmap[goal_sel], desc.strip(), "offered", datetime.now().isoformat()))
                    st.success("Intervention added.")
        ints = fetch("""SELECT id,goal_id,description,status,helpful,reflection,created_at,completed_at
                        FROM interventions WHERE profile_id=? ORDER BY id DESC LIMIT 20""",(pid,))
        if ints:
                    for iid, desc, status, completed_date in fetch(
            "SELECT id, description, status, completed_date FROM interventions WHERE profile_id=?", 
            (pid,)
        ):
            st.write(f"**{desc}** — {status}")
            
            # Completion button (only if not already complete)
            if status != "completed":
                if st.button("Complete", key=f"complete_{iid}"):
                    save(
                        "UPDATE interventions SET status=?, completed_date=? WHERE id=?", 
                        ("completed", datetime.now().isoformat(), iid)
                    )
                    st.rerun()
            
            # Feedback form (only for completed items)
            if status == "completed":
                helpful = st.selectbox(
                    "Helpful?", ["Yes", "No"], key=f"helpful_{iid}"
                )
                reflection = st.text_input(
                    "Reflection", key=f"reflection_{iid}"
                )
                if st.button("Save Feedback", key=f"feedback_{iid}"):
                    save(
                        "UPDATE interventions SET helpful=?, reflection=? WHERE id=?", 
                        (helpful, reflection, iid)
                    )
                    st.success("Feedback saved!")


# ---------------------------
# DIAGNOSTICS
# ---------------------------
with tabs[7]:
    st.subheader("Diagnostics")
    pid = current_profile()
    if not pid:
        st.info("Select a profile.")
    else:
        loops = fetch("SELECT category,value FROM loops WHERE profile_id=?", (pid,))
        if not loops:
            st.info("No loops yet.")
        else:
            forces, drags, neutral = {}, {}, {}
            force_keywords = ["write","exercise","save","sleep_good","study","meditate","walk","water","planning","budget_check","save_invest"]
            drag_keywords  = ["scroll","late_sleep","junk","skip","procrastinate","smoke","drink","junk_food"]
            for cat,val in loops:
                c = (cat or "").lower()
                if any(w in c for w in force_keywords):
                    forces[cat] = forces.get(cat,0)+float(val or 0)
                elif any(w in c for w in drag_keywords):
                    drags[cat] = drags.get(cat,0)+float(val or 0)
                else:
                    neutral[cat] = neutral.get(cat,0)+float(val or 0)
            if forces:
                st.plotly_chart(px.bar(x=list(forces.keys()), y=list(forces.values()),
                                       title="Forces (+)", labels={"x":"Loop","y":"Total"}), use_container_width=True)
            if drags:
                st.plotly_chart(px.bar(x=list(drags.keys()), y=list(drags.values()),
                                       title="Drags (−)", labels={"x":"Loop","y":"Total"}), use_container_width=True)
            tf, td = sum(forces.values()), sum(drags.values())
            if tf+td>0:
                ratio = tf/(tf+td)
                st.metric("Forces/Drags Balance", f"{ratio:.2f}")
            prompt = f"Diagnostics: Forces={tf:.1f}, Drags={td:.1f}, Ratio={(tf/(tf+td) if tf+td>0 else 0):.2f}. "
            prompt += "Suggest the smallest corrective move in plain language."
            narr = ai_narration(pid, prompt) or (lens_line(pid, ["Recursion","Emergence","Collapse","Neutral"]) or "")
            if narr:
                st.markdown(f"<div class='narration-box'>{narr}</div>", unsafe_allow_html=True)

# ---------------------------
# LENS
# ---------------------------
with tabs[8]:
    st.subheader("Lens")
    pid = current_profile()
    if not pid:
        st.info("Select a profile.")
    else:
        with st.form("lens_form"):
            psg = st.text_area("Passage")
            cat = st.selectbox("Category", ["Recursion","Emergence","Collapse","Neutral"])
            col = st.text_input("Collection (optional)", value="core")
            act = st.checkbox("Active", value=True)
            if st.form_submit_button("Add Passage"):
                save("""INSERT INTO lens(profile_id,passage,category,collection,active,created_at)
                        VALUES(?,?,?,?,?,?)""",
                     (pid, psg.strip(), cat, col.strip(), 1 if act else 0, datetime.now().isoformat()))
                st.success("Passage saved.")
        # manage
        rows = fetch("""SELECT id, passage, category, collection, active
                        FROM lens WHERE profile_id=? ORDER BY id DESC LIMIT 20""",(pid,))
        if rows:
            st.markdown("### Recent Passages")
            for lid, p, c, col, a in rows:
                colA,colB,colC = st.columns([6,2,2])
                colA.markdown(f"**{c}** · *{col or '—'}* → {p}")
                new_state = colB.selectbox("Active?", ["Yes","No"], index=(0 if str(a)=="1" else 1), key=f"act{lid}")
                if colC.button("Update", key=f"upd{lid}"):
                    save("UPDATE lens SET active=? WHERE id=?", ("1" if new_state=="Yes" else "0", lid))
                    st.success("Updated.")

# ---------------------------
# SETTINGS
# ---------------------------
with tabs[9]:
    st.subheader("Settings")
    pid = current_profile()
    if not pid:
        st.info("Select a profile.")
    else:
        rows = fetch("SELECT ai_toggle, api_key, model, thresholds, demo_enabled FROM profiles WHERE id=?", (pid,))
        ai_t, api_k, mdl, thr, dem = (rows[0] if rows else (0,"","gpt-4o-mini","40",0))
        c1,c2 = st.columns(2)
        with c1:
            ai_toggle = st.checkbox("Enable AI Narration", value=bool(int(ai_t or 0)))
            api_key = st.text_input("OpenAI API Key", type="password", value=api_k or "")
            model = st.selectbox("AI Model", ["gpt-4o-mini","gpt-4o","gpt-4-turbo"], index=(["gpt-4o-mini","gpt-4o","gpt-4-turbo"].index(mdl) if mdl in ["gpt-4o-mini","gpt-4o","gpt-4-turbo"] else 0))
        with c2:
            threshold = st.slider("Letter Resurface Threshold (%)", 0, 100, int(thr or 40))
            demo_toggle = st.checkbox("Enable Demo Data (seed DemoUser)", value=bool(int(dem or 0)))
        if st.button("Save Settings"):
            save("""UPDATE profiles
                    SET ai_toggle=?, api_key=?, model=?, thresholds=?, demo_enabled=?
                    WHERE id=?""",
                 (1 if ai_toggle else 0, api_key, model, str(threshold), 1 if demo_toggle else 0, pid))
            if demo_toggle:
                seed_demo()
            st.success("Settings updated.")
        if not OpenAI and ai_toggle:
            st.warning("`openai` Python package not installed. AI will fall back to Lens narration.")
