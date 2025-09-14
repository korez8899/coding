# ============================================
# TimeSculpt – Phase 6.3 Final Expanded Build
# ============================================

import streamlit as st
import sqlite3, bcrypt, random, os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from openai import OpenAI

# =========================
# CONFIG & STYLING
# =========================
st.set_page_config(page_title="TimeSculpt", layout="wide")

st.markdown("""
<style>
body, .stApp {
    background-color: #0a0a0a;
    color: #e0e0e0;
    font-size: 18px;
}
.stTabs [role="tab"] {
    color: #e0e0e0 !important;
    font-weight: bold;
}
.stTabs [role="tab"][aria-selected="true"] {
    border-bottom: 3px solid gold !important;
}
textarea, input, select {
    background-color: #1a1a1a !important;
    color: #f0f0f0 !important;
    border-radius: 8px !important;
}
div[data-testid="stMetricValue"] {
    color: gold !important;
    font-weight: bold;
}
.reportview-container .main .block-container{
    padding-top: 1rem;
    padding-right: 2rem;
    padding-left: 2rem;
    padding-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# DATABASE
# =========================
DB = "timesculpt.db"

def init_db():
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS profiles(
                        id INTEGER PRIMARY KEY, 
                        name TEXT, 
                        pin_hash TEXT, 
                        api_key TEXT, 
                        ai_enabled INT, 
                        demo INT DEFAULT 0,
                        traits TEXT,
                        loops TEXT,
                        letter TEXT
                    )""")
        c.execute("""CREATE TABLE IF NOT EXISTS goals(
                        id INTEGER PRIMARY KEY, 
                        profile_id INT, 
                        name TEXT, 
                        target REAL, 
                        unit TEXT, 
                        deadline TEXT, 
                        priority INT,
                        milestone TEXT
                    )""")
        c.execute("""CREATE TABLE IF NOT EXISTS loops(
                        id INTEGER PRIMARY KEY, 
                        profile_id INT, 
                        category TEXT, 
                        value REAL, 
                        date TEXT
                    )""")
        c.execute("""CREATE TABLE IF NOT EXISTS future_self(
                        id INTEGER PRIMARY KEY, 
                        profile_id INT, 
                        title TEXT, 
                        traits TEXT, 
                        loops TEXT, 
                        letter TEXT,
                        milestone TEXT
                    )""")
        c.execute("""CREATE TABLE IF NOT EXISTS interventions(
                        id INTEGER PRIMARY KEY, 
                        profile_id INT, 
                        description TEXT, 
                        status TEXT, 
                        completed_date TEXT, 
                        helpful TEXT, 
                        reflection TEXT
                    )""")
        c.execute("""CREATE TABLE IF NOT EXISTS lens(
                        id INTEGER PRIMARY KEY, 
                        profile_id INT, 
                        passage TEXT, 
                        category TEXT
                    )""")
        conn.commit()

def save(q, params=()):
    with sqlite3.connect(DB) as conn:
        conn.execute(q, params)
        conn.commit()

def fetch(q, params=()):
    with sqlite3.connect(DB) as conn:
        return conn.execute(q, params).fetchall()

init_db()

# =========================
# SESSION
# =========================
if "profile" not in st.session_state:
    st.session_state.profile = None

def current_profile():
    return st.session_state.profile

# =========================
# NARRATION ENGINE
# =========================
def get_ai_client(pid):
    prof = fetch("SELECT api_key, ai_enabled FROM profiles WHERE id=?", (pid,))
    if prof and prof[0][0] and prof[0][1]:
        return OpenAI(api_key=prof[0][0])
    return None

def ai_narration(pid, prompt, categories=None):
    client = get_ai_client(pid)
    if not client: return None
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}]
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"(AI error: {e})"

def blended_lens_line(pid, cats):
    if not cats: return None
    rows = fetch("SELECT passage FROM lens WHERE profile_id=? AND category IN (%s)" %
                 ",".join("?"*len(cats)), [pid]+cats)
    if not rows: return None
    return random.choice(rows)[0]

# =========================
# DEMO DATA
# =========================
def seed_demo(pid):
    today = datetime.now().date()

    # Wipe existing demo data first
    save("DELETE FROM loops WHERE profile_id=? AND category LIKE '[DEMO]%'", (pid,))
    save("DELETE FROM goals WHERE profile_id=? AND name LIKE '[DEMO]%'", (pid,))
    save("DELETE FROM interventions WHERE profile_id=? AND description LIKE '[DEMO]%'", (pid,))
    save("DELETE FROM lens WHERE profile_id=? AND passage LIKE '[DEMO]%'", (pid,))

    # Insert demo goals
    demo_goals = [
        ("[DEMO] Finish Book", 50, "pages", today + timedelta(days=30), 1),
        ("[DEMO] Daily Workout", 20, "sessions", today + timedelta(days=14), 2),
        ("[DEMO] Meditate", 15, "sessions", today + timedelta(days=10), 3),
    ]
    for name, target, unit, deadline, priority in demo_goals:
        save("INSERT INTO goals(profile_id, name, target, unit, deadline, priority) VALUES(?,?,?,?,?,?)",
             (pid, name, target, unit, deadline.isoformat(), priority))

    # Insert demo loops (progress logs)
    for i in range(15):  # 15 days of demo logs
        d = today - timedelta(days=15-i)
        save("INSERT INTO loops(profile_id, category, value, date) VALUES(?,?,?,?)",
             (pid, "[DEMO] Finish Book", random.randint(1, 5), d.isoformat()))
        save("INSERT INTO loops(profile_id, category, value, date) VALUES(?,?,?,?)",
             (pid, "[DEMO] Daily Workout", 1, d.isoformat() if i % 2 == 0 else None))
        save("INSERT INTO loops(profile_id, category, value, date) VALUES(?,?,?,?)",
             (pid, "[DEMO] Meditate", 1, d.isoformat() if i % 3 == 0 else None))

    # Insert demo interventions
    demo_interventions = [
        ("[DEMO] Write 20m daily", "pending"),
        ("[DEMO] Stretch after workout", "completed"),
        ("[DEMO] Evening reflection journaling", "pending"),
    ]
    for desc, status in demo_interventions:
        save("INSERT INTO interventions(profile_id, description, status) VALUES(?,?,?)", (pid, desc, status))

    # Insert demo lens passages
    demo_lens = [
        ("[DEMO] Each dawn bends toward clarity.", "recursion"),
        ("[DEMO] Small actions converge into irreversible momentum.", "emergence"),
        ("[DEMO] Even silence sculpts.", "neutral"),
    ]
    for passage, cat in demo_lens:
        save("INSERT INTO lens(profile_id, passage, category) VALUES(?,?,?)", (pid, passage, cat))


# =========================
# GUIDE TAB
# =========================
def show_guide():
    st.header("📖 TimeSculpt Guide")
    st.markdown("""
Welcome to **TimeSculpt** — a system for sculpting your future self.

---

### 🔑 Profiles
Create a profile with a name + PIN. Each profile stores **your entire field** — goals, loops, interventions, lens, and future self identity.

### 🌠 Future Self
Define your trajectory:  
- **Traits →** who you are.  
- **Loops →** key actions your Future Self repeats.  
- **Letter →** your guidance to yourself.  
- **Milestone →** a defining checkpoint of becoming.

### 🎯 Goals
Concrete targets with measurable units, deadlines, and milestones. Goals provide **scaffolding for your loops**.

### 🔄 Loops
Daily actions. Each loop strengthens or drags your trajectory. Loops fuel Forecast and Diagnostics.

### 📈 Forecast
Charts show your trajectory. Narration (AI or Lens) translates data into insight. Forecast weaves your loops and goals into a timeline.

### 🛠️ Interventions
Strategic actions. Plan them, mark them complete, reflect on their helpfulness. Interventions **bend the arc of probability**.

### 📚 Lens
Passages you feed into the system. They sculpt the narration style, shaping how feedback returns to you.

### ⚖️ Diagnostics
Reveals **Forces (+)** and **Drags (–)**. Calculates a balance ratio. Narration surfaces the hidden shape of your habits.

### ⚙️ Settings
Control AI use, API keys, demo data, and resets.  
""")

# =========================
# PROFILES TAB
# =========================
def show_profiles():
    st.header("👤 Profiles")
    name = st.text_input("Profile Name")
    pin = st.text_input("PIN", type="password")
    traits = st.text_area("Traits (comma separated)")
    loops = st.text_area("Loops (comma separated)")
    letter = st.text_area("Letter to Self")

    if st.button("Create Profile"):
        if name and pin:
            hashed = bcrypt.hashpw(pin.encode(), bcrypt.gensalt()).decode()
            save("INSERT INTO profiles(name,pin_hash,traits,loops,letter) VALUES(?,?,?,?,?)",
                 (name,hashed,traits,loops,letter))
            st.success("Profile created.")

    profs = fetch("SELECT id,name FROM profiles")
    if profs:
        sel = st.selectbox("Select Profile",[p[1] for p in profs])
        if st.button("Login"):
            row = fetch("SELECT id FROM profiles WHERE name=?",(sel,))
            if row: 
                st.session_state.profile=row[0][0]
                st.success(f"Logged in as {sel}")

# =========================
# FUTURE SELF TAB
# =========================
def show_future():
    st.header("🌠 Future Self")
    pid = current_profile()
    if not pid:
        st.info("Select a profile to define your Future Self.")
        return

    # Retrieve stored data
    data = fetch("SELECT title, traits, loops, letter, obstacles, milestones FROM future_self WHERE profile_id=?", (pid,))
    title, traits, loops, letter, obstacles, milestones = (data[0] if data else ("","","","","",""))

    # Inputs with unique keys
    title = st.text_input("Title", title, key=f"future_title_{pid}")
    traits = st.text_area("Traits (comma separated)", traits, key=f"future_traits_{pid}")
    loops = st.text_area("Loops (habits/actions, comma separated)", loops, key=f"future_loops_{pid}")
    letter = st.text_area("Letter to Self", letter, key=f"future_letter_{pid}")
    obstacles = st.text_area("Obstacles", obstacles, key=f"future_obstacles_{pid}")
    milestones = st.text_area("Milestones (comma separated)", milestones, key=f"future_milestones_{pid}")

    if st.button("Save Future Self", key=f"save_future_{pid}"):
        if data:
            save("UPDATE future_self SET title=?, traits=?, loops=?, letter=?, obstacles=?, milestones=? WHERE profile_id=?",
                 (title, traits, loops, letter, obstacles, milestones, pid))
        else:
            save("INSERT INTO future_self(profile_id, title, traits, loops, letter, obstacles, milestones) VALUES(?,?,?,?,?,?,?)",
                 (pid, title, traits, loops, letter, obstacles, milestones))
        st.success("Future Self updated!")


# =========================
# GOALS TAB
# =========================
def show_goals():
    st.header("🎯 Goals")
    pid=current_profile()
    if not pid: st.info("Select a profile"); return
    g=st.text_input("Goal Name")
    t=st.number_input("Target",step=1.0)
    u=st.text_input("Unit")
    d=st.date_input("Deadline")
    p=st.slider("Priority",1,5,3)
    m=st.text_input("Milestone")
    if st.button("Save Goal"):
        save("INSERT INTO goals(profile_id,name,target,unit,deadline,priority,milestone) VALUES(?,?,?,?,?,?,?)",
             (pid,g,t,u,d.isoformat(),p,m))
        st.success("Goal saved.")

# =========================
# LOOPS TAB
# =========================
def show_loops():
    st.header("🔄 Loops")
    pid=current_profile()
    if not pid: st.info("Select a profile"); return
    c=st.text_input("Category")
    v=st.number_input("Value",step=1.0)
    d=st.date_input("Date")
    if st.button("Log Loop"):
        save("INSERT INTO loops(profile_id,category,value,date) VALUES(?,?,?,?)",
             (pid,c,v,d.isoformat()))
        st.success("Loop saved.")

# =========================
# FORECAST TAB
# =========================
def show_forecast():
    pid = current_profile()
    if not pid:
        st.info("Select a profile to see forecast.")
        return

    st.header("📈 Forecast")

    # Get goals for this profile
    goals = fetch("SELECT id, name, target, unit, deadline, priority FROM goals WHERE profile_id=?", (pid,))

    if not goals:
        st.warning("No goals yet. Add some in the Goals tab.")
        return

    # Calculate progress for each goal
    today = datetime.now().date()
    for gid, name, target, unit, deadline, priority in goals:
        loops = fetch("SELECT value, date FROM loops WHERE profile_id=? AND category LIKE ?", (pid, f"%{name}%"))
        total = sum([float(v) for v, d in loops if v])

        try:
            progress = (total / float(target)) * 100 if target else 0
        except ZeroDivisionError:
            progress = 0

        # --- Gauge chart ---
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=progress,
            title={'text': f"{name} Progress", 'font': {'size': 18, 'color': "#e0e0e0"}},
            number={'font': {'size': 28, 'color': "#ffd700"}},
            gauge={
                'axis': {'range': [0, 100], 'tickfont': {'color': "#e0e0e0"}},
                'bar': {'color': "#ffd700"},
                'bgcolor': "#1c1c1c",
                'borderwidth': 2,
                'bordercolor': "#2a2f45"
            }
        ))
        fig_g.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="#0a0a0a",
            font=dict(color="#e0e0e0", size=14)
        )

        st.plotly_chart(fig_g, use_container_width=True, key=f"forecast_gauge_{pid}_{gid}")

        # --- Trend chart (progress over time) ---
        if loops:
            df = pd.DataFrame(loops, columns=["value", "date"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.groupby("date").sum().reset_index()

            fig_t = px.line(
                df, x="date", y="value",
                title=f"{name} – Progress Over Time",
                markers=True
            )
            fig_t.update_traces(line_color="#ffd700", marker=dict(size=6))
            fig_t.update_layout(
                plot_bgcolor="#0a0a0a",
                paper_bgcolor="#0a0a0a",
                font=dict(color="#e0e0e0"),
                margin=dict(l=20, r=20, t=40, b=20)
            )

            st.plotly_chart(fig_t, use_container_width=True, key=f"forecast_trend_{pid}_{gid}")

        # --- Narration (Lens / AI) ---
        if get_ai_client(pid):
            narration = ai_narration(pid, f"Goal '{name}' progress is {progress:.2f}%. Forecast its outcome.")
        else:
            narration = blended_lens_line(pid, ["recursion", "emergence"]) or "Your path is unfolding."
        
        st.markdown(f"**Narration:** {narration}")
        st.markdown("---")

# =========================
# INTERVENTIONS TAB
# =========================
def show_interventions():
    st.header("🛠️ Interventions")
    pid=current_profile()
    if not pid: st.info("Select a profile"); return
    d=st.text_input("Description")
    if st.button("Add Intervention"):
        save("INSERT INTO interventions(profile_id,description,status) VALUES(?,?,?)",(pid,d,"pending"))
    rows=fetch("SELECT id,description,status,completed_date,helpful,reflection FROM interventions WHERE profile_id=?",(pid,))
    for iid,desc,status,cd,helpful,ref in rows:
        st.write(f"**{desc}** — {status}")
        if status!="completed":
            if st.button("Complete",key=f"c{iid}"):
                save("UPDATE interventions SET status=?,completed_date=? WHERE id=?",("completed",datetime.now().isoformat(),iid))
                st.rerun()
        if status=="completed":
            h=st.selectbox("Helpful?",["Yes","No"],key=f"h{iid}",index=0 if not helpful else ["Yes","No"].index(helpful))
            r=st.text_input("Reflection",value=ref or "",key=f"r{iid}")
            if st.button("Save Feedback",key=f"s{iid}"):
                save("UPDATE interventions SET helpful=?,reflection=? WHERE id=?",(h,r,iid))
                st.success("Feedback saved.")

# =========================
# LENS TAB
# =========================
def show_lens():
    st.header("📚 Lens")
    pid=current_profile()
    if not pid: st.info("Select a profile"); return
    passage=st.text_area("Passage")
    cat=st.selectbox("Category",["recursion","emergence","neutral"])
    if st.button("Add Passage"):
        save("INSERT INTO lens(profile_id,passage,category) VALUES(?,?,?)",(pid,passage,cat))
    rows=fetch("SELECT passage,category FROM lens WHERE profile_id=?",(pid,))
    for p,c in rows:
        st.markdown(f"**{c}:** {p}")

# =========================
# DIAGNOSTICS TAB
# =========================
def show_diag():
    st.header("⚖️ Diagnostics")
    pid=current_profile()
    if not pid: st.info("Select a profile"); return
    loops=fetch("SELECT category,value,date FROM loops WHERE profile_id=?",(pid,))
    if not loops: st.info("No loops."); return
    forces,drags,neutral={}, {}, {}
    fk=["write","exercise","save","sleep","study","meditate","walk","water"]
    dk=["scroll","late","junk","skip","procrastinate","smoke","drink"]
    for c,v,d in loops:
        cl=c.lower()
        if any(w in cl for w in fk): forces[c]=forces.get(c,0)+v
        elif any(w in cl for w in dk): drags[c]=drags.get(c,0)+v
        else: neutral[c]=neutral.get(c,0)+v
    if forces: st.plotly_chart(px.bar(x=list(forces.keys()),y=list(forces.values()),title="Forces (+)",labels={"x":"Loop","y":"Value"}))
    if drags: st.plotly_chart(px.bar(x=list(drags.keys()),y=list(drags.values()),title="Drags (-)",labels={"x":"Loop","y":"Value"}))
    tf,td=sum(forces.values()),sum(drags.values())
    if tf+td>0:
        ratio=tf/(tf+td)
        st.metric("Forces/Drags Balance",f"{ratio:.2f}")
        narr=ai_narration(pid,f"Forces {tf}, Drags {td}, Ratio {ratio:.2f}")
        if narr: st.markdown(f"*{narr}*")

# =========================
# SETTINGS TAB
# =========================
def show_settings():
    st.header("⚙️ Settings")
    pid=current_profile()
    if not pid: st.info("Select a profile"); return
    ai_on=st.checkbox("Enable AI",key=f"ai_{pid}")
    api=st.text_input("OpenAI API Key",type="password",key=f"api_{pid}")
    demo=st.checkbox("Enable Demo Data",key=f"demo_{pid}")
    if st.button("Save Settings",key=f"sav_{pid}"):
        save("UPDATE profiles SET ai_enabled=?,api_key=?,demo=? WHERE id=?",(1 if ai_on else 0,api,demo,pid))
        if demo: seed_demo(pid)
        st.success("Settings saved.")

# =========================
# MAIN
# =========================
tabs=st.tabs([
    "📖 Guide","👤 Profiles","🌠 Future Self","🎯 Goals","🔄 Loops",
    "📈 Forecast","🛠️ Interventions","📚 Lens","⚖️ Diagnostics","⚙️ Settings"
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
