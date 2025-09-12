import streamlit as st
import sqlite3, bcrypt, random, os
import plotly.express as px
from datetime import datetime, timedelta
from openai import OpenAI

# =========================
# CONFIG
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
</style>
""", unsafe_allow_html=True)

# =========================
# DB
# =========================
DB = "timesculpt.db"

def init_db():
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS profiles(id INTEGER PRIMARY KEY, name TEXT, pin_hash TEXT, api_key TEXT, ai_enabled INT, demo INT DEFAULT 0)")
        c.execute("CREATE TABLE IF NOT EXISTS goals(id INTEGER PRIMARY KEY, profile_id INT, name TEXT, target REAL, unit TEXT, deadline TEXT, priority INT)")
        c.execute("CREATE TABLE IF NOT EXISTS loops(id INTEGER PRIMARY KEY, profile_id INT, category TEXT, value REAL, date TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS future_self(id INTEGER PRIMARY KEY, profile_id INT, title TEXT, traits TEXT, rituals TEXT, letter TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS interventions(id INTEGER PRIMARY KEY, profile_id INT, description TEXT, status TEXT, completed_date TEXT, helpful TEXT, reflection TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS lens(id INTEGER PRIMARY KEY, profile_id INT, passage TEXT, category TEXT)")
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
# AI
# =========================
def get_ai_client(pid):
    prof = fetch("SELECT api_key, ai_enabled FROM profiles WHERE id=?", (pid,))
    if prof and prof[0][0] and prof[0][1]:
        return OpenAI(api_key=prof[0][0])
    return None

def ai_narration(pid, prompt):
    client = get_ai_client(pid)
    if not client: return None
    try:
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}])
        return resp.choices[0].message.content
    except Exception as e:
        return f"(AI error: {e})"

def blended_lens_line(pid, cats):
    rows = fetch("SELECT passage FROM lens WHERE profile_id=? AND category IN (%s)" %
                 ",".join("?"*len(cats)), [pid]+cats)
    if not rows: return None
    return random.choice(rows)[0]

# =========================
# DEMO DATA
# =========================
def seed_demo(pid):
    today = datetime.now()
    for i in range(30):
        d = today - timedelta(days=i)
        save("INSERT INTO loops(profile_id, category, value, date) VALUES(?,?,?,?)",
             (pid, random.choice(["write","exercise","scroll","meditate"]),
              random.randint(1,3), d.date().isoformat()))
    save("INSERT INTO goals(profile_id,name,target,unit,deadline,priority) VALUES(?,?,?,?,?,?)",
         (pid,"Finish Book",50,"pages",(today+timedelta(days=30)).isoformat(),5))
    save("INSERT INTO interventions(profile_id,description,status) VALUES(?,?,?)",
         (pid,"Write 20m daily","pending"))
    save("INSERT INTO lens(profile_id,passage,category) VALUES(?,?,?)",
         (pid,"Each dawn bends toward clarity.","recursion"))

# =========================
# GUIDE
# =========================
def show_guide():
    st.header("📖 TimeSculpt Guide")
    st.markdown("""
Welcome to **TimeSculpt** — a system for sculpting your future self.

---

### 🔑 Profiles
Create and log into a profile with a name + PIN. Each profile stores its own goals, loops, lens, and future self.

### 🌠 Future Self
Define who you want to become: **title, traits, rituals, and letters to your present self.**

### 🎯 Goals
Add measurable goals (target + deadline). Goals drive forecasts.

### 🔄 Loops
Log daily actions (loops). Each loop strengthens or drags your momentum.

### 📈 Forecast
Charts and AI/Lens narration show your likely trajectory toward goals.

### 🛠️ Interventions
Plan actions to change trajectory. Mark them complete, then log if they were helpful and reflect.

### 📚 Lens
Upload or write passages. These shape narration and give identity-flavored feedback.

### ⚖️ Diagnostics
See which habits are **Forces (+)** and which are **Drags (-)**. A balance ratio is calculated. Narration summarizes the trends.

### ⚙️ Settings
Enable AI, enter an API key, select demo data, or disable demo.
""")

# =========================
# PROFILES
# =========================
def show_profiles():
    st.header("👤 Profiles")
    name = st.text_input("Profile Name")
    pin = st.text_input("PIN", type="password")
    if st.button("Create Profile"):
        if name and pin:
            hashed = bcrypt.hashpw(pin.encode(), bcrypt.gensalt()).decode()
            save("INSERT INTO profiles(name,pin_hash) VALUES(?,?)",(name,hashed))
            st.success("Profile created.")
    profs = fetch("SELECT id,name FROM profiles")
    if profs:
        sel = st.selectbox("Select Profile",[p[1] for p in profs])
        if st.button("Login"):
            row = fetch("SELECT id FROM profiles WHERE name=?",(sel,))
            if row: st.session_state.profile=row[0][0]; st.success(f"Logged in as {sel}")

# =========================
# FUTURE SELF
# =========================
def show_future():
    st.header("🌠 Future Self")
    pid=current_profile()
    if not pid: st.info("Select a profile"); return
    title=st.text_input("Title")
    traits=st.text_area("Traits")
    rituals=st.text_area("Rituals")
    letter=st.text_area("Letter")
    if st.button("Save Future Self"):
        save("INSERT INTO future_self(profile_id,title,traits,rituals,letter) VALUES(?,?,?,?,?)",
             (pid,title,traits,rituals,letter))
        st.success("Future Self saved.")

# =========================
# GOALS
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
    if st.button("Save Goal"):
        save("INSERT INTO goals(profile_id,name,target,unit,deadline,priority) VALUES(?,?,?,?,?,?)",
             (pid,g,t,u,d.isoformat(),p))
        st.success("Goal saved.")

# =========================
# LOOPS
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
# FORECAST
# =========================
def show_forecast():
    st.header("📈 Forecast")
    pid=current_profile()
    if not pid: st.info("Select a profile"); return
    goals=fetch("SELECT id,name,target,unit,deadline FROM goals WHERE profile_id=?",(pid,))
    if not goals: st.info("No goals."); return
    for gid,name,t,u,dl in goals:
        loops=fetch("SELECT SUM(value) FROM loops WHERE profile_id=?",(pid,))
        done=loops[0][0] or 0
        perc=done/t if t else 0
        st.plotly_chart(px.line(x=[0,1],y=[done,t],title=name))
        st.metric(f"{name} Progress", f"{perc*100:.1f}%")
        narr=ai_narration(pid,f"Goal {name}, progress {perc*100:.1f}%")
        if narr: st.markdown(f"*{narr}*")

# =========================
# INTERVENTIONS
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
# LENS
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
# DIAGNOSTICS
# =========================
def show_diag():
    st.header("⚖️ Diagnostics")
    pid=current_profile()
    if not pid: st.info("Select a profile"); return
    loops=fetch("SELECT category,value FROM loops WHERE profile_id=?",(pid,))
    if not loops: st.info("No loops."); return
    forces,drags,neutral={}, {}, {}
    fk=["write","exercise","save","sleep","study","meditate","walk","water"]
    dk=["scroll","late","junk","skip","procrastinate","smoke","drink"]
    for c,v in loops:
        cl=c.lower()
        if any(w in cl for w in fk): forces[c]=forces.get(c,0)+v
        elif any(w in cl for w in dk): drags[c]=drags.get(c,0)+v
        else: neutral[c]=neutral.get(c,0)+v
    if forces: st.plotly_chart(px.bar(x=list(forces.keys()),y=list(forces.values()),title="Forces (+)"))
    if drags: st.plotly_chart(px.bar(x=list(drags.keys()),y=list(drags.values()),title="Drags (-)"))
    tf,td=sum(forces.values()),sum(drags.values())
    if tf+td>0:
        ratio=tf/(tf+td)
        st.metric("Forces/Drags Balance",f"{ratio:.2f}")
        narr=ai_narration(pid,f"Forces {tf}, Drags {td}, Ratio {ratio:.2f}")
        if narr: st.markdown(f"*{narr}*")

# =========================
# SETTINGS
# =========================
def show_settings():
    st.header("⚙️ Settings")
    pid=current_profile()
    if not pid: st.info("Select a profile"); return
    ai_on=st.checkbox("Enable AI")
    api=st.text_input("OpenAI API Key",type="password")
    demo=st.checkbox("Enable Demo Data")
    if st.button("Save Settings"):
        save("UPDATE profiles SET ai_enabled=?,api_key=?,demo=? WHERE id=?",(1 if ai_on else 0,api,demo,pid))
        if demo: seed_demo(pid)
        st.success("Settings saved.")

# =========================
# MAIN
# =========================
tabs=st.tabs(["📖 Guide","👤 Profiles","🌠 Future Self","🎯 Goals","🔄 Loops","📈 Forecast","🛠️ Interventions","📚 Lens","⚖️ Diagnostics","⚙️ Settings"])
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
