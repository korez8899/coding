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
    today = datetime.now()
    # Loops
    categories = ["write","exercise","scroll","meditate","save","drink","study"]
    for i in range(45):  # 1.5 months of loops
        d = today - timedelta(days=i)
        save("INSERT INTO loops(profile_id, category, value, date) VALUES(?,?,?,?)",
             (pid, random.choice(categories),
              random.randint(1,3), d.date().isoformat()))
    # Goals
    save("INSERT INTO goals(profile_id,name,target,unit,deadline,priority,milestone) VALUES(?,?,?,?,?,?,?)",
         (pid,"Finish Book",50,"pages",(today+timedelta(days=30)).isoformat(),5,"Write 20 pages"))
    save("INSERT INTO goals(profile_id,name,target,unit,deadline,priority,milestone) VALUES(?,?,?,?,?,?,?)",
         (pid,"Run 10km",10,"km",(today+timedelta(days=15)).isoformat(),4,"Reach 5km comfortably"))
    # Future self
    save("INSERT INTO future_self(profile_id,title,traits,loops,letter,milestone) VALUES(?,?,?,?,?,?)",
         (pid,"Disciplined Writer","Creative, Focused","Daily writing, Exercise",
          "Keep going. Each day’s line becomes the book.","Publish first draft"))
    # Interventions
    save("INSERT INTO interventions(profile_id,description,status) VALUES(?,?,?)",
         (pid,"Write 20m daily","pending"))
    save("INSERT INTO interventions(profile_id,description,status) VALUES(?,?,?)",
         (pid,"Run every morning","pending"))
    # Lens
    passages = [
        ("Each dawn bends toward clarity.","recursion"),
        ("The steps you repeat carve the person you become.","emergence"),
        ("Neutral moments whisper choices.","neutral")
    ]
    for text, cat in passages:
        save("INSERT INTO lens(profile_id,passage,category) VALUES(?,?,?)",(pid,text,cat))

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
        st.info("Select a profile")
        return

    title = st.text_input("Future Self Title", key=f"title_{pid}")
    traits = st.text_area("Traits (comma separated)", key=f"future_traits_{pid}")
    loops = st.text_area("Loops (comma separated)", key=f"future_loops_{pid}")
    letter = st.text_area("Letter to Self", key=f"future_letter_{pid}")
    milestone = st.text_input("Milestone", key=f"future_milestone_{pid}")

    if st.button("Save Future Self", key=f"save_future_{pid}"):
        save(
            "INSERT INTO future_self(profile_id,title,traits,loops,letter,milestone) VALUES(?,?,?,?,?,?)",
            (pid, title, traits, loops, letter, milestone)
        )
        st.success("Future Self saved.")


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
    st.header("📈 Forecast")
    pid=current_profile()
    if not pid: st.info("Select a profile"); return
    goals=fetch("SELECT id,name,target,unit,deadline,priority,milestone FROM goals WHERE profile_id=?",(pid,))
    if not goals: st.info("No goals."); return
    for gid,name,t,u,dl,p,m in goals:
        loops=fetch("SELECT SUM(value) FROM loops WHERE profile_id=?",(pid,))
        done=loops[0][0] or 0
        perc=done/t if t else 0
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=perc*100,
            title={"text":f"{name} Progress"},
            gauge={
                "axis":{"range":[0,100]},
                "bar":{"color":"gold"},
                "bgcolor":"black",
                "borderwidth":2,
                "bordercolor":"#2a2f45"
            }
        ))
        st.plotly_chart(fig_g,use_container_width=True)
        st.metric(f"{name} %", f"{perc*100:.1f}%")
        narr=ai_narration(pid,f"Goal {name}, progress {perc*100:.1f}%. Milestone: {m}")
        if narr: st.markdown(f"*{narr}*")

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
