import streamlit as st
import sqlite3
import bcrypt
import datetime
import matplotlib.pyplot as plt
import docx
import PyPDF2
import random

st.set_page_config(page_title="TimeSculpt", layout="wide")

st.markdown("""
    <style>
    .stApp {background: linear-gradient(to bottom, #0a0a0f, #111133);}
    section[data-testid="stSidebar"] {background-color: #0f0f1f;}
    label, .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: #FFD700 !important; font-weight: bold;
    }
    input, textarea {
        background-color: #1e1e2e !important; border-radius: 8px !important; color: white !important;
    }
    .card {
        padding: 1rem; border-radius: 10px; margin-bottom: 1rem;
        background-color: #111122; color: white; border: 1px solid #FFD700;
    }
    .highlight {color: #FFD700; font-weight: bold;}
    h1, h2, h3 {color: #FFD700 !important; font-weight: bold;}
    button[kind="primary"] {
        background: linear-gradient(90deg, #FFD700, #FFA500) !important;
        color: black !important; border-radius: 8px !important; font-weight: bold !important;
    }
    #watermark {
        position: fixed; bottom: 10px; right: 20px;
        color: #FFD700; font-size: 14px; opacity: 0.7;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div id="watermark">TimeSculpt</div>', unsafe_allow_html=True)

def init_db():
    conn = sqlite3.connect("timesculpt.db")
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        pin_hash TEXT,
        ai_toggle INTEGER DEFAULT 0,
        api_key TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS goals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        name TEXT,
        target REAL,
        unit TEXT,
        deadline TEXT,
        priority REAL,
        FOREIGN KEY(profile_id) REFERENCES profiles(id))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS loops (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        loop_name TEXT,
        value REAL,
        unit TEXT,
        timestamp TEXT,
        FOREIGN KEY(profile_id) REFERENCES profiles(id))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS traits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        trait TEXT,
        FOREIGN KEY(profile_id) REFERENCES profiles(id))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS letters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        content TEXT,
        FOREIGN KEY(profile_id) REFERENCES profiles(id))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS lens_lines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        line TEXT,
        FOREIGN KEY(profile_id) REFERENCES profiles(id))""")
    conn.commit(); conn.close()
init_db()

def hash_pin(pin): return bcrypt.hashpw(pin.encode(), bcrypt.gensalt()).decode()
def get_profiles():
    conn = sqlite3.connect("timesculpt.db"); cur = conn.cursor()
    cur.execute("SELECT id, name, ai_toggle, api_key FROM profiles"); rows = cur.fetchall()
    conn.close(); return rows
def create_profile(name, pin):
    conn = sqlite3.connect("timesculpt.db"); cur = conn.cursor()
    cur.execute("INSERT INTO profiles (name, pin_hash) VALUES (?, ?)", (name, hash_pin(pin)))
    conn.commit(); conn.close()
def log_loop(profile_id, loop_name, value, unit, dt):
    conn = sqlite3.connect("timesculpt.db"); cur = conn.cursor()
    cur.execute("INSERT INTO loops (profile_id, loop_name, value, unit, timestamp) VALUES (?,?,?,?,?)",
                (profile_id, loop_name, value, unit, dt.isoformat()))
    conn.commit(); conn.close()
def get_loops(profile_id):
    conn = sqlite3.connect("timesculpt.db"); cur = conn.cursor()
    cur.execute("SELECT loop_name, value, unit, timestamp FROM loops WHERE profile_id=?", (profile_id,))
    rows = cur.fetchall(); conn.close(); return rows
def save_lens_line(profile_id, line):
    conn = sqlite3.connect("timesculpt.db"); cur = conn.cursor()
    cur.execute("INSERT INTO lens_lines (profile_id, line) VALUES (?, ?)", (profile_id, line))
    conn.commit(); conn.close()
def get_lens_lines(profile_id):
    conn = sqlite3.connect("timesculpt.db"); cur = conn.cursor()
    cur.execute("SELECT line FROM lens_lines WHERE profile_id=?", (profile_id,))
    rows = [r[0] for r in cur.fetchall()]; conn.close(); return rows
def save_goal(profile_id, name, target, unit, deadline, priority):
    conn = sqlite3.connect("timesculpt.db"); cur = conn.cursor()
    cur.execute("INSERT INTO goals (profile_id, name, target, unit, deadline, priority) VALUES (?,?,?,?,?,?)",
                (profile_id, name, target, unit, deadline, priority))
    conn.commit(); conn.close()
def get_goals(profile_id):
    conn = sqlite3.connect("timesculpt.db"); cur = conn.cursor()
    cur.execute("SELECT name, target, unit, deadline, priority FROM goals WHERE profile_id=?", (profile_id,))
    rows = cur.fetchall(); conn.close(); return rows

tabs = {
    "Guide":"🧭 Guide","Future Self":"🌌 Future Self","Input":"✍️ Input",
    "Forecast":"🔮 Forecast","Interventions":"🎯 Interventions",
    "Diagnostics":"📊 Diagnostics","Lens":"📚 Lens","Settings":"⚙️ Settings"}
choice = st.sidebar.radio("Navigate", list(tabs.keys()), format_func=lambda x: tabs[x])

if choice == "Guide":
    st.title("🧭 TimeSculpt — Instructional Guide")
    st.subheader("Opening Transmission")
    st.markdown("**TimeSculpt is not a tracker. It is a sculptor’s tool.** Each log, each loop, each choice bends probability toward the self you’ve already chosen.")
    st.subheader("📌 Visual Roadmap")
    st.markdown("📂 Profiles → 🌌 Future Self → 🎯 Goals → 🌀 Loops → 📊 Forecast → 🎯 Interventions → 📈 Diagnostics → 📖 Lens → ⚙️ Settings")
    st.subheader("📂 Profiles")
    st.markdown("Create and secure your profile with a name + PIN. Switch between profiles easily. Each profile stores its own traits, goals, loops, and forecasts.")
    st.subheader("🌌 Future Self")
    st.markdown("Define traits and write letters from your Future Self. Attach goals with deadlines and priorities. Example: Trait = Disciplined. Letter = ‘I already finished the book.’")
    st.subheader("🎯 Goals")
    st.markdown("Attach measurable targets to your Future Self. Example: Write 40,000 words by June 1st. Prioritize goals to focus Forecast and Interventions.")
    st.subheader("🌀 Input (Loops)")
    st.markdown("Log daily activities. Even 5 minutes counts. Loops are timestamped and feed Forecast and Diagnostics. Example: Writing 20 minutes → moves Writing Goal forward.")
    st.subheader("📊 Forecast")
    st.markdown("Shows progress toward goals with percentages, countdowns, and ETA projections. Styled charts track momentum. Lens lines echo narrative guidance.")
    st.subheader("🎯 Interventions")
    st.markdown("Suggests the smallest next move to bend trajectory toward your Future Self. Example: Add 200 words today. AI can expand into strategies if enabled.")
    st.subheader("📈 Diagnostics")
    st.markdown("Analyzes your logged loops to show Forces (positive drivers) and Drags (negative patterns). Example: Morning Writing = Force. Late Sleep = Drag.")
    st.subheader("📖 Lens")
    st.markdown("Upload or add guiding lines. These resurface in Forecast, Interventions, and Diagnostics as echoes, shaping narrative and reinforcing identity.")
    st.subheader("⚙️ Settings")
    st.markdown("Manage profiles, AI API key, and enable/disable AI narration. Each profile keeps its own AI toggle and key.")
    st.subheader("✅ Daily Flow Checklist")
    st.markdown("""
    - Morning: 🔮 Check Forecast + Lens echo.  
    - Daytime: 🌀 Log loops + 🎯 Apply interventions.  
    - Evening: 📈 Review Diagnostics + refine 🎯 Goals + write 🌌 Future Self letters.  
    """)
    st.subheader("📈 Expectation")
    st.markdown("With consistent use, TimeSculpt sharpens clarity, aligns identity with action, and provides probability-based forecasts that shape real outcomes.")

elif choice == "Future Self":
    profiles = get_profiles()
    if not profiles: st.warning("No profile found."); st.stop()
    profile_id = profiles[0][0]
    st.header("Traits")
    t = st.text_input("New trait"); 
    if st.button("Save Trait") and t:
        conn = sqlite3.connect("timesculpt.db"); cur = conn.cursor()
        cur.execute("INSERT INTO traits (profile_id, trait) VALUES (?, ?)", (profile_id, t))
        conn.commit(); conn.close(); st.success("Saved")
    conn = sqlite3.connect("timesculpt.db"); cur = conn.cursor()
    cur.execute("SELECT trait FROM traits WHERE profile_id=?", (profile_id,))
    for tr in cur.fetchall(): st.markdown(f"- {tr[0]}")
    st.header("Letter")
    l = st.text_area("From your Future Self"); 
    if st.button("Save Letter") and l:
        conn = sqlite3.connect("timesculpt.db"); cur = conn.cursor()
        cur.execute("INSERT INTO letters (profile_id, content) VALUES (?, ?)", (profile_id, l))
        conn.commit(); conn.close(); st.success("Saved")
    st.header("Goals")
    gname = st.text_input("Goal name"); gtarget = st.number_input("Target", 0.0)
    gunit = st.text_input("Unit"); gdeadline = st.date_input("Deadline")
    gprio = st.slider("Priority",1,10,5)
    if st.button("Save Goal") and gname:
        save_goal(profile_id,gname,gtarget,gunit,str(gdeadline),gprio); st.success("Saved")
    for g in get_goals(profile_id):
        st.markdown(f"<div class='card'>**{g[0]}** — {g[1]} {g[2]}, Deadline: {g[3]}, Priority: {g[4]}</div>", unsafe_allow_html=True)

elif choice == "Input":
    profiles = get_profiles()
    if not profiles: st.warning("No profile."); st.stop()
    profile_id = profiles[0][0]
    st.header("Log Loops")
    lname = st.text_input("Loop name"); val = st.number_input("Value",0.0)
    unit = st.text_input("Unit"); d = st.date_input("Date", datetime.date.today())
    t = st.time_input("Time", datetime.datetime.now().time())
    if st.button("Log") and lname:
        log_loop(profile_id,lname,val,unit,datetime.datetime.combine(d,t))
        st.success("Loop saved")
    st.subheader("Recent")
    for l in get_loops(profile_id)[-5:]:
        st.markdown(f"<div class='card'>**{l[0]}**: {l[1]} {l[2]} at {l[3]}</div>", unsafe_allow_html=True)

elif choice == "Forecast":
    profiles = get_profiles()
    if not profiles: st.warning("No profile."); st.stop()
    profile_id = profiles[0][0]; goals=get_goals(profile_id); loops=get_loops(profile_id)
    st.header("Forecast")
    if not goals: st.info("No goals yet."); st.stop()
    for g in goals:
        total=sum(l[1] for l in loops if l[0]==g[0]); pct= min(100,(total/g[1]*100) if g[1]>0 else 0)
        deadline=datetime.date.fromisoformat(g[3]); days=(deadline-datetime.date.today()).days
        eta=(datetime.date.today()+datetime.timedelta(days=(g[1]-total)/max(1,total/ max(1,len(loops))))) if loops else None
        st.markdown(f"**{g[0]}** — {pct:.1f}% complete")
        st.progress(pct/100)
        st.markdown(f"Deadline: {days} days left" if days>=0 else "⚠️ Missed")
        if eta: st.markdown(f"ETA at current pace: {eta}")
    if loops:
        days=[datetime.datetime.fromisoformat(l[3]).date() for l in loops]; vals=[l[1] for l in loops]
        fig,ax=plt.subplots(); ax.plot(days,vals,marker="o",color="#FFD700")
        ax.set_title("Loop Progress",color="white"); ax.tick_params(colors="white")
        fig.patch.set_facecolor("#111133"); ax.set_facecolor("#111133"); st.pyplot(fig)
    lines=get_lens_lines(profile_id); 
    if lines: st.markdown(f"*Lens echo:* {random.choice(lines)}")

elif choice == "Interventions":
    profiles=get_profiles()
    if not profiles: st.warning("No profile."); st.stop()
    profile_id=profiles[0][0]; goals=get_goals(profile_id); loops=get_loops(profile_id)
    st.header("Interventions")
    if not goals: st.info("No goals."); st.stop()
    goal=min(goals,key=lambda g:g[4])
    total=sum(l[1] for l in loops if l[0]==goal[0]); gap=goal[1]-total
    move=f"Add {min(gap, goal[1]*0.1):.1f} {goal[2]} to {goal[0]}"
    st.markdown(f"<div class='card'><span class='highlight'>Top Move:</span> {move}</div>", unsafe_allow_html=True)
    lines=get_lens_lines(profile_id); 
    if lines: st.markdown(f"*Lens echo:* {random.choice(lines)}")

elif choice == "Diagnostics":
    profiles=get_profiles()
    if not profiles: st.warning("No profile."); st.stop()
    profile_id=profiles[0][0]; loops=get_loops(profile_id)
    st.header("Diagnostics")
    if not loops: st.info("No loops."); st.stop()
    forces={}; drags={}
    for l in loops:
        if "sleep" in l[0].lower(): drags[l[0]]=drags.get(l[0],0)-1
        else: forces[l[0]]=forces.get(l[0],0)+1
    labels=list(forces.keys())+list(drags.keys())
    values=list(forces.values())+list(drags.values())
    colors=["#50C878"]*len(forces)+["#DC143C"]*len(drags)
    fig,ax=plt.subplots(); ax.bar(labels,values,color=colors); ax.axhline(0,color="white")
    ax.set_title("Force & Drag",color="white"); ax.tick_params(colors="white")
    fig.patch.set_facecolor("#111133"); ax.set_facecolor("#111133"); st.pyplot(fig)
    lines=get_lens_lines(profile_id); 
    if lines: st.markdown(f"*Lens echo:* {random.choice(lines)}")

elif choice == "Lens":
    profiles=get_profiles()
    if not profiles: st.warning("No profile."); st.stop()
    profile_id=profiles[0][0]; st.header("Lens")
    uploaded=st.file_uploader("Upload .txt/.docx/.pdf",type=["txt","docx","pdf"])
    if uploaded:
        if uploaded.name.endswith(".txt"): text=uploaded.read().decode("utf-8")
        elif uploaded.name.endswith(".docx"): text="\n".join([p.text for p in docx.Document(uploaded).paragraphs])
        else: text="\n".join([page.extract_text() for page in PyPDF2.PdfReader(uploaded).pages if page.extract_text()])
        for line in text.split("\n"):
            if line.strip(): save_lens_line(profile_id,line.strip()); st.success("Lens updated")
    m=st.text_input("Add line"); 
    if st.button("Save Line") and m: save_lens_line(profile_id,m); st.success("Saved")
    for l in get_lens_lines(profile_id)[-5:]: st.markdown(f"- {l}")

elif choice == "Settings":
    st.header("Profiles & AI")
    new_name=st.text_input("New Profile Name"); new_pin=st.text_input("PIN",type="password")
    if st.button("Create Profile") and new_name and new_pin: create_profile(new_name,new_pin); st.success("Created")
    profiles=get_profiles()
    if profiles:
        sel=st.selectbox("Select Profile",[p[1] for p in profiles]); prof=[p for p in profiles if p[1]==sel][0]
        prof_id,_,ai_toggle,api_key=prof
        key=st.text_input("API Key",value=api_key if api_key else "",type="password")
        toggle=st.checkbox("Enable AI",value=bool(ai_toggle))
        if st.button("Save AI Settings"):
            conn=sqlite3.connect("timesculpt.db"); cur=conn.cursor()
            cur.execute("UPDATE profiles SET api_key=?, ai_toggle=? WHERE id=?", (key,int(toggle),prof_id))
            conn.commit(); conn.close(); st.success("Saved")
        if key and toggle: st.markdown('<div class="card">✅ AI Connected</div>',unsafe_allow_html=True)
        elif key: st.markdown('<div class="card">⚠️ Key saved, AI disabled</div>',unsafe_allow_html=True)
        else: st.markdown('<div class="card">❌ No AI</div>',unsafe_allow_html=True)
