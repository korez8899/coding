# TimeSculpt – Final Production Build
# Features: Profiles, Future Self, Goals, Loops, Forecast, Interventions, Diagnostics, Lens, Settings, Guide
# Theme: Cosmic Night (default) + Aurora Light toggle
# Tab icons + larger inviting text

import streamlit as st
import sqlite3
from datetime import datetime, date, time
import hashlib
import pandas as pd
import random
import re

DB_FILE = "timesculpt.db"

# ---------- DB SCHEMA AUTO-PATCH ----------
def ensure_schema():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Profiles
    c.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            pin_hash TEXT,
            ai_toggle INTEGER DEFAULT 0,
            api_key TEXT
        )
    """)

    # Future Self
    c.execute("""
        CREATE TABLE IF NOT EXISTS future_self (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id INTEGER,
            trait TEXT,
            letter TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Goals
    c.execute("""
        CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id INTEGER,
            name TEXT,
            unit TEXT,
            target REAL,
            deadline TEXT,
            priority INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("PRAGMA table_info(goals)")
    cols = [row[1] for row in c.fetchall()]
    if "created_at" not in cols:
        c.execute("ALTER TABLE goals ADD COLUMN created_at TEXT DEFAULT CURRENT_TIMESTAMP")

    # Loops
    c.execute("""
        CREATE TABLE IF NOT EXISTS loops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id INTEGER,
            name TEXT,
            value REAL,
            unit TEXT,
            timestamp TEXT
        )
    """)

    # Lens
    c.execute("""
        CREATE TABLE IF NOT EXISTS lens_lines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id INTEGER,
            collection TEXT,
            category TEXT,
            passage TEXT,
            used INTEGER DEFAULT 0
        )
    """)
    c.execute("PRAGMA table_info(lens_lines)")
    cols = [row[1] for row in c.fetchall()]
    if "collection" not in cols:
        c.execute("ALTER TABLE lens_lines ADD COLUMN collection TEXT DEFAULT 'default'")

    conn.commit()
    conn.close()

ensure_schema()

# ---------- HELPERS ----------
def sha256(s): return hashlib.sha256(s.encode()).hexdigest()
def get_conn(): return sqlite3.connect(DB_FILE)
def now_iso(): return datetime.now().isoformat(timespec="seconds")

def get_lens_lines(pid, n=1):
    conn = get_conn(); cur = conn.cursor()
    rows = cur.execute("SELECT id,passage FROM lens_lines WHERE profile_id=? AND used=0 LIMIT ?",(pid,n)).fetchall()
    if not rows:  # reset if all used
        cur.execute("UPDATE lens_lines SET used=0 WHERE profile_id=?",(pid,))
        conn.commit()
        rows = cur.execute("SELECT id,passage FROM lens_lines WHERE profile_id=? LIMIT ?",(pid,n)).fetchall()
    lines = []
    for rid,p in rows:
        lines.append(p)
        cur.execute("UPDATE lens_lines SET used=1 WHERE id=?",(rid,))
    conn.commit(); conn.close()
    return lines

# ---------- THEME ----------
st.set_page_config(page_title="TimeSculpt", page_icon="⏳", layout="wide")

if "ts_theme" not in st.session_state:
    st.session_state.ts_theme = "Cosmic Night"

COSMIC_NIGHT = """
:root{
  --bg0:#0b1020; --bg1:#0e1630; --bg2:#12204a; --card:#0f1a33;
  --ink:#eaeaf2; --ink-dim:#cfd3de; --muted:#9aa7bd; --accent:#ffd54a;
  --accent-2:#ffb300; --ring:#394b70; --input:#141e36; --border:#2a3860;
}
html,body,[data-testid="stAppViewContainer"]{
  background: radial-gradient(1200px 1200px at 20% 10%, var(--bg2) 0%, var(--bg1) 35%, var(--bg0) 100%) !important;
  color: var(--ink); font-size:18px;
}
h1,h2,h3{ color: var(--accent) !important; font-weight:800 !important; }
.stTextInput input, .stNumberInput input, .stDateInput input, .stTextArea textarea{
  color: var(--ink) !important; background: var(--input) !important;
}
"""

AURORA_LIGHT = """
:root{
  --bg:#f6f7fb; --ink:#0f172a; --muted:#64748b; --accent:#f59e0b;
  --card:#ffffff; --border:#e5e7eb; --input:#ffffff;
}
html,body,[data-testid="stAppViewContainer"]{
  background: var(--bg) !important; color: var(--ink); font-size:18px;
}
h1,h2,h3{ color: var(--accent) !important; font-weight:800 !important; }
.stTextInput input, .stNumberInput input, .stDateInput input, .stTextArea textarea{
  color: var(--ink) !important; background: var(--input) !important;
}
"""

def apply_theme(theme):
    css = COSMIC_NIGHT if theme=="Cosmic Night" else AURORA_LIGHT
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

apply_theme(st.session_state.ts_theme)

# ---------- APP STATE ----------
if "profile_id" not in st.session_state: st.session_state.profile_id=None
if "profile_name" not in st.session_state: st.session_state.profile_name=None

# ---------- NAV ----------
tabs=["📘 Guide","👤 Profiles","🌠 Future Self","🎯 Goals","🔄 Input (Loops)",
      "📈 Forecast","🛠️ Interventions","⚖️ Diagnostics","📚 Lens","⚙️ Settings"]
choice=st.sidebar.radio("Navigate",tabs)

# ---------- GUIDE ----------
if choice=="📘 Guide":
    st.title("⏳ TimeSculpt – Instructional Guide")
    st.markdown("""
    TimeSculpt is not a habit tracker. It is a **recursive identity system**. 
    Each loop you log, each goal you set, and each choice you take bends probability toward the self you choose.
    """)
    st.subheader("Roadmap")
    st.markdown("""
    - 👤 Profiles → Create and log into your identity  
    - 🌠 Future Self → Define traits, rituals, and letters  
    - 🎯 Goals → Attach measurable targets  
    - 🔄 Input (Loops) → Log daily actions  
    - 📈 Forecast → See probabilities, ETAs, and narrative  
    - 🛠️ Interventions → Get nudges for next steps  
    - ⚖️ Diagnostics → Forces and drags shaping progress  
    - 📚 Lens → Upload texts to enrich narration  
    - ⚙️ Settings → Control AI toggle, API key, and Theme  
    """)
    st.subheader("Step by Step")
    st.markdown("""
    1. Create a profile (name + PIN).  
    2. Define traits + letters in 🌠 Future Self.  
    3. Add measurable 🎯 Goals.  
    4. Log 🔄 Loops daily with value + timestamp.  
    5. 📈 Forecast shows probabilities, ETA, narration.  
    6. 🛠️ Interventions suggest smallest next moves.  
    7. ⚖️ Diagnostics show Forces vs Drags.  
    8. 📚 Lens enriches narration with your texts.  
    9. ⚙️ Settings lets you manage AI + Theme.  
    """)
    st.subheader("Theme Toggle")
    st.markdown("Default is 🌌 Cosmic Night. Switch to 🌅 Aurora Light in ⚙️ Settings.")

# ---------- PROFILES ----------
elif choice=="👤 Profiles":
    st.header("Profiles")
    conn=get_conn();cur=conn.cursor()
    with st.form("new_profile"):
        name=st.text_input("Profile Name")
        pin=st.text_input("PIN",type="password")
        if st.form_submit_button("Create"):
            if name and pin:
                cur.execute("INSERT INTO profiles (name,pin_hash) VALUES (?,?)",(name,sha256(pin)))
                conn.commit(); st.success("Profile created.")
    profiles=cur.execute("SELECT id,name FROM profiles").fetchall(); conn.close()
    if profiles:
        names=[p[1] for p in profiles]
        pick=st.selectbox("Select Profile",names)
        pin=st.text_input("PIN to Login",type="password")
        if st.button("Login"):
            conn=get_conn();cur=conn.cursor()
            cur.execute("SELECT id,pin_hash FROM profiles WHERE name=?",(pick,))
            row=cur.fetchone(); conn.close()
            if row and row[1]==sha256(pin):
                st.session_state.profile_id=row[0]; st.session_state.profile_name=pick
                st.success(f"Logged in as {pick}")

# ---------- FUTURE SELF ----------
elif choice=="🌠 Future Self":
    st.header("Future Self")
    pid=st.session_state.profile_id
    if not pid: st.info("Log in first.")
    else:
        conn=get_conn();cur=conn.cursor()
        trait=st.text_input("Add Trait")
        if st.button("Save Trait"): cur.execute("INSERT INTO future_self (profile_id,trait) VALUES (?,?)",(pid,trait)); conn.commit()
        letter=st.text_area("Future Self Letter")
        if st.button("Save Letter"): cur.execute("INSERT INTO future_self (profile_id,letter) VALUES (?,?)",(pid,letter)); conn.commit()
        rows=cur.execute("SELECT trait,letter,created_at FROM future_self WHERE profile_id=?",(pid,)).fetchall()
        for t,l,c in rows[-5:]: st.markdown(f"- **{t or ''}** {l or ''} ({c})")
        conn.close()

# ---------- GOALS ----------
elif choice=="🎯 Goals":
    st.header("Goals")
    pid=st.session_state.profile_id
    if not pid: st.info("Log in first.")
    else:
        with st.form("add_goal"):
            gname=st.text_input("Goal Name")
            unit=st.text_input("Unit")
            target=st.number_input("Target",min_value=0.0)
            deadline=st.date_input("Deadline")
            priority=st.slider("Priority",1,5,3)
            if st.form_submit_button("Save Goal"):
                conn=get_conn();cur=conn.cursor()
                cur.execute("INSERT INTO goals (profile_id,name,unit,target,deadline,priority) VALUES (?,?,?,?,?,?)",
                            (pid,gname,unit,target,str(deadline),priority))
                conn.commit(); conn.close()
        conn=get_conn();cur=conn.cursor()
        rows=cur.execute("SELECT name,unit,target,deadline,priority FROM goals WHERE profile_id=?",(pid,)).fetchall()
        conn.close()
        for n,u,t,d,p in rows: st.markdown(f"- **{n}** ({t} {u}) by {d} (priority {p})")

# ---------- LOOPS ----------
elif choice=="🔄 Input (Loops)":
    st.header("Log Loops")
    pid=st.session_state.profile_id
    if not pid: st.info("Log in first.")
    else:
        with st.form("log_loop"):
            lname=st.text_input("Loop Name")
            val=st.number_input("Value",min_value=0.0)
            unit=st.text_input("Unit")
            ts_date=st.date_input("Date",value=date.today())
            ts_time=st.time_input("Time",value=datetime.now().time())
            ts=datetime.combine(ts_date,ts_time).isoformat()
            if st.form_submit_button("Log"):
                conn=get_conn();cur=conn.cursor()
                cur.execute("INSERT INTO loops (profile_id,name,value,unit,timestamp) VALUES (?,?,?,?,?)",
                            (pid,lname,val,unit,ts))
                conn.commit(); conn.close(); st.success("Loop logged.")

# ---------- FORECAST ----------
elif choice=="📈 Forecast":
    st.header("Forecast")
    pid=st.session_state.profile_id
    if not pid: st.info("Log in first.")
    else:
        conn=get_conn();cur=conn.cursor()
        goals=cur.execute("SELECT id,name,unit,target,deadline FROM goals WHERE profile_id=?",(pid,)).fetchall()
        loops=pd.read_sql_query("SELECT * FROM loops WHERE profile_id=?",(conn,),params=(pid,))
        conn.close()
        if goals:
            for gid,gname,gunit,gtarget,gdeadline in goals:
                df=loops[loops["unit"]==gunit]
                prog=df["value"].sum() if not df.empty else 0
                pace=df.groupby("timestamp")["value"].sum().mean() if not df.empty else 0
                prob=min(prog/gtarget,1) if gtarget>0 else 0
                eta=(gtarget-prog)/pace if pace>0 else float("inf")
                lens=get_lens_lines(pid,1)
                lens_text=f" — {lens[0]}" if lens else ""
                st.subheader(gname)
                st.metric("Success Chance",f"{prob*100:.1f}%")
                st.metric("ETA (days)","∞" if eta==float("inf") else f"{eta:.1f}")
                st.caption(f"Forecast narrative{lens_text}")
            if not loops.empty:
                loops["timestamp"]=pd.to_datetime(loops["timestamp"]); loops["day"]=loops["timestamp"].dt.date
                fig=px.line(loops.groupby("day")["value"].sum().reset_index(),x="day",y="value",title="Daily Loop Totals")
                st.plotly_chart(fig,use_container_width=True)

# ---------- INTERVENTIONS ----------
elif choice=="🛠️ Interventions":
    st.header("Interventions")
    pid=st.session_state.profile_id
    if not pid: st.info("Log in first.")
    else:
        conn=get_conn();cur=conn.cursor()
        goals=cur.execute("SELECT name,unit,target FROM goals WHERE profile_id=?",(pid,)).fetchall()
        loops=pd.read_sql_query("SELECT * FROM loops WHERE profile_id=?",(conn,),params=(pid,))
        conn.close()
        if not goals: st.info("No goals yet.")
        else:
            st.subheader("Top Interventions")
            for gname,gunit,gtarget in goals:
                df=loops[loops["unit"]==gunit]
                prog=df["value"].sum() if not df.empty else 0
                gap=gtarget-prog
                if gap>0:
                    move=min(gap,random.randint(1,5))
                    lens=get_lens_lines(pid,1)
                    st.write(f"- {move} {gunit} toward **{gname}** {('→ '+lens[0]) if lens else ''}")

# ---------- DIAGNOSTICS ----------
elif choice=="⚖️ Diagnostics":
    st.header("Diagnostics")
    pid=st.session_state.profile_id
    if not pid: st.info("Log in first.")
    else:
        loops=pd.read_sql_query("SELECT * FROM loops WHERE profile_id=?",(get_conn(),),params=(pid,))
        if loops.empty: st.info("No loops yet.")
        else:
            loops["timestamp"]=pd.to_datetime(loops["timestamp"])
            by_loop=loops.groupby("name")["value"].sum().reset_index()
            by_loop["type"]=by_loop["name"].apply(lambda x:"Drag" if re.search(r"scroll|late|junk",x.lower()) else "Force")
            forces=by_loop[by_loop["type"]=="Force"]; drags=by_loop[by_loop["type"]=="Drag"]
            if not forces.empty:
                st.subheader("Forces")
                fig=px.bar(forces,x="name",y="value",color="value",title="Positive Drivers")
                st.plotly_chart(fig,use_container_width=True)
            if not drags.empty:
                st.subheader("Drags")
                fig=px.bar(drags,x="name",y="value",color="value",title="Negative Drags")
                st.plotly_chart(fig,use_container_width=True)

# ---------- LENS ----------
elif choice=="📚 Lens":
    st.header("Lens")
    pid=st.session_state.profile_id
    if not pid: st.info("Log in first.")
    else:
        with st.form("add_lens"):
            col=st.text_input("Collection")
            cat=st.selectbox("Category",["Collapse","Recursion","Emergence","Neutral"])
            txt=st.text_area("Passages (one per line)")
            if st.form_submit_button("Save"):
                conn=get_conn();cur=conn.cursor()
                for line in txt.splitlines():
                    if line.strip():
                        cur.execute("INSERT INTO lens_lines (profile_id,collection,category,passage) VALUES (?,?,?,?)",(pid,col,cat,line.strip()))
                conn.commit(); conn.close(); st.success("Lens saved.")

# ---------- SETTINGS ----------
elif choice=="⚙️ Settings":
    st.header("Settings")
    pid=st.session_state.profile_id
    if not pid: st.info("Log in first.")
    else:
        conn=get_conn();cur=conn.cursor()
        ai=st.checkbox("Enable AI")
        key=st.text_input("API Key",type="password")
        theme=st.radio("Theme",["Cosmic Night","Aurora Light"], index=0 if st.session_state.ts_theme=="Cosmic Night" else 1, horizontal=True)
        if st.button("Save"):
            cur.execute("UPDATE profiles SET ai_toggle=?, api_key=? WHERE id=?",(int(ai),key,pid))
            conn.commit(); st.success("Settings saved.")
            if theme != st.session_state.ts_theme:
                st.session_state.ts_theme = theme
                apply_theme(theme)
        conn.close()
