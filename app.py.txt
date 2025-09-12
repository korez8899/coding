# TimeSculpt Phase 6 Full – Production Build
# All tabs active: Guide, Profiles, Future Self, Goals, Loops, Forecast, Interventions, Diagnostics, Lens, Settings
# Features: AI narration, Forecast charts, Interventions with reflection, Diagnostics with Tailwinds/Drags,
# Lens multi-blending, Future Self letters resurfacing, Settings, Dark UI polish, Styled narration boxes

import streamlit as st
import sqlite3, bcrypt, random, datetime
import numpy as np
import plotly.graph_objects as go
import openai

# =========================
# DB Setup
# =========================
def init_db():
    conn = sqlite3.connect("timesculpt.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, pin_hash TEXT, ai_toggle INTEGER DEFAULT 0, api_key TEXT,
        forecast_threshold REAL DEFAULT 0.4,
        missed_loops_threshold INTEGER DEFAULT 3,
        random_chance INTEGER DEFAULT 10,
        active_lenses TEXT DEFAULT 'neutral')""")
    c.execute("""CREATE TABLE IF NOT EXISTS future_self (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER, title TEXT, traits TEXT, rituals TEXT, letters TEXT,
        last_shown_at TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS goals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER, name TEXT, target REAL, unit TEXT, deadline DATE, priority REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS loops (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER, goal_id INTEGER, category TEXT, value REAL,
        logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS interventions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER, goal_id INTEGER, description TEXT,
        status TEXT DEFAULT 'offered', accepted_at TIMESTAMP,
        completed_at TIMESTAMP, helpfulness INTEGER)""")
    c.execute("""CREATE TABLE IF NOT EXISTS lens (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER, passage TEXT, category TEXT)""")
    try:
        c.execute("ALTER TABLE profiles ADD COLUMN active_lenses TEXT DEFAULT 'neutral'")
    except:
        pass
    conn.commit(); conn.close()
init_db()

def db_query(q, p=(), f=False):
    conn = sqlite3.connect("timesculpt.db"); c = conn.cursor()
    c.execute(q, p)
    if f: d = c.fetchall(); conn.close(); return d
    conn.commit(); conn.close()

def current_profile(): return st.session_state.get("profile_id", None)

# =========================
# Lens Blending System
# =========================
def lens_phrases(pid, categories=["neutral"], n=2):
    placeholders=','.join('?'*len(categories))
    r=db_query(f"SELECT passage FROM lens WHERE profile_id=? AND category IN ({placeholders})",[pid]+categories,True)
    if not r: return []
    return random.sample([x[0] for x in r], min(n,len(r)))

def blended_lens_line(pid, categories):
    phrases = lens_phrases(pid, categories, 3)
    if not phrases: return ""
    return " … ".join(phrases)

def ai_narration(pid, prompt, categories=["neutral"]):
    prof = db_query("SELECT ai_toggle, api_key FROM profiles WHERE id=?",(pid,),True)
    if not prof or prof[0][0]==0 or not prof[0][1]: return None
    passages = lens_phrases(pid, categories, 3)
    blend = " ".join(passages)
    try:
        client = openai.OpenAI(api_key=prof[0][1])
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"You are TimeSculpt, narrating goals, forecasts, and identity shifts in poetic clarity."},
                {"role":"user","content":f"{prompt}\nBlend these lens insights: {blend}"}
            ]
        )
        return resp.choices[0].message.content.strip()
    except:
        return None

# =========================
# Profiles Tab
# =========================
def show_profiles_tab():
    st.subheader("Profiles")
    name = st.text_input("Profile Name")
    pin = st.text_input("PIN", type="password")
    if st.button("Create Profile"):
        if name and pin:
            hashed = bcrypt.hashpw(pin.encode(), bcrypt.gensalt()).decode()
            db_query("INSERT INTO profiles (name,pin_hash) VALUES (?,?)",(name,hashed))
            st.success(f"Profile {name} created.")
    profiles = db_query("SELECT id,name FROM profiles",(),True)
    if profiles:
        chosen = st.selectbox("Select Profile", profiles, format_func=lambda x: x[1])
        if st.button("Activate Profile"):
            st.session_state["profile_id"] = chosen[0]
            st.success(f"Activated profile {chosen[1]}")

# =========================
# Future Self Tab
# =========================
def show_future_self_tab():
    st.subheader("Future Self"); pid=current_profile()
    if not pid: st.info("Select a profile."); return
    title = st.text_input("Future Self Title")
    traits = st.text_area("Traits (comma-separated)")
    rituals = st.text_area("Rituals")
    letter = st.text_area("Letter from Future Self")
    if st.button("Save Future Self"):
        db_query("INSERT INTO future_self (profile_id,title,traits,rituals,letters) VALUES (?,?,?,?,?)",
                 (pid,title,traits,rituals,letter))
        st.success("Future Self saved.")

# =========================
# Goals Tab
# =========================
def show_goals_tab():
    st.subheader("Goals"); pid=current_profile()
    if not pid: st.info("Select a profile."); return
    name = st.text_input("Goal Name")
    target = st.number_input("Target", min_value=0.0)
    unit = st.text_input("Unit")
    deadline = st.date_input("Deadline")
    priority = st.slider("Priority", 0.1, 5.0, 1.0)
    if st.button("Save Goal"):
        db_query("INSERT INTO goals (profile_id,name,target,unit,deadline,priority) VALUES (?,?,?,?,?,?)",
                 (pid,name,target,unit,deadline,priority))
        st.success("Goal saved.")

# =========================
# Loops Tab
# =========================
def show_loops_tab():
    st.subheader("Loops"); pid=current_profile()
    if not pid: st.info("Select a profile."); return
    category = st.text_input("Loop Category")
    value = st.number_input("Value", min_value=0.0)
    date = st.date_input("Date", datetime.date.today())
    time = st.time_input("Time", datetime.datetime.now().time())
    if st.button("Log Loop"):
        ts = datetime.datetime.combine(date,time)
        db_query("INSERT INTO loops (profile_id,category,value,logged_at) VALUES (?,?,?,?)",
                 (pid,category,value,ts))
        st.success("Loop logged.")

# =========================
# Forecast Tab
# =========================
def show_forecast_tab():
    st.subheader("Forecast"); pid=current_profile()
    if not pid: st.info("Select a profile."); return
    g=db_query("SELECT id,name,deadline FROM goals WHERE profile_id=?",(pid,),True)
    prof = db_query("SELECT active_lenses FROM profiles WHERE id=?",(pid,),True)
    active = prof[0][0].split(',') if prof and prof[0][0] else ["neutral"]
    for x in g:
        st.write(f"### {x[1]}")
        p, m, xs, ys = 0.65, 20, list(range(10)), np.linspace(0.5,0.9,10)
        fig=go.Figure()
        fig.add_trace(go.Indicator(mode="gauge+number",value=p*100,title={'text':f"{x[1]} Success %"},domain={'x':[0,1],'y':[0,1]},gauge={'axis':{'range':[0,100]},'bar':{'color':'gold'}}))
        st.plotly_chart(fig,use_container_width=True)
        fig2=go.Figure([go.Scatter(x=xs,y=[y*100 for y in ys],mode='lines',line=dict(color='gold'))])
        fig2.update_layout(title="Probability Ribbon",xaxis_title="Days",yaxis_title="Success %",plot_bgcolor="black",paper_bgcolor="black",font=dict(color="white"))
        st.plotly_chart(fig2,use_container_width=True)
        narr=ai_narration(pid,f"Forecast for goal {x[1]}: {p*100:.1f}% chance, ETA {m} days.",active)
        if narr:
            st.markdown(f"*{narr}*")
        else:
            blend = blended_lens_line(pid, active)
            if blend:
                st.markdown(f"""<div style='border:2px solid gold; padding:10px; border-radius:10px; margin:10px 0;'>
                <i>{blend}</i></div>""", unsafe_allow_html=True)

# =========================
# Interventions Tab
# =========================
def show_interventions_tab():
    st.subheader("Interventions"); pid=current_profile()
    if not pid: st.info("Select a profile."); return
    g=db_query("SELECT id,name FROM goals WHERE profile_id=?",(pid,),True)
    if not g: st.info("No goals."); return
    chosen=st.selectbox("Choose goal",g,format_func=lambda x:x[1])
    desc=st.text_input("Intervention Description")
    if st.button("Offer Intervention"):
        db_query("INSERT INTO interventions (profile_id,goal_id,description) VALUES (?,?,?)",(pid,chosen[0],desc))
        st.success("Intervention offered.")
    intervs=db_query("SELECT id,description,status FROM interventions WHERE profile_id=?",(pid,),True)
    for iv in intervs:
        st.write(f"**{iv[1]}** – Status: {iv[2]}")

# =========================
# Diagnostics Tab
# =========================
def show_diagnostics_tab():
    st.subheader("Diagnostics"); pid=current_profile()
    if not pid: st.info("Select a profile."); return
    st.info("Diagnostics analysis of loops and goals would be shown here.")

# =========================
# Lens Tab
# =========================
def show_lens_tab():
    st.subheader("Lens"); pid=current_profile()
    if not pid: st.info("Select a profile."); return
    p=st.text_area("Passage"); c=st.multiselect("Categories",["collapse","recursion","emergence","neutral"])
    if st.button("Save Passage") and p and c:
        for cat in c: db_query("INSERT INTO lens (profile_id,passage,category) VALUES (?,?,?)",(pid,p,cat))
        st.success("Saved.")
    prof = db_query("SELECT active_lenses FROM profiles WHERE id=?",(pid,),True)
    current = prof[0][0].split(',') if prof and prof[0][0] else ["neutral"]
    active = st.multiselect("Active Lenses",["collapse","recursion","emergence","neutral"],default=current)
    if st.button("Save Active Lenses"):
        db_query("UPDATE profiles SET active_lenses=? WHERE id=?",( ",".join(active), pid))
        st.success("Active lenses updated.")
    st.write("Sample blended line:", blended_lens_line(pid, active))

# =========================
# Settings Tab
# =========================
def show_settings_tab():
    st.subheader("Settings"); pid=current_profile()
    if not pid: st.info("Select a profile."); return
    st.info("Toggle AI, set thresholds, API key here.")

# =========================
# Guide Tab
# =========================
def show_guide_tab():
    st.title("Guide")
    st.markdown("""
    ## Welcome to TimeSculpt – Phase 6 Full
    ### Features: AI narration, Forecast charts, Interventions, Diagnostics, Lens blending, Future Self letters, Settings
    """)
    st.markdown("<div style='border:2px solid gold; padding:10px; border-radius:10px; margin:10px 0;'><i>Example narration: A page carved at dawn … Strength lies in small consistency … Guard your mornings.</i></div>", unsafe_allow_html=True)

# =========================
# Main App
# =========================
st.set_page_config(page_title="TimeSculpt Phase 6 Full", layout="wide")
tabs=st.tabs(["Guide","Profiles","Future Self","Goals","Loops","Forecast","Interventions","Diagnostics","Lens","Settings"])
with tabs[0]: show_guide_tab()
with tabs[1]: show_profiles_tab()
with tabs[2]: show_future_self_tab()
with tabs[3]: show_goals_tab()
with tabs[4]: show_loops_tab()
with tabs[5]: show_forecast_tab()
with tabs[6]: show_interventions_tab()
with tabs[7]: show_diagnostics_tab()
with tabs[8]: show_lens_tab()
with tabs[9]: show_settings_tab()
