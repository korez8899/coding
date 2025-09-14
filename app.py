# =========================
# IMPORTS & CONFIG
# =========================
import streamlit as st
import sqlite3, bcrypt, random, os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from openai import OpenAI

# --- Page style ---
st.set_page_config(page_title="TimeSculpt", layout="wide")
st.markdown("""
<style>
body, .stApp {
    background-color: #fafafa;
    color: #222;
    font-size: 18px;
}
.stTabs [role="tab"] {
    color: #444 !important;
    font-weight: bold;
}
.stTabs [role="tab"][aria-selected="true"] {
    border-bottom: 3px solid #4b9cd3 !important;
}
textarea, input, select {
    background-color: #fff !important;
    color: #222 !important;
    border-radius: 6px !important;
    border: 1px solid #ccc !important;
}
div[data-testid="stMetricValue"] {
    color: #4b9cd3 !important;
    font-weight: bold;
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
        c.execute("CREATE TABLE IF NOT EXISTS profiles(id INTEGER PRIMARY KEY, name TEXT, pin_hash TEXT, api_key TEXT, ai_enabled INT, demo INT DEFAULT 0)")
        c.execute("CREATE TABLE IF NOT EXISTS goals(id INTEGER PRIMARY KEY, profile_id INT, name TEXT, target REAL, unit TEXT, deadline TEXT, priority INT)")
        c.execute("CREATE TABLE IF NOT EXISTS loops(id INTEGER PRIMARY KEY, profile_id INT, category TEXT, value REAL, date TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS future_self(id INTEGER PRIMARY KEY, profile_id INT, title TEXT, traits TEXT, loops TEXT, letter TEXT, vision TEXT)")
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
# AI / NARRATION
# =========================
def get_ai_client(pid):
    prof = fetch("SELECT api_key, ai_enabled FROM profiles WHERE id=?", (pid,))
    if prof and prof[0][0] and prof[0][1]:
        return OpenAI(api_key=prof[0][0])
    return None

def ai_narration(pid, prompt):
    client = get_ai_client(pid)
    if not client:
        # fallback → lens-based narration
        lens_lines = fetch("SELECT passage FROM lens WHERE profile_id=? ORDER BY RANDOM() LIMIT 1", (pid,))
        if lens_lines:
            return f"Lens reflection: {lens_lines[0][0]}"
        return None
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"(AI error: {e})"

# =========================
# DEMO DATA
# =========================
def seed_demo(pid):
    today = datetime.now()
    for i in range(45):
        d = today - timedelta(days=i)
        save("INSERT INTO loops(profile_id, category, value, date) VALUES(?,?,?,?)",
             (pid, random.choice(["write","exercise","scroll","meditate","plan","read","junk","skip"]),
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
    with st.expander("🔑 Profiles"):
        st.markdown("""
Profiles are **your separate identities**.  
Each one stores its own goals, loops, lens, and future self.  
Useful if you want to track multiple journeys.
""")
    with st.expander("🌠 Future Self"):
        st.markdown("""
Define **who you want to become**:  
- **Traits** → Who you are.  
- **Loops** → The daily actions your Future Self performs.  
- **Letter** → A direct note to yourself.  
- **Vision** → The bigger picture of what you’re walking toward.  
""")
    with st.expander("🎯 Goals"):
        st.markdown("""
Add measurable outcomes with targets, units, deadlines, and priority.  
Goals align your loops toward a finish line.
""")
    with st.expander("🔄 Loops"):
        st.markdown("""
Log your repeated actions.  
Every loop pushes or pulls your trajectory.  
These drive both Forecast and Diagnostics.
""")
    with st.expander("📈 Forecast"):
        st.markdown("""
Forecast projects where you’re heading.  
It merges goals + loops into charts, percentages, and narration.
""")
    with st.expander("🛠 Interventions"):
        st.markdown("""
Actions you choose to disrupt your path.  
Plan, complete, reflect, and track their helpfulness.
""")
    with st.expander("📚 Lens"):
        st.markdown("""
Your inner library. Add passages, quotes, or reflections.  
The system pulls from these to narrate and inspire.
""")
    with st.expander("⚖️ Diagnostics"):
        st.markdown("""
Shows the balance of **Forces (+)** and **Drags (–)**.  
Highlights what habits move you forward, and what slows you down.
""")
    with st.expander("⚙️ Settings"):
        st.markdown("""
Turn AI on/off, set your API key, enable demo data, or reset a profile.
""")

# =========================
# PROFILES
# =========================
def show_profiles():
    st.header("👤 Profiles")
    name = st.text_input("Profile Name", key="profile_name")
    pin = st.text_input("PIN", type="password", key="profile_pin")
    if st.button("Create Profile", key="create_profile_btn"):
        if name and pin:
            hashed = bcrypt.hashpw(pin.encode(), bcrypt.gensalt()).decode()
            save("INSERT INTO profiles(name,pin_hash) VALUES(?,?)", (name, hashed))
            st.success("Profile created.")
    profs = fetch("SELECT id,name FROM profiles")
    if profs:
        sel = st.selectbox("Select Profile", [p[1] for p in profs], key="profile_select")
        if st.button("Login", key="login_profile_btn"):
            row = fetch("SELECT id FROM profiles WHERE name=?", (sel,))
            if row:
                st.session_state.profile = row[0][0]
                st.success(f"Logged in as {sel}")

# =========================
# FUTURE SELF
# =========================
def show_future():
    st.header("🌠 Future Self")
    pid = current_profile()
    if not pid:
        st.info("Select a profile first.")
        return

    st.markdown("""
Define your **Future Self** in 4 layers:  
- **Traits** → who you are.  
- **Loops** → the key actions that build momentum.  
- **Letter** → direct guidance to yourself.  
- **Vision** → the horizon you move toward.  
""")

    title = st.text_input("Title (e.g. The Focused Writer)", key=f"fs_title_{pid}")
    traits = st.text_area("Traits (comma separated)", key=f"fs_traits_{pid}")
    loops = st.text_area("Loops (comma separated)", key=f"fs_loops_{pid}")
    letter = st.text_area("Letter to Yourself", key=f"fs_letter_{pid}")
    vision = st.text_area("Vision Statement", key=f"fs_vision_{pid}")

    if st.button("Save Future Self", key=f"fs_save_{pid}"):
        save("INSERT INTO future_self(profile_id,title,traits,loops,letter,vision) VALUES(?,?,?,?,?,?)",
             (pid, title, traits, loops, letter, vision))
        st.success("Future Self saved.")

    rows = fetch("SELECT title,traits,loops,letter,vision FROM future_self WHERE profile_id=?", (pid,))
    if rows:
        for t, tr, lp, lt, vs in rows:
            st.markdown(f"**{t}**")
            st.write(f"Traits: {tr}")
            st.write(f"Loops: {lp}")
            st.write(f"Letter: {lt}")
            st.write(f"Vision: {vs}")

# =========================
# GOALS
# =========================
def show_goals():
    st.header("🎯 Goals")
    pid = current_profile()
    if not pid:
        st.info("Select a profile first.")
        return
    g = st.text_input("Goal Name", key=f"goal_name_{pid}")
    t = st.number_input("Target", step=1.0, key=f"goal_target_{pid}")
    u = st.text_input("Unit", key=f"goal_unit_{pid}")
    d = st.date_input("Deadline", key=f"goal_deadline_{pid}")
    p = st.slider("Priority", 1, 5, 3, key=f"goal_priority_{pid}")
    if st.button("Save Goal", key=f"save_goal_{pid}"):
        save("INSERT INTO goals(profile_id,name,target,unit,deadline,priority) VALUES(?,?,?,?,?,?)",
             (pid, g, t, u, d.isoformat(), p))
        st.success("Goal saved.")
    rows = fetch("SELECT name,target,unit,deadline,priority FROM goals WHERE profile_id=?", (pid,))
    if rows:
        st.subheader("Your Goals")
        for n, t, u, d, p in rows:
            st.write(f"**{n}** — Target: {t} {u}, Deadline: {d}, Priority: {p}")

# =========================
# LOOPS
# =========================
def show_loops():
    st.header("🔄 Loops")
    pid = current_profile()
    if not pid:
        st.info("Select a profile to log loops.")
        return

    st.markdown("""
Loops are your **repeated actions** — the daily moves that sculpt momentum.  
They feed directly into **Forecast** and **Diagnostics**, so consistency here
is the backbone of the whole system.
""")

    with st.form(f"log_loop_form_{pid}", clear_on_submit=True):
        category = st.text_input("Category (e.g., Write, Exercise, Scroll)", key=f"loop_cat_{pid}")
        value = st.number_input("Value (Intensity / Duration / Count)", step=1.0, key=f"loop_val_{pid}")
        date = st.date_input("Date", datetime.now().date(), key=f"loop_date_{pid}")
        submitted = st.form_submit_button("Log Loop")
        if submitted and category:
            save("INSERT INTO loops(profile_id,category,value,date) VALUES(?,?,?,?)",
                 (pid, category.strip(), value, date.isoformat()))
            st.success(f"Loop '{category}' logged.")

    rows = fetch("SELECT category, value, date FROM loops WHERE profile_id=? ORDER BY date DESC LIMIT 20", (pid,))
    if rows:
        df = pd.DataFrame(rows, columns=["Category", "Value", "Date"])
        df["Date"] = pd.to_datetime(df["Date"])
        st.dataframe(df, use_container_width=True)

        fig = px.line(df.sort_values("Date"), x="Date", y="Value", color="Category",
                      markers=True, template="plotly_white")
        fig.update_layout(title="Loop Trends", xaxis_title="Date", yaxis_title="Value",
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No loops logged yet. Start by recording your first action.")
# =========================
# FORECAST
# =========================
def show_forecast():
    st.header("📈 Forecast")
    pid = current_profile()
    if not pid:
        st.info("Select a profile first.")
        return

    goals = fetch("SELECT id,name,target,unit,deadline FROM goals WHERE profile_id=?", (pid,))
    if not goals:
        st.info("No goals set yet. Add one under Goals.")
        return

    st.markdown("Your **forecast** shows how today’s loops bend tomorrow’s outcomes.")

    for gid, name, target, unit, deadline in goals:
        loops_done = fetch("SELECT SUM(value) FROM loops WHERE profile_id=?", (pid,))
        done = loops_done[0][0] or 0
        perc = (done / target) if target else 0

        # Gauge chart
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=perc*100,
            title={'text': f"{name} Progress (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#4b9cd3"},
                'steps': [
                    {'range': [0, 50], 'color': "#f2f2f2"},
                    {'range': [50, 100], 'color': "#d0e3f7"},
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 3},
                    'thickness': 0.75,
                    'value': 100
                }
            }
        ))
        st.plotly_chart(fig_g, use_container_width=True)

        st.metric(f"{name} Progress", f"{perc*100:.1f}% toward {target} {unit}")

        # AI / Lens narration
        narr = ai_narration(pid, f"Goal: {name}, Progress: {perc*100:.1f}%. Deadline: {deadline}.")
        if narr:
            st.markdown(
                f"""<div style='border-left:4px solid #4b9cd3; padding:10px; margin:10px 0; background:#f9f9f9;'>
                <i>{narr}</i></div>""",
                unsafe_allow_html=True
            )

# =========================
# INTERVENTIONS
# =========================
def show_interventions():
    st.header("🛠️ Interventions")
    pid = current_profile()
    if not pid:
        st.info("Select a profile.")
        return

    st.markdown("""
Interventions are deliberate **course corrections**.  
They let you disrupt inertia and shift direction consciously.
""")

    desc = st.text_input("New Intervention", key=f"intervention_desc_{pid}")
    if st.button("Add Intervention", key=f"intervention_add_{pid}"):
        save("INSERT INTO interventions(profile_id,description,status) VALUES(?,?,?)",
             (pid, desc, "pending"))
        st.success("Intervention added.")

    rows = fetch("SELECT id,description,status,completed_date,helpful,reflection FROM interventions WHERE profile_id=?", (pid,))
    for iid, desc, status, cd, helpful, ref in rows:
        st.write(f"**{desc}** — {status}")
        if status != "completed":
            if st.button("Complete", key=f"intervention_complete_{iid}"):
                save("UPDATE interventions SET status=?, completed_date=? WHERE id=?",
                     ("completed", datetime.now().isoformat(), iid))
                st.rerun()
        if status == "completed":
            h = st.selectbox("Helpful?", ["Yes", "No"], key=f"intervention_helpful_{iid}", index=0 if not helpful else ["Yes","No"].index(helpful))
            r = st.text_area("Reflection", value=ref or "", key=f"intervention_reflection_{iid}")
            if st.button("Save Feedback", key=f"intervention_feedback_{iid}"):
                save("UPDATE interventions SET helpful=?,reflection=? WHERE id=?", (h, r, iid))
                st.success("Feedback saved.")

# =========================
# LENS
# =========================
def show_lens():
    st.header("📚 Lens")
    pid = current_profile()
    if not pid:
        st.info("Select a profile.")
        return

    st.markdown("""
The **Lens** is your living library.  
Passages you add here are woven into narration and reflections across the app.
""")

    passage = st.text_area("Passage", key=f"lens_passage_{pid}")
    cat = st.selectbox("Category", ["recursion", "emergence", "neutral"], key=f"lens_category_{pid}")
    if st.button("Add Passage", key=f"lens_add_{pid}"):
        save("INSERT INTO lens(profile_id,passage,category) VALUES(?,?,?)", (pid, passage, cat))
        st.success("Passage added.")

    # Lens search
    search = st.text_input("Search Lens", key=f"lens_search_{pid}")
    query = f"%{search}%" if search else "%"
    rows = fetch("SELECT passage,category FROM lens WHERE profile_id=? AND passage LIKE ?", (pid, query))
    if rows:
        for p, c in rows:
            st.markdown(f"<div style='margin:6px 0;'><b>{c}</b>: {p}</div>", unsafe_allow_html=True)
    else:
        st.info("No passages found.")

# =========================
# DIAGNOSTICS
# =========================
def show_diag():
    st.header("⚖️ Diagnostics")
    pid = current_profile()
    if not pid:
        st.info("Select a profile.")
        return

    st.markdown("Your **forces vs drags** balance over the last month.")

    loops = fetch("SELECT category,value,date FROM loops WHERE profile_id=?", (pid,))
    if not loops:
        st.info("No loops recorded.")
        return

    forces, drags, neutral = {}, {}, {}
    force_keywords = ["write","exercise","save","sleep","study","meditate","walk","water","plan","read"]
    drag_keywords  = ["scroll","late","junk","skip","procrastinate","smoke","drink"]

    for c,v,d in loops:
        cl = c.lower()
        if any(w in cl for w in force_keywords): forces[c] = forces.get(c,0)+v
        elif any(w in cl for w in drag_keywords): drags[c] = drags.get(c,0)+v
        else: neutral[c] = neutral.get(c,0)+v

    if forces:
        st.subheader("Forces (+)")
        st.plotly_chart(px.bar(x=list(forces.keys()), y=list(forces.values()), labels={"x":"Loop","y":"Value"}, template="plotly_white"), use_container_width=True)

    if drags:
        st.subheader("Drags (-)")
        st.plotly_chart(px.bar(x=list(drags.keys()), y=list(drags.values()), labels={"x":"Loop","y":"Value"}, template="plotly_white"), use_container_width=True)

    total_forces, total_drags = sum(forces.values()), sum(drags.values())
    if total_forces + total_drags > 0:
        ratio = total_forces / (total_forces + total_drags)
        st.metric("Forces/Drags Balance", f"{ratio:.2f}")

        narr = ai_narration(pid, f"Diagnostics — Forces={total_forces}, Drags={total_drags}, Ratio={ratio:.2f}")
        if narr:
            st.markdown(f"<div style='border-left:4px solid #4b9cd3; padding:10px; background:#f9f9f9;'><i>{narr}</i></div>", unsafe_allow_html=True)

    # Trends
    df = pd.DataFrame(loops, columns=["Category","Value","Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    df["Type"] = df["Category"].apply(lambda x: "Force" if any(w in x.lower() for w in force_keywords) else "Drag" if any(w in x.lower() for w in drag_keywords) else "Neutral")
    df = df.groupby(["Date","Type"])["Value"].sum().reset_index()

    fig = px.line(df, x="Date", y="Value", color="Type", template="plotly_white", markers=True)
    fig.update_layout(title="Forces vs Drags Over Time", xaxis_title="Date", yaxis_title="Total Value")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# GUIDE POLISH
# =========================
def show_guide():
    st.header("📖 TimeSculpt Guide")
    st.markdown("""
TimeSculpt is a **self-directed evolution system**.  
Every action logged is not just data, but a **stroke of identity**.  

Use this guide to learn **why each tab matters**, and how to use it effectively.
""")
    with st.expander("👤 Profiles"):
        st.write("Keep identities separate — useful if you want distinct journeys (e.g. work vs creative self).")
    with st.expander("🌠 Future Self"):
        st.write("Define who you are becoming — traits, loops, letter, and vision. This shapes how Forecast and Lens narrate your journey.")
    with st.expander("🎯 Goals"):
        st.write("Create measurable targets with deadlines. Goals give your loops a horizon to aim toward.")
    with st.expander("🔄 Loops"):
        st.write("Log actions consistently. Loops are the fuel that power Forecast and Diagnostics.")
    with st.expander("📈 Forecast"):
        st.write("See your likely trajectory toward goals. AI or Lens narration interprets your momentum.")
    with st.expander("🛠 Interventions"):
        st.write("Plan deliberate corrections. Reflect on whether they worked.")
    with st.expander("📚 Lens"):
        st.write("A living library of passages that blend into narration. Search for inspiration or let the system auto-fuse.")
    with st.expander("⚖️ Diagnostics"):
        st.write("Balance of Forces vs Drags shows where your weight lies. Over time, trends reveal your true habits.")
    with st.expander("⚙️ Settings"):
        st.write("Configure AI, demo mode, and reset profiles.")
# =========================
# SETTINGS
# =========================
def show_settings():
    st.header("⚙️ Settings")
    pid = current_profile()
    if not pid:
        st.info("Select a profile.")
        return

    prof = fetch("SELECT ai_enabled, api_key, demo FROM profiles WHERE id=?", (pid,))
    ai_on, api, demo = prof[0] if prof else (0, "", 0)

    st.markdown("Customize your environment and connections.")

    # Unique keys per profile to avoid duplicate widget errors
    ai_on_new = st.toggle("Enable AI", bool(ai_on), key=f"enable_ai_toggle_{pid}")
    api_new = st.text_input("OpenAI API Key", value=api, type="password", key=f"api_key_{pid}")
    model_new = st.selectbox("AI Model", ["gpt-4o-mini","gpt-4o","gpt-3.5-turbo"], key=f"model_select_{pid}")
    demo_new = st.toggle("Enable Demo Data (30 days seeded)", bool(demo), key=f"demo_toggle_{pid}")

    if st.button("Save Settings", key=f"settings_save_{pid}"):
        save("UPDATE profiles SET ai_enabled=?, api_key=?, demo=? WHERE id=?", (1 if ai_on_new else 0, api_new, 1 if demo_new else 0, pid))
        if demo_new:
            seed_demo(pid)
        st.success("Settings updated.")

    if st.button("Reset Profile Data", key=f"reset_profile_{pid}"):
        save("DELETE FROM goals WHERE profile_id=?", (pid,))
        save("DELETE FROM loops WHERE profile_id=?", (pid,))
        save("DELETE FROM future_self WHERE profile_id=?", (pid,))
        save("DELETE FROM interventions WHERE profile_id=?", (pid,))
        save("DELETE FROM lens WHERE profile_id=?", (pid,))
        st.warning("Profile reset completed.")

# =========================
# DEMO MODE EXPANSION
# =========================
def seed_demo(pid):
    today = datetime.now()
    categories = ["write","exercise","scroll","meditate","plan","read","junk","smoke"]
    for i in range(30):
        d = today - timedelta(days=i)
        for _ in range(random.randint(1, 3)):
            cat = random.choice(categories)
            val = random.randint(1, 3)
            save("INSERT INTO loops(profile_id, category, value, date) VALUES(?,?,?,?)",
                 (pid, cat, val, d.date().isoformat()))

    save("INSERT INTO goals(profile_id,name,target,unit,deadline,priority) VALUES(?,?,?,?,?,?)",
         (pid,"Finish Book",50,"pages",(today+timedelta(days=30)).isoformat(),5))
    save("INSERT INTO goals(profile_id,name,target,unit,deadline,priority) VALUES(?,?,?,?,?,?)",
         (pid,"Run 100km",100,"km",(today+timedelta(days=45)).isoformat(),4))

    save("INSERT INTO interventions(profile_id,description,status) VALUES(?,?,?)",
         (pid,"Write for 20 minutes daily","pending"))
    save("INSERT INTO interventions(profile_id,description,status) VALUES(?,?,?)",
         (pid,"Morning meditation","pending"))

    save("INSERT INTO lens(profile_id,passage,category) VALUES(?,?,?)",
         (pid,"Each dawn bends toward clarity.","recursion"))
    save("INSERT INTO lens(profile_id,passage,category) VALUES(?,?,?)",
         (pid,"Your effort compounds beyond the visible.","emergence"))

# =========================
# POLISH (Styling)
# =========================
st.markdown("""
<style>
body, .stApp {
    background-color: #fafafa;
    color: #111;
    font-size: 18px;
}
.stTabs [role="tab"] {
    color: #333 !important;
    font-weight: bold;
}
.stTabs [role="tab"][aria-selected="true"] {
    border-bottom: 3px solid #4b9cd3 !important;
}
textarea, input, select {
    background-color: #fff !important;
    color: #000 !important;
    border-radius: 8px !important;
    border: 1px solid #ccc !important;
}
div[data-testid="stMetricValue"] {
    color: #4b9cd3 !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================
# MAIN ORCHESTRATION
# =========================
tabs = st.tabs([
    "📖 Guide",
    "👤 Profiles",
    "🌠 Future Self",
    "🎯 Goals",
    "🔄 Loops",
    "📈 Forecast",
    "🛠️ Interventions",
    "📚 Lens",
    "⚖️ Diagnostics",
    "⚙️ Settings"
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
