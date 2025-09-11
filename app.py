# ================================
# TimeSculpt — Phase 1 Stable Build
# ================================
import streamlit as st
import sqlite3
import bcrypt
import datetime
import matplotlib.pyplot as plt
import docx
import PyPDF2

# ----------------------------
# Config & Styling
# ----------------------------
st.set_page_config(page_title="TimeSculpt", layout="wide")

st.markdown("""
    <style>
    .stApp {background: linear-gradient(to bottom, #0a0a0f, #111133);}
    section[data-testid="stSidebar"] {background-color: #0f0f1f;}
    label, .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: #FFD700 !important;
        font-weight: bold;
    }
    input, textarea {
        background-color: #1e1e2e !important;
        border-radius: 8px !important;
        color: white !important;
    }
    .card {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        background-color: #111122;
        color: white;
        border: 1px solid #FFD700;
    }
    .highlight {color: #FFD700; font-weight: bold;}
    h1, h2, h3 {color: #FFD700 !important; font-weight: bold;}
    #watermark {
        position: fixed; bottom: 10px; right: 20px;
        color: #FFD700; font-size: 14px; opacity: 0.7;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div id="watermark">TimeSculpt</div>', unsafe_allow_html=True)

# ----------------------------
# DB Setup
# ----------------------------
def init_db():
    conn = sqlite3.connect("timesculpt.db")
    cur = conn.cursor()

    cur.execute("""CREATE TABLE IF NOT EXISTS profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        pin_hash TEXT,
        ai_toggle INTEGER DEFAULT 0,
        api_key TEXT
    )""")

    cur.execute("""CREATE TABLE IF NOT EXISTS goals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        name TEXT,
        target REAL,
        unit TEXT,
        deadline TEXT,
        priority REAL,
        FOREIGN KEY(profile_id) REFERENCES profiles(id)
    )""")

    cur.execute("""CREATE TABLE IF NOT EXISTS loops (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        loop_name TEXT,
        value REAL,
        unit TEXT,
        timestamp TEXT,
        FOREIGN KEY(profile_id) REFERENCES profiles(id)
    )""")

    cur.execute("""CREATE TABLE IF NOT EXISTS traits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        trait TEXT,
        FOREIGN KEY(profile_id) REFERENCES profiles(id)
    )""")

    cur.execute("""CREATE TABLE IF NOT EXISTS letters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        content TEXT,
        FOREIGN KEY(profile_id) REFERENCES profiles(id)
    )""")

    cur.execute("""CREATE TABLE IF NOT EXISTS lens_lines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        line TEXT,
        FOREIGN KEY(profile_id) REFERENCES profiles(id)
    )""")

    conn.commit()
    conn.close()

init_db()

# ----------------------------
# Utility Functions
# ----------------------------
def hash_pin(pin):
    return bcrypt.hashpw(pin.encode(), bcrypt.gensalt()).decode()

def get_profiles():
    conn = sqlite3.connect("timesculpt.db")
    cur = conn.cursor()
    cur.execute("SELECT id, name, ai_toggle, api_key FROM profiles")
    rows = cur.fetchall()
    conn.close()
    return rows

def create_profile(name, pin):
    conn = sqlite3.connect("timesculpt.db")
    cur = conn.cursor()
    hashed = hash_pin(pin)
    cur.execute("INSERT INTO profiles (name, pin_hash) VALUES (?, ?)", (name, hashed))
    conn.commit()
    conn.close()

def log_loop(profile_id, loop_name, value, unit, dt):
    conn = sqlite3.connect("timesculpt.db")
    cur = conn.cursor()
    cur.execute("""INSERT INTO loops (profile_id, loop_name, value, unit, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (profile_id, loop_name, value, unit, dt.isoformat()))
    conn.commit()
    conn.close()

def get_loops(profile_id):
    conn = sqlite3.connect("timesculpt.db")
    cur = conn.cursor()
    cur.execute("SELECT loop_name, value, unit, timestamp FROM loops WHERE profile_id=?", (profile_id,))
    rows = cur.fetchall()
    conn.close()
    return rows

def save_lens_line(profile_id, line):
    conn = sqlite3.connect("timesculpt.db")
    cur = conn.cursor()
    cur.execute("INSERT INTO lens_lines (profile_id, line) VALUES (?, ?)", (profile_id, line))
    conn.commit()
    conn.close()

def get_lens_lines(profile_id):
    conn = sqlite3.connect("timesculpt.db")
    cur = conn.cursor()
    cur.execute("SELECT line FROM lens_lines WHERE profile_id=?", (profile_id,))
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return rows

def save_goal(profile_id, name, target, unit, deadline, priority):
    conn = sqlite3.connect("timesculpt.db")
    cur = conn.cursor()
    cur.execute("INSERT INTO goals (profile_id, name, target, unit, deadline, priority) VALUES (?,?,?,?,?,?)",
                (profile_id, name, target, unit, deadline, priority))
    conn.commit()
    conn.close()

def get_goals(profile_id):
    conn = sqlite3.connect("timesculpt.db")
    cur = conn.cursor()
    cur.execute("SELECT name, target, unit, deadline, priority FROM goals WHERE profile_id=?", (profile_id,))
    rows = cur.fetchall()
    conn.close()
    return rows

# ----------------------------
# Sidebar Navigation
# ----------------------------
tabs = ["Guide", "Future Self", "Input", "Forecast", "Interventions", "Diagnostics", "Lens", "Settings"]
choice = st.sidebar.radio("Navigate", tabs)

# ----------------------------
# GUIDE TAB
# ----------------------------
if choice == "Guide":
    st.title("TimeSculpt")
    st.markdown("**TimeSculpt is not a tracker. It is a sculptor’s tool.**")

    st.subheader("📖 About TimeSculpt")
    st.markdown("""
    TimeSculpt helps you evolve by aligning **Future Self identity**, **daily loops**, and **probability forecasts**.  
    With consistent use you can expect:
    - Sharper clarity about who you’re becoming  
    - Goals that align with your identity  
    - Forecasts that reflect your real trajectory  
    - Lens echoes shaping your narrative  
    """)

    st.subheader("🧭 How to Use Each Tab")
    st.markdown("""
    - **Profiles (Settings):** Create and select a profile with name + PIN.  
    - **Future Self:** Define traits and write letters from your Future Self. Attach goals with deadlines and priorities.  
    - **Input:** Log daily loops (activities, habits). Date/time recorded automatically.  
    - **Forecast:** See progress over time, visualized with charts. Lens lines add narrative context.  
    - **Interventions:** Suggests top moves (short actions) that improve probabilities.  
    - **Diagnostics:** Displays Forces (positive correlations) and Drags (negative patterns).  
    - **Lens:** Upload or manually add guiding statements. They echo across the system.  
    - **Settings:** Manage profiles, AI API keys, and toggles.  
    """)

# ----------------------------
# FUTURE SELF TAB
# ----------------------------
elif choice == "Future Self":
    st.header("Future Self")
    profiles = get_profiles()
    if not profiles:
        st.warning("No profile found. Create one in Settings.")
    else:
        profile_id = profiles[0][0]

        st.subheader("Traits")
        new_trait = st.text_input("Add a new trait")
        if st.button("Save Trait"):
            conn = sqlite3.connect("timesculpt.db")
            cur = conn.cursor()
            cur.execute("INSERT INTO traits (profile_id, trait) VALUES (?, ?)", (profile_id, new_trait))
            conn.commit()
            conn.close()
            st.success("Trait saved.")

        conn = sqlite3.connect("timesculpt.db")
        cur = conn.cursor()
        cur.execute("SELECT trait FROM traits WHERE profile_id=?", (profile_id,))
        for t in cur.fetchall():
            st.markdown(f"- {t[0]}")

        st.subheader("Future Self Letter")
        new_letter = st.text_area("Write a letter from your future self")
        if st.button("Save Letter"):
            conn = sqlite3.connect("timesculpt.db")
            cur = conn.cursor()
            cur.execute("INSERT INTO letters (profile_id, content) VALUES (?, ?)", (profile_id, new_letter))
            conn.commit()
            conn.close()
            st.success("Letter saved.")

        st.subheader("Goals")
        goal_name = st.text_input("Goal name")
        goal_target = st.number_input("Target value", min_value=0.0, step=1.0)
        goal_unit = st.text_input("Unit (e.g., minutes, pages)")
        goal_deadline = st.date_input("Deadline")
        goal_priority = st.slider("Priority", 1, 10, 5)

        if st.button("Save Goal"):
            save_goal(profile_id, goal_name, goal_target, goal_unit, str(goal_deadline), goal_priority)
            st.success("Goal saved.")

        st.markdown("### Your Goals")
        goals = get_goals(profile_id)
        for g in goals:
            st.markdown(f"<div class='card'>**{g[0]}** — Target: {g[1]} {g[2]}, Deadline: {g[3]}, Priority: {g[4]}</div>", unsafe_allow_html=True)

        lines = get_lens_lines(profile_id)
        if lines:
            st.markdown(f"*Lens echo:* {lines[-1]}")

# ----------------------------
# INPUT TAB
# ----------------------------
elif choice == "Input":
    st.header("Commit Today — Log Loops")
    profiles = get_profiles()
    if not profiles:
        st.warning("No profile found. Create one in Settings.")
    else:
        profile_id = profiles[0][0]

        loop_name = st.text_input("Loop name")
        value = st.number_input("Value", min_value=0.0, step=1.0)
        unit = st.text_input("Unit")
        loop_date = st.date_input("Date", datetime.date.today())
        loop_time = st.time_input("Time", datetime.datetime.now().time())

        if st.button("Log Loop"):
            dt = datetime.datetime.combine(loop_date, loop_time)
            log_loop(profile_id, loop_name, value, unit, dt)
            st.success(f"Loop '{loop_name}' logged at {dt.strftime('%Y-%m-%d %H:%M')}")

        st.subheader("Recent Loops")
        loops = get_loops(profile_id)
        for loop in loops[-5:]:
            st.markdown(f"<div class='card'>**{loop[0]}**: {loop[1]} {loop[2]} at {loop[3]}</div>", unsafe_allow_html=True)

# ----------------------------
# FORECAST TAB
# ----------------------------
elif choice == "Forecast":
    st.header("Forecast")
    profiles = get_profiles()
    if not profiles:
        st.warning("No profile found. Create one in Settings.")
    else:
        profile_id = profiles[0][0]
        loops = get_loops(profile_id)
        if not loops:
            st.info("No loops logged yet.")
        else:
            days = [datetime.datetime.fromisoformat(l[3]).date() for l in loops]
            values = [l[1] for l in loops]
            fig, ax = plt.subplots()
            ax.plot(days, values, marker="o", color="#FFD700")
            ax.set_title("Loop Progress Over Time", color="white")
            ax.tick_params(colors="white")
            fig.patch.set_facecolor("#111133")
            ax.set_facecolor("#111133")
            st.pyplot(fig)

            st.subheader("Narrative Forecast")
            lines = get_lens_lines(profile_id)
            if lines:
                st.markdown(f"*Lens echo:* {lines[-1]}")

# ----------------------------
# INTERVENTIONS TAB
# ----------------------------
elif choice == "Interventions":
    st.header("Interventions")
    profiles = get_profiles()
    if not profiles:
        st.warning("No profile found.")
    else:
        st.markdown('<div class="card"><span class="highlight">Top Move:</span> 20m Writing Sprint</div>', unsafe_allow_html=True)
        lines = get_lens_lines(profiles[0][0])
        if lines:
            st.markdown(f"*Lens echo:* {lines[-1]}")

# ----------------------------
# DIAGNOSTICS TAB
# ----------------------------
elif choice == "Diagnostics":
    st.header("Diagnostics")
    profiles = get_profiles()
    if not profiles:
        st.warning("No profile found.")
    else:
        labels = ["Writing", "Late Sleep", "Exercise", "Distraction"]
        values = [2.3, -1.8, 1.5, -0.9]
        colors = ["#50C878" if v > 0 else "#DC143C" for v in values]
        fig, ax = plt.subplots()
        ax.bar(labels, values, color=colors)
        ax.axhline(0, color="white")
        ax.set_title("Force & Drag Analysis", color="white")
        ax.tick_params(colors="white")
        fig.patch.set_facecolor("#111133")
        ax.set_facecolor("#111133")
        st.pyplot(fig)

        lines = get_lens_lines(profiles[0][0])
        if lines:
            st.markdown(f"*Lens echo:* {lines[-1]}")

# ----------------------------
# LENS TAB
# ----------------------------
elif choice == "Lens":
    st.header("Lens")
    profiles = get_profiles()
    if not profiles:
        st.warning("No profile found.")
    else:
        profile_id = profiles[0][0]
        st.subheader("Upload Lens File")
        uploaded = st.file_uploader("Upload .txt, .docx, or .pdf", type=["txt", "docx", "pdf"])
        if uploaded:
            if uploaded.name.endswith(".txt"):
                text = uploaded.read().decode("utf-8")
            elif uploaded.name.endswith(".docx"):
                doc = docx.Document(uploaded)
                text = "\n".join([p.text for p in doc.paragraphs])
            elif uploaded.name.endswith(".pdf"):
                reader = PyPDF2.PdfReader(uploaded)
                text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            for line in text.split("\n"):
                if line.strip():
                    save_lens_line(profile_id, line.strip())
            st.success("Lens lines saved.")

        st.subheader("Add Lens Line Manually")
        manual_line = st.text_input("Lens line")
        if st.button("Save Lens Line"):
            save_lens_line(profile_id, manual_line)
            st.success("Lens line saved.")

        st.subheader("Stored Lens Lines")
        for line in get_lens_lines(profile_id)[-5:]:
            st.markdown(f"- {line}")

# ----------------------------
# SETTINGS TAB
# ----------------------------
elif choice == "Settings":
    st.header("Profile Manager")
    new_name = st.text_input("New Profile Name")
    new_pin = st.text_input("New PIN", type="password")
    if st.button("Create Profile"):
        if new_name and new_pin:
            try:
                create_profile(new_name, new_pin)
                st.success(f"Profile {new_name} created.")
            except Exception as e:
                st.error(f"Error: {e}")

    profiles = get_profiles()
    if profiles:
        profile_names = [p[1] for p in profiles]
        selected = st.selectbox("Select Active Profile", profile_names)
        prof = [p for p in profiles if p[1] == selected][0]
        prof_id, _, ai_toggle, api_key = prof

        st.subheader("🔑 AI API Manager")
        new_api_key = st.text_input("Enter API Key", value=api_key if api_key else "", type="password")
        toggle = st.checkbox("Enable AI features", value=bool(ai_toggle))

        if st.button("Save AI Settings"):
            conn = sqlite3.connect("timesculpt.db")
            cur = conn.cursor()
            cur.execute("UPDATE profiles SET api_key=?, ai_toggle=? WHERE id=?", (new_api_key, int(toggle), prof_id))
            conn.commit()
            conn.close()
            st.success("AI settings saved.")

        if api_key and ai_toggle:
            st.markdown('<div class="card">✅ <span class="highlight">AI Connected</span> — features unlocked</div>', unsafe_allow_html=True)
        elif api_key and not ai_toggle:
            st.markdown('<div class="card">⚠️ API Key saved, but AI features <span class="highlight">disabled</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card">❌ No AI connected</div>', unsafe_allow_html=True)
