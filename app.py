# ================================
# TimeSculpt — Polished Visual Build
# ================================
import streamlit as st
import sqlite3
import bcrypt
import datetime
import matplotlib.pyplot as plt

# ----------------------------
# Config & Styling
# ----------------------------
st.set_page_config(page_title="TimeSculpt", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom, #0a0a0f, #111133);
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f0f1f;
    }
    .css-1v3fvcr, .css-1d391kg, .css-qri22k, .css-16idsys, .css-1v0mbdj {
        color: white !important;
    }
    .css-1v3fvcr:hover {
        color: #FFD700 !important;
        font-weight: bold;
    }
    /* Input labels */
    label, .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: #FFD700 !important;
        font-weight: bold;
    }
    input, textarea {
        background-color: #1e1e2e !important;
        border-radius: 8px !important;
        color: white !important;
    }
    /* Cards */
    .card {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        background-color: #111122;
        color: white;
        border: 1px solid #FFD700;
    }
    .highlight {color: #FFD700; font-weight: bold;}
    /* Page titles */
    h1, h2, h3 {
        color: #FFD700 !important;
        font-weight: bold;
    }
    /* Watermark */
    #watermark {
        position: fixed;
        bottom: 10px;
        right: 20px;
        color: #FFD700;
        font-size: 14px;
        opacity: 0.7;
    }
    </style>
""", unsafe_allow_html=True)

# Add watermark
st.markdown('<div id="watermark">TimeSculpt</div>', unsafe_allow_html=True)

# ----------------------------
# DB Setup
# ----------------------------
def init_db():
    conn = sqlite3.connect("timesculpt.db")
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        pin_hash TEXT,
        ai_toggle INTEGER DEFAULT 0,
        api_key TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS goals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        name TEXT,
        target REAL,
        unit TEXT,
        deadline TEXT,
        priority REAL,
        FOREIGN KEY(profile_id) REFERENCES profiles(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS loops (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        loop_name TEXT,
        value REAL,
        unit TEXT,
        timestamp TEXT,
        FOREIGN KEY(profile_id) REFERENCES profiles(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS traits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        trait TEXT,
        FOREIGN KEY(profile_id) REFERENCES profiles(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS letters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        content TEXT,
        FOREIGN KEY(profile_id) REFERENCES profiles(id)
    )
    """)

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

def log_loop(profile_id, loop_name, value, unit):
    conn = sqlite3.connect("timesculpt.db")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO loops (profile_id, loop_name, value, unit, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (profile_id, loop_name, value, unit, datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_loops(profile_id):
    conn = sqlite3.connect("timesculpt.db")
    cur = conn.cursor()
    cur.execute("SELECT loop_name, value, unit, timestamp FROM loops WHERE profile_id=?", (profile_id,))
    rows = cur.fetchall()
    conn.close()
    return rows

# ----------------------------
# Sidebar Navigation
# ----------------------------
tabs = ["Guide", "Future Self", "Input", "Forecast", "Interventions", "Diagnostics", "Settings"]
choice = st.sidebar.radio("Navigate", tabs)

# ----------------------------
# GUIDE TAB
# ----------------------------
if choice == "Guide":
    st.title("TimeSculpt")
    st.markdown("**TimeSculpt is not a tracker. It is a sculptor’s tool.**")
    st.markdown("""
    Every log, loop, and choice bends probability toward the self you choose.  
    This guide shows you how to walk that path.
    """)

    st.subheader("📖 About TimeSculpt")
    st.markdown("""
    TimeSculpt helps you evolve by aligning **Future Self identity**, **daily loops**, and **probability forecasts**.  

    🔹 With consistent use you can expect:  
    - Sharper clarity about who you’re becoming  
    - Goals that naturally align with your identity  
    - Recurring patterns surfaced in Diagnostics  
    - Forecasts that show ETA and probability shifts  
    - Small Interventions that bend outcomes fast  

    Over weeks and months, the app becomes less a tool, more a mirror — reflecting the Sculptor you already are.
    """)

    st.subheader("🧭 How to Use Each Tab")
    st.markdown("""
    **Step 1 — Profiles**  
    Go to Settings → Profile Manager. Create a profile with your name and PIN.  

    **Step 2 — Future Self**  
    Define traits, rituals, and letters from your Future Self.  

    **Step 3 — Goals**  
    Attach goals with name, target, deadline, and priority.  

    **Step 4 — Input (Log Loops)**  
    Record daily actions (loops). These normalize into your goals.  

    **Step 5 — Forecast**  
    View probability, ETA, and charts of your trajectory.  

    **Step 6 — Interventions**  
    Smallest move suggestions to collapse probability faster.  

    **Step 7 — Diagnostics**  
    Forces (helpful patterns) and Drags (resisting loops) revealed.  
    """)

# ----------------------------
# FUTURE SELF TAB
# ----------------------------
elif choice == "Future Self":
    st.header("Future Self")

    profiles = get_profiles()
    if not profiles:
        st.warning("No profile found. Please create one in Settings.")
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
        traits = cur.fetchall()
        conn.close()
        for t in traits:
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

# ----------------------------
# INPUT TAB
# ----------------------------
elif choice == "Input":
    st.header("Commit Today — Log Loops")

    profiles = get_profiles()
    if not profiles:
        st.warning("No profile found. Please create one in Settings.")
    else:
        profile_id = profiles[0][0]
        loop_name = st.text_input("Loop name")
        value = st.number_input("Value", min_value=0.0, step=1.0)
        unit = st.text_input("Unit (e.g., minutes, reps)")

        if st.button("Log Loop"):
            if loop_name and unit:
                log_loop(profile_id, loop_name, value, unit)
                st.success(f"Loop '{loop_name}' logged at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.error("Please enter both loop name and unit.")

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
        st.warning("No profile found. Please create one in Settings.")
    else:
        profile_id = profiles[0][0]
        loops = get_loops(profile_id)
        if not loops:
            st.info("No loops logged yet. Add some in Input.")
        else:
            days = [datetime.datetime.fromisoformat(l[3]).date() for l in loops]
            values = [l[1] for l in loops]

            fig, ax = plt.subplots()
            ax.plot(days, values, marker="o", color="#FFD700", linewidth=2)
            ax.set_title("Loop Progress Over Time", color="white")
            ax.set_xlabel("Date", color="white")
            ax.set_ylabel("Value", color="white")
            ax.tick_params(colors="white")
            fig.patch.set_facecolor("#111133")
            ax.set_facecolor("#111133")
            st.pyplot(fig)

# ----------------------------
# INTERVENTIONS TAB
# ----------------------------
elif choice == "Interventions":
    st.header("Interventions")

    profiles = get_profiles()
    if not profiles:
        st.warning("No profile found. Please create one in Settings.")
    else:
        st.markdown('<div class="card"><span class="highlight">Top Move:</span> 20m Writing Sprint</div>', unsafe_allow_html=True)

        x = ["Writing Sprint", "Morning Routine", "Less Scrolling"]
        y = [0.8, 0.6, 0.4]
        fig, ax = plt.subplots()
        ax.bar(x, y, color="#FFD700")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Impact Score", color="white")
        ax.set_title("Intervention Effectiveness", color="white")
        ax.tick_params(colors="white")
        fig.patch.set_facecolor("#111133")
        ax.set_facecolor("#111133")
        st.pyplot(fig)

# ----------------------------
# DIAGNOSTICS TAB
# ----------------------------
elif choice == "Diagnostics":
    st.header("Diagnostics")

    profiles = get_profiles()
    if not profiles:
        st.warning("No profile found. Please create one in Settings.")
    else:
        labels = ["Writing", "Late Sleep", "Exercise", "Distraction"]
        values = [2.3, -1.8, 1.5, -0.9]
        colors = ["#50C878" if v > 0 else "#DC143C" for v in values]

        fig, ax = plt.subplots()
        ax.bar(labels, values, color=colors)
        ax.axhline(0, color="white", linewidth=0.8)
        ax.set_title("Force & Drag Analysis", color="white")
        ax.tick_params(colors="white")
        fig.patch.set_facecolor("#111133")
        ax.set_facecolor("#111133")
        st.pyplot(fig)

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
        else:
            st.error("Please enter a name and PIN.")

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
