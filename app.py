# ======================
# TimeSculpt Navigator 6.4
# ======================
# Monolithic Build — Phase 6.4 (Locked)
# Features: Loops, Goals, Forecast, Diagnostics, Lens, Companion, Identity Map, Guide
# Includes optional AI narration + guidance (toggle per profile)

import streamlit as st
import sqlite3
import os, re, io, uuid
from datetime import datetime
from typing import List

# Optional parsers (graceful fail if not installed)
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import docx
except Exception:
    docx = None

from graphviz import Digraph

# ----------------------
# Config & Styling
# ----------------------
st.set_page_config(page_title="TimeSculpt Navigator 6.4", page_icon="⏳", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] {
    font-size: 16px;
}
h1, h2, h3 {
    letter-spacing: -0.5px;
}
</style>
""", unsafe_allow_html=True)

DB = "timesculpt.db"

# ----------------------
# Utility Functions
# ----------------------

def normalize_percent(value: float) -> float:
    """Clamp percentage values to 0–100 range."""
    try:
        return max(0.0, min(100.0, float(value)))
    except Exception:
        return 0.0

def explain_percentages(kind: str) -> str:
    """Return tooltip explanations for % values."""
    if kind == "loops":
        return "Momentum alignment: ratio of Forces (+) to Drags (–)."
    elif kind == "goals":
        return "Weighted completion of milestones within a goal."
    elif kind == "forecast":
        return "Trajectory projection: Goals (40%) + Loops (40%) + Interventions (20%)."
    elif kind == "diagnostics":
        return "Balance index: Forces vs Drags across your activity history."
    return "Percentage meaning not defined."

# ----------------------
# AI Integration
# ----------------------
try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    OpenAI = None
    _OPENAI_OK = False

def get_openai_client(api_key: str | None):
    """Return an OpenAI client or None if unavailable."""
    if not _OPENAI_OK:
        return None
    key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None

def ai_narrate(prompt: str, api_key: str | None, model: str = "gpt-4o-mini") -> str | None:
    """
    Optional AI narration. Returns None if client/key missing.
    Used by Diagnostics + Companion.
    """
    client = get_openai_client(api_key)
    if not client:
        return None
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are the TimeSculpt narrative voice: concise, precise, poetic when useful."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=350,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None
# ======================
# Database Schema + Migrations
# ======================

def get_connection():
    """Open DB connection with thread-safety off (Streamlit compatible)."""
    return sqlite3.connect(DB, check_same_thread=False)

def migrate(conn):
    """Create or upgrade all required tables."""
    c = conn.cursor()

    # --- Profiles (with AI toggle + key) ---
    c.execute("""
    CREATE TABLE IF NOT EXISTS profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        pin_hash TEXT,
        api_key TEXT,
        use_ai INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # --- Loops ---
    c.execute("""
    CREATE TABLE IF NOT EXISTS loops (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        name TEXT,
        is_force INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # --- Goals ---
    c.execute("""
    CREATE TABLE IF NOT EXISTS goals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        name TEXT,
        description TEXT,
        weight INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # --- Milestones ---
    c.execute("""
    CREATE TABLE IF NOT EXISTS milestones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        goal_id INTEGER,
        name TEXT,
        is_complete INTEGER DEFAULT 0,
        weight REAL DEFAULT 1.0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # --- Lens (Knowledge Field) ---
    c.execute("""
    CREATE TABLE IF NOT EXISTS lens (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        text TEXT,
        category TEXT,
        source TEXT,
        active INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # --- Lens Preferences (Fuse Modes etc.) ---
    c.execute("""
    CREATE TABLE IF NOT EXISTS lens_prefs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        fuse_mode TEXT DEFAULT 'basic'
    )
    """)

    conn.commit()
# ======================
# Loops (Momentum Engine)
# ======================

def add_loop(conn, profile_id: int, name: str, is_force: bool):
    c = conn.cursor()
    c.execute("INSERT INTO loops (profile_id, name, is_force) VALUES (?, ?, ?)",
              (profile_id, name, 1 if is_force else 0))
    conn.commit()

def get_loops_summary(conn, profile_id: int):
    c = conn.cursor()
    c.execute("SELECT is_force, COUNT(*) FROM loops WHERE profile_id=? GROUP BY is_force", (profile_id,))
    counts = {row[0]: row[1] for row in c.fetchall()}
    forces = counts.get(1, 0)
    drags = counts.get(0, 0)
    total = forces + drags
    percent = normalize_percent((forces / total) * 100 if total > 0 else 0)
    return forces, drags, percent

def show_loops(conn, profile_id: int):
    st.header("🔄 Loops — Momentum")

    # Input new loop
    with st.form("new_loop_form", clear_on_submit=True):
        loop_name = st.text_input("Loop name (e.g., Journaling, Skipped Workout)")
        loop_type = st.radio("Type", ["Force (+)", "Drag (–)"], horizontal=True)
        submitted = st.form_submit_button("Add Loop")
        if submitted and loop_name.strip():
            add_loop(conn, profile_id, loop_name.strip(), is_force=(loop_type == "Force (+)"))
            st.success(f"Added loop: {loop_name} [{loop_type}]")

    # Show summary
    forces, drags, percent = get_loops_summary(conn, profile_id)
    st.metric("Momentum Alignment", f"{percent:.1f}%", help=explain_percentages("loops"))
    st.progress(percent / 100.0 if percent else 0)

    st.caption(f"Logged: {forces} Forces, {drags} Drags.")


# ======================
# Goals (Anchors Engine)
# ======================

def add_goal(conn, profile_id: int, name: str, description: str, weight: int):
    c = conn.cursor()
    c.execute("INSERT INTO goals (profile_id, name, description, weight) VALUES (?, ?, ?, ?)",
              (profile_id, name, description, weight))
    conn.commit()

def add_milestone(conn, goal_id: int, name: str, weight: float):
    c = conn.cursor()
    c.execute("INSERT INTO milestones (goal_id, name, weight) VALUES (?, ?, ?)",
              (goal_id, name, weight))
    conn.commit()

def calculate_goal_progress(conn, goal_id: int) -> float:
    c = conn.cursor()
    c.execute("SELECT weight, is_complete FROM milestones WHERE goal_id=?", (goal_id,))
    rows = c.fetchall()
    if not rows:
        return 0.0
    total_weight = sum([r[0] for r in rows])
    completed = sum([r[0] for r in rows if r[1] == 1])
    return normalize_percent((completed / total_weight) * 100 if total_weight > 0 else 0)

def goal_narrative(progress: float, weight: int) -> str:
    if progress == 0:
        return f"This goal carries weight {weight}, but no step has yet been taken."
    elif progress < 50:
        return f"You’ve begun, but most of the weight still lies ahead."
    elif progress < 100:
        return f"Momentum is carrying you forward — the finish is visible, but not yet claimed."
    else:
        return f"The weight has been carried. This goal is complete."

def show_goals(conn, profile_id: int):
    st.header("🎯 Goals — Anchors")

    # Add new goal
    with st.expander("➕ Add New Goal"):
        name = st.text_input("Goal Name")
        description = st.text_area("Description")
        weight = st.slider("Importance (1=low, 10=high)", 1, 10, 5)
        if st.button("Save Goal"):
            if name.strip():
                add_goal(conn, profile_id, name.strip(), description.strip(), weight)
                st.success(f"Goal '{name}' added.")

    # List goals
    c = conn.cursor()
    c.execute("SELECT id, name, description, weight FROM goals WHERE profile_id=?", (profile_id,))
    goals = c.fetchall()

    for gid, gname, gdesc, gweight in goals:
        st.subheader(f"{gname} (Weight {gweight})")
        if gdesc:
            st.caption(gdesc)

        # Add milestone
        with st.expander("➕ Add Milestone"):
            ms_name = st.text_input(f"Milestone for {gname}", key=f"ms_{gid}")
            ms_weight = st.number_input("Weight", 0.1, 10.0, 1.0, step=0.1, key=f"msw_{gid}")
            if st.button(f"Add Milestone to {gname}", key=f"addms_{gid}"):
                if ms_name.strip():
                    add_milestone(conn, gid, ms_name.strip(), ms_weight)
                    st.success("Milestone added.")

        # Show milestones
        c.execute("SELECT id, name, is_complete, weight FROM milestones WHERE goal_id=?", (gid,))
        milestones = c.fetchall()
        for mid, mname, mis_complete, mweight in milestones:
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                st.write(f"- {mname} ({mweight} weight)")
            with col2:
                if st.checkbox("Done", value=bool(mis_complete), key=f"msdone_{mid}"):
                    c.execute("UPDATE milestones SET is_complete=1 WHERE id=?", (mid,))
                    conn.commit()
            with col3:
                if st.button("❌", key=f"delms_{mid}"):
                    c.execute("DELETE FROM milestones WHERE id=?", (mid,))
                    conn.commit()

        # Progress
        progress = calculate_goal_progress(conn, gid)
        st.progress(progress / 100)
        st.metric("Progress", f"{progress:.1f}%", help=explain_percentages("goals"))

        # Narrative
        st.info(goal_narrative(progress, gweight))
# ======================
# Forecast (Trajectory Projection)
# ======================

def calculate_forecast(loop_percent: float, goal_percent: float, intervention_score: float = 50.0) -> float:
    """Blend loops, goals, and interventions into forecast alignment %."""
    alignment = (0.4 * loop_percent) + (0.4 * goal_percent) + (0.2 * intervention_score)
    return normalize_percent(alignment)

def show_forecast(conn, profile_id: int, api_key: str | None, use_ai: bool):
    st.header("📈 Forecast — Trajectory")

    # Gather data
    forces, drags, loop_percent = get_loops_summary(conn, profile_id)
    c = conn.cursor()
    c.execute("SELECT id FROM goals WHERE profile_id=?", (profile_id,))
    goals = c.fetchall()
    if goals:
        goal_progresses = [calculate_goal_progress(conn, gid[0]) for gid in goals]
        goal_percent = sum(goal_progresses) / len(goal_progresses)
    else:
        goal_percent = 0.0

    # Forecast
    forecast_percent = calculate_forecast(loop_percent, goal_percent)
    st.metric("Alignment %", f"{forecast_percent:.1f}%", help=explain_percentages("forecast"))
    st.progress(forecast_percent / 100.0)

    # Narrative (AI optional)
    if use_ai and api_key:
        prompt = (
            f"Loops momentum: {loop_percent:.1f}%. "
            f"Goals progress: {goal_percent:.1f}%. "
            f"Forecast alignment: {forecast_percent:.1f}%. "
            "Write a short, poetic, declarative line in TimeSculpt's tone explaining the trajectory."
        )
        ai_text = ai_narrate(prompt, api_key)
        if ai_text:
            st.info(ai_text)


# ======================
# Diagnostics (Story of Behavior)
# ======================

def generate_diagnostics_narrative(forces: int, drags: int, forecast_percent: float,
                                   api_key: str | None, use_ai: bool) -> str:
    """Narrative for Diagnostics — AI if enabled, else fallback."""
    base_story = f"You logged {forces} Forces and {drags} Drags. Momentum is {forecast_percent:.1f}% aligned."

    if not (use_ai and api_key):
        return base_story

    prompt = (
        f"User has {forces} Forces and {drags} Drags logged. "
        f"Forecast alignment is {forecast_percent:.1f}%. "
        "Explain, in the TimeSculpt tone, why momentum is rising, stalling, or fracturing."
    )
    ai_response = ai_narrate(prompt, api_key)
    return ai_response or base_story

def show_diagnostics(conn, profile_id: int, api_key: str | None, use_ai: bool):
    st.header("⚖️ Diagnostics — Story")

    forces, drags, loop_percent = get_loops_summary(conn, profile_id)
    c = conn.cursor()
    c.execute("SELECT id FROM goals WHERE profile_id=?", (profile_id,))
    goals = c.fetchall()
    if goals:
        goal_progresses = [calculate_goal_progress(conn, gid[0]) for gid in goals]
        goal_percent = sum(goal_progresses) / len(goal_progresses)
    else:
        goal_percent = 0.0

    forecast_percent = calculate_forecast(loop_percent, goal_percent)

    # Display balance
    st.metric("Forces", forces)
    st.metric("Drags", drags)
    st.metric("Balance Index", f"{loop_percent:.1f}%", help=explain_percentages("diagnostics"))

    # Narrative
    narrative = generate_diagnostics_narrative(forces, drags, forecast_percent, api_key, use_ai)
    st.info(narrative)

    # Placeholder: Trend chart (would show Forces vs Drags over time if loop timestamps are used)
    st.caption("Trend view of Forces vs Drags (to be expanded into timeline chart).")
# ======================
# Lens (Meaning Field)
# ======================

def add_lens_entry(conn, profile_id: int, text: str, category: str, source: str):
    c = conn.cursor()
    c.execute("INSERT INTO lens (profile_id, text, category, source) VALUES (?, ?, ?, ?)",
              (profile_id, text, category, source))
    conn.commit()

def get_lens_entries(conn, profile_id: int):
    c = conn.cursor()
    c.execute("SELECT id, text, category, source, active FROM lens WHERE profile_id=?", (profile_id,))
    return c.fetchall()

def show_lens(conn, profile_id: int):
    st.header("📚 Lens — Meaning")

    # Add new entry
    with st.expander("➕ Add Lens Entry"):
        lens_text = st.text_area("Text / Quote / Reflection")
        category = st.text_input("Category (optional)")
        source = st.text_input("Source (optional)")
        if st.button("Save Entry"):
            if lens_text.strip():
                add_lens_entry(conn, profile_id, lens_text.strip(), category.strip(), source.strip())
                st.success("Entry added to Lens.")

    # Show entries
    entries = get_lens_entries(conn, profile_id)
    for eid, text, category, source, active in entries:
        st.write(f"**{category or 'General'}** — {source or 'Unknown'}")
        st.caption(text)
        st.toggle("Active", value=bool(active), key=f"lens_active_{eid}")
        st.divider()


# ======================
# Companion (Advisor)
# ======================

def generate_companion_advice(loop_percent: float, goal_percent: float, forecast_percent: float,
                              api_key: str | None, use_ai: bool):
    """
    Generate advice list — AI if enabled, else fallback.
    """
    advice = []
    if loop_percent < 50:
        advice.append("Increase your Forces — reduce Drags.")
    if goal_percent < 50:
        advice.append("Revisit your milestones — anchors are weak.")
    if forecast_percent < 50:
        advice.append("Trajectory is slipping — recalibrate today.")
    if not advice:
        advice.append("Momentum strong. Continue reinforcing current path.")

    if not (use_ai and api_key):
        return advice

    # AI-enhanced version
    prompt = (
        f"Loops momentum: {loop_percent:.1f}%. "
        f"Goal progress: {goal_percent:.1f}%. "
        f"Forecast alignment: {forecast_percent:.1f}%. "
        "Provide 3 pieces of advice in the TimeSculpt tone, ranked and declarative."
    )
    ai_response = ai_narrate(prompt, api_key)
    if ai_response:
        ai_list = [line.strip("-• ") for line in ai_response.splitlines() if line.strip()]
        return ai_list or advice

    return advice

def show_companion(conn, profile_id: int, api_key: str | None, use_ai: bool):
    st.header("🧭 Companion — Guidance")

    # Gather data
    forces, drags, loop_percent = get_loops_summary(conn, profile_id)
    c = conn.cursor()
    c.execute("SELECT id FROM goals WHERE profile_id=?", (profile_id,))
    goals = c.fetchall()
    if goals:
        goal_progresses = [calculate_goal_progress(conn, gid[0]) for gid in goals]
        goal_percent = sum(goal_progresses) / len(goal_progresses)
    else:
        goal_percent = 0.0

    forecast_percent = calculate_forecast(loop_percent, goal_percent)

    # Advice
    advice_list = generate_companion_advice(loop_percent, goal_percent, forecast_percent, api_key, use_ai)
    for idx, item in enumerate(advice_list, 1):
        st.write(f"{idx}. {item}")
# ======================
# Identity Map (Recursive Mesh)
# ======================

def show_identity_map(conn, profile_id: int):
    st.header("🌀 Identity Map")

    forces, drags, loop_percent = get_loops_summary(conn, profile_id)

    c = conn.cursor()
    c.execute("SELECT id FROM goals WHERE profile_id=?", (profile_id,))
    goals = c.fetchall()
    if goals:
        goal_progresses = [calculate_goal_progress(conn, gid[0]) for gid in goals]
        goal_percent = sum(goal_progresses) / len(goal_progresses)
    else:
        goal_percent = 0.0

    forecast_percent = calculate_forecast(loop_percent, goal_percent)

    # Lens count
    c.execute("SELECT COUNT(*) FROM lens WHERE profile_id=? AND active=1", (profile_id,))
    lens_count = c.fetchone()[0]

    # Build graph
    dot = Digraph(comment="Identity Map", format="png")
    dot.attr(rankdir="LR", size="8,5")

    # Nodes (dynamic size/color)
    loop_color = "gold" if loop_percent >= 50 else "crimson"
    dot.node("Loops", f"Loops\n{loop_percent:.1f}%", shape="ellipse", style="filled", color=loop_color)

    dot.node("Goals", f"Goals\n{goal_percent:.1f}%", shape="box", style="filled", color="lightblue")

    forecast_color = "purple" if forecast_percent >= 50 else "gray"
    dot.node("Forecast", f"Forecast\n{forecast_percent:.1f}%", shape="diamond", style="filled", color=forecast_color)

    dot.node("Diagnostics", f"Diagnostics\nBalance {loop_percent:.1f}%", shape="parallelogram", style="filled", color="red")

    dot.node("Lens", f"Lens\n{lens_count} entries", shape="note", style="filled", color="green")

    dot.node("Companion", "Companion\n(Guidance)", shape="hexagon", style="filled", color="orange")

    # Edges (recursive mesh)
    dot.edge("Loops", "Goals", label="tags")
    dot.edge("Loops", "Forecast", label="feeds")
    dot.edge("Goals", "Forecast", label="anchors")
    dot.edge("Forecast", "Diagnostics", label="story")
    dot.edge("Diagnostics", "Lens", label="infuses meaning")
    dot.edge("Lens", "Companion", label="identity echoes")
    dot.edge("Diagnostics", "Companion", label="guidance")
    dot.edge("Forecast", "Companion", label="probability weighting")
    dot.edge("Companion", "Loops", label="advice → new loops")

    # Render
    file_path = "/tmp/identity_map"
    dot.render(file_path, cleanup=True)

    st.image(file_path + ".png", use_column_width=True)
    st.caption("Identity Map: the recursive mesh of Loops, Goals, Forecast, Diagnostics, Lens, and Companion.")


# ======================
# Guide Tab (Orientation Scroll)
# ======================

def show_guide():
    st.header("📘 TimeSculpt Navigator — Guide")

    st.markdown("""
    ### 🜂 Overview
    This is not a tracker.  
    It is a mirror.  
    It shows you what you do, where it leads, and how to sculpt it.  

    Every log, every milestone, every choice becomes a signal.  
    Together they form the field of your trajectory.  
    This app does not predict your future — it shows you the one you are already becoming.  
    """)

    st.markdown("""
    ### 🔄 Loops — Momentum
    - Each loop is a daily act.  
    - Tag it as a **Force (+)** or a **Drag (–)**.  
    - The balance becomes your **Momentum %**.  

    > “Momentum is truth. You move the way your actions align, not the way you hope.”  

    **How to use:**  
    - Log at least 3 loops daily.  
    - Review your Force/Drag ratio weekly.  
    """)

    st.markdown("""
    ### 🎯 Goals — Anchors
    - Define a **Goal**. Assign it weight (1–10).  
    - Break it into **Milestones**, each with its own weight.  
    - Progress = **weighted completion %**.  

    > “A goal is not a checklist. It is an anchor for who you are becoming.”  

    **How to use:**  
    - Be honest with weights — they shape your forecast.  
    - Review milestones when loops stall.  
    """)

    st.markdown("""
    ### 📈 Forecast — Trajectory
    - Blends:  
      - **Goal %** (destination)  
      - **Loop Momentum %** (movement)  
      - **Interventions** (course corrections)  
    - Outputs a single **Alignment %**.  

    > “The future is probability until you align. Forecast shows where your steps are carrying you.”  

    **How to use:**  
    - Check often, but don’t obsess.  
    - Use Forecast as a compass, not a verdict.  
    """)

    st.markdown("""
    ### ⚖️ Diagnostics — Story
    - Force vs Drag → **Balance Index**.  
    - Converts numbers into narrative.  
    - Shows *why* you are moving or stuck.  

    > “Numbers reveal what happened. Diagnostics reveals why.”  

    **How to use:**  
    - Review patterns weekly.  
    - Let the story inform your next loop.  
    """)

    st.markdown("""
    ### 📚 Lens — Meaning
    - Upload texts. Add notes. Store reflections.  
    - Lens fuses your words with your data.  
    - When consulted, it speaks back as **identity echoes**.  

    > “The field remembers what you feed it.”  

    **How to use:**  
    - Collect passages that matter to you.  
    - Search and revisit them often — the echoes shift with your state.  
    """)

    st.markdown("""
    ### 🧭 Companion — Guidance
    - Reads Forecast, Diagnostics, and Lens.  
    - Suggests ranked advice.  
    - Learns from your response.  

    > “No oracle is complete without counsel. The Companion advises, but you choose.”  

    **How to use:**  
    - Accept advice if it resonates.  
    - Reject if misaligned — the Companion will refine.  
    """)

    st.markdown("""
    ### ✍️ Closing
    This app is not passive.  
    It is recursive.  
    The more you use it, the sharper it becomes.  
    Every log, every goal, every word you give it sculpts the mirror — and in turn, sculpts you.  
    """)
# ======================
# Main App Router
# ======================

def get_profile(conn, profile_id: int):
    c = conn.cursor()
    c.execute("SELECT id, name, api_key, use_ai FROM profiles WHERE id=?", (profile_id,))
    row = c.fetchone()
    if row:
        return {"id": row[0], "name": row[1], "api_key": row[2], "use_ai": bool(row[3])}
    return None

def list_profiles(conn):
    c = conn.cursor()
    c.execute("SELECT id, name FROM profiles ORDER BY created_at DESC")
    return c.fetchall()

def create_profile(conn, name: str, api_key: str = "", use_ai: bool = False):
    c = conn.cursor()
    c.execute("INSERT INTO profiles (name, api_key, use_ai) VALUES (?, ?, ?)",
              (name, api_key, 1 if use_ai else 0))
    conn.commit()

def update_profile_ai(conn, profile_id: int, api_key: str, use_ai: bool):
    c = conn.cursor()
    c.execute("UPDATE profiles SET api_key=?, use_ai=? WHERE id=?",
              (api_key, 1 if use_ai else 0, profile_id))
    conn.commit()

def main():
    st.title("⏳ TimeSculpt Navigator 6.4")

    conn = get_connection()
    migrate(conn)

    # Profile selection
    st.sidebar.header("Profiles")
    profiles = list_profiles(conn)
    profile_names = [p[1] for p in profiles]
    profile_lookup = {p[1]: p[0] for p in profiles}

    choice = st.sidebar.selectbox("Select Profile", ["(None)"] + profile_names)
    profile_id = profile_lookup.get(choice)

    if not profile_id:
        st.sidebar.subheader("Create Profile")
        new_name = st.sidebar.text_input("Profile Name")
        new_api = st.sidebar.text_input("OpenAI API Key (optional)", type="password")
        new_ai = st.sidebar.checkbox("Enable AI Narration", value=False)
        if st.sidebar.button("Create"):
            if new_name.strip():
                create_profile(conn, new_name.strip(), new_api.strip(), new_ai)
                st.experimental_rerun()
        st.info("⚠️ No profile selected. Create one in the sidebar.")
        return

    # Load profile
    profile = get_profile(conn, profile_id)
    api_key = profile["api_key"]
    use_ai = profile["use_ai"]

    # Profile settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("Profile Settings")
    api_input = st.sidebar.text_input("OpenAI API Key", value=api_key or "", type="password")
    ai_toggle = st.sidebar.checkbox("Enable AI Narration", value=use_ai)
    if st.sidebar.button("Update Profile"):
        update_profile_ai(conn, profile_id, api_input.strip(), ai_toggle)
        st.success("Profile updated.")
        st.experimental_rerun()

    # Navigation
    st.sidebar.markdown("---")
    tab_choice = st.sidebar.radio("Navigate", [
        "Loops", "Goals", "Forecast", "Diagnostics", "Lens", "Companion", "Identity Map", "Guide"
    ])

    if tab_choice == "Loops":
        show_loops(conn, profile_id)
    elif tab_choice == "Goals":
        show_goals(conn, profile_id)
    elif tab_choice == "Forecast":
        show_forecast(conn, profile_id, api_key, use_ai)
    elif tab_choice == "Diagnostics":
        show_diagnostics(conn, profile_id, api_key, use_ai)
    elif tab_choice == "Lens":
        show_lens(conn, profile_id)
    elif tab_choice == "Companion":
        show_companion(conn, profile_id, api_key, use_ai)
    elif tab_choice == "Identity Map":
        show_identity_map(conn, profile_id)
    elif tab_choice == "Guide":
        show_guide()

if __name__ == "__main__":
    main()
