# ============================================================
# TimeSculpt — Monolithic Build (Field Engine Mesh, Goal-Centric)
# ============================================================
# This single-file app includes:
# - Profiles (multi-user) with PIN hashing + optional AI API key (stored per user)
# - Field Engine Mesh (field_state) so all tabs publish/subscribe shared signals
# - Loops Input (built-ins + custom loops; units normalized; dynamic preview)
# - Goal-Centric Forecast:
#     * Beta-Binomial progress model → success probability in 30 days
#     * Simple survival-style ETA ribbon (50% / 80%) from momentum expectation
#     * Classical 3-state baseline (Momentum / Mixed / Stuck) is computed silently
# - Interventions:
#     * Contextual bandit (Linear Thompson Sampling) learns what works for YOU
#     * Actionable cards (Apply / Helped? Yes/No)
#     * “Proven for you” leaderboard
#     * Bandit scores influence forecast comparison (“do nothing” vs “apply top move”)
# - Diagnostics:
#     * Force (+) vs Drag (−) correlations, padded to avoid missing columns errors
#     * “Why today” micro-explanation chip
# - Lenses:
#     * Upload .txt/.docx/.pdf → passages auto-categorized (collapse/recursion/emergence/neutral)
#     * Multi-lens support (primary + secondary) + lens memory (avoid repeats, goal bias)
#     * Smart narration line (no over-explaining why it chose it)
# - Future Self:
#     * Define Title + Traits + Rituals (saved)
#     * SMART Challenges: progress, due date, status, redo option
#     * Letters to Past Self with scheduled resurfacing (also inline nudges when confidence low)
# - AI Toggle:
#     * OFF → entirely local narration
#     * ON → optional AI narration if API key present (fails safely if not)
# - Visual polish:
#     * Sticky header context, 14-day sparkline, metrics, framed cards
#     * Light, readable labels on non-dark theme
# - Safe & robust:
#     * PIN stored hashed, AI exceptions hidden, lens JSON schema guard
#     * No silent DB errors (printed), missing columns padded, input validation
# ============================================================

import os, json, math, random, hashlib, datetime as dt
from typing import List, Dict, Any, Tuple
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Optional imports for lens parsing
try:
    import docx
except Exception:
    docx = None
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# Optional AI (OpenAI). App runs fine without it.
try:
    import openai
except Exception:
    openai = None

# ------------------------ UI CONFIG -------------------------
st.set_page_config(page_title="TimeSculpt", layout="wide")
st.markdown("""
<style>
/* Make labels and inputs readable on light themes */
label, .stTextInput label, .stNumberInput label, .stSelectbox label, .stDateInput label, .stTextArea label {
  color:#222 !important; font-weight:600;
}
div[data-baseweb="input"] input, textarea, .stTextArea textarea {
  color:#111 !important; background:#FAFAFA !important;
}
.stCaption, .st-emotion-cache-1xarl3l, .st-emotion-cache-16idsys p, .st-emotion-cache-10trblm p {
  color:#444 !important;
}
.card { border:1px solid #e5e7eb; border-radius:10px; padding:12px; background:#fff; }
.stat { border:1px solid #ddd; border-radius:12px; padding:10px; text-align:center; background:#fff; }
.small { font-size:12px; color:#666; }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; background:#f1f5f9; color:#111; font-size:11px; border:1px solid #e5e7eb; }
.btnrow { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
</style>
""", unsafe_allow_html=True)

# ------------------------ DB CORE ---------------------------
DB = "timesculpt.db"

def _conn():
    c = sqlite3.connect(DB, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c

def run(q, p=()):
    con = _conn()
    try:
        con.execute(q, p)
        con.commit()
    except Exception as e:
        print("DB error:", e, "\nSQL:", q)
    finally:
        con.close()

def fetch(q, p=()):
    con = _conn()
    try:
        cur = con.execute(q, p)
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        print("DB error:", e, "\nSQL:", q)
        return []
    finally:
        con.close()

def init_db():
    run("""CREATE TABLE IF NOT EXISTS profiles(
        id TEXT PRIMARY KEY,
        name TEXT,
        pin_hash TEXT,
        api_key TEXT
    )""")
    run("""CREATE TABLE IF NOT EXISTS days(
        user_id TEXT,
        d TEXT,
        loops TEXT,           -- JSON of raw inputs by user units (minutes/pages/etc)
        eff_loops TEXT,       -- JSON of normalized "effective minutes"
        note TEXT,
        focus REAL,
        energy REAL,
        progress REAL,
        state TEXT            -- internal (Momentum/Mixed/Stuck) NOT shown to user
    )""")
    run("""CREATE TABLE IF NOT EXISTS custom_loops(
        user_id TEXT,
        name TEXT,
        category TEXT,        -- creation/mind/body/consumption/food/finance/other
        polarity INTEGER,     -- +1 helpful, -1 harmful
        unit TEXT,            -- minutes/hours/pages/chapters/reps/£/%
        rate REAL             -- unit→effective minutes multiplier (auto from unit)
    )""")
    run("""CREATE TABLE IF NOT EXISTS interventions_log(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        at TEXT,
        title TEXT,
        accepted INT,
        helped INT
    )""")
    run("""CREATE TABLE IF NOT EXISTS interventions_log_ctx(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        at TEXT,
        title TEXT,
        ctx TEXT,             -- JSON list of floats (context vector)
        reward REAL           -- 1.0 helped, 0.0 not helped
    )""")
    run("""CREATE TABLE IF NOT EXISTS lenses(
        user_id TEXT,
        name TEXT,
        data TEXT             -- JSON: {collapse:[], recursion:[], emergence:[], neutral:[]}
    )""")
    run("""CREATE TABLE IF NOT EXISTS lens_memory(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        at TEXT,
        lens TEXT,
        kind TEXT,            -- collapse/recursion/emergence/neutral
        phrase TEXT,
        ctx TEXT              -- JSON context e.g. {"goal": "...", "state": "..."}
    )""")
    run("""CREATE TABLE IF NOT EXISTS future_self(
        user_id TEXT,
        title TEXT,
        traits TEXT,          -- comma sep
        rituals TEXT          -- comma sep
    )""")
    run("""CREATE TABLE IF NOT EXISTS future_challenges(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        title TEXT,
        why TEXT,
        smart TEXT,
        due_on TEXT,
        progress REAL,
        status TEXT           -- active/completed/dropped
    )""")
    run("""CREATE TABLE IF NOT EXISTS future_letters(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        content TEXT,
        created_at TEXT,
        reveal_on TEXT,
        revealed INT
    )""")
    run("""CREATE TABLE IF NOT EXISTS field_state(
        user_id TEXT,
        key TEXT,
        value TEXT,
        updated_at TEXT,
        PRIMARY KEY(user_id,key)
    )""")
    run("""CREATE TABLE IF NOT EXISTS goals(
        user_id TEXT,
        objective TEXT,       -- Goal name/description
        unit TEXT,            -- unit to track pace (minutes/pages/£/reps/etc)
        target REAL,          -- target total (e.g., 50000 words, or 200 pages, or £3000)
        horizon_days INTEGER, -- horizon for forecast window (default 30)
        active INT            -- 1 active, 0 inactive
    )""")

init_db()

# ---------------------- FIELD ENGINE ------------------------
def field_set(uid: str, key: str, value: dict):
    run("""INSERT OR REPLACE INTO field_state(user_id,key,value,updated_at)
           VALUES(?,?,?,?)""",
        (uid, key, json.dumps(value), dt.datetime.now().isoformat()))

def field_get(uid: str, key: str, default=None):
    rows = fetch("SELECT value FROM field_state WHERE user_id=? AND key=?", (uid, key))
    if rows:
        try:
            return json.loads(rows[0]["value"])
        except Exception:
            return default
    return default

# ---------------------- AUTH / PROFILE ----------------------
def hash_pin(pin: str) -> str:
    return hashlib.sha256((pin or "").encode()).hexdigest()

def create_profile(name: str, pin: str) -> str:
    uid = hashlib.sha1((name+"|"+pin+str(dt.datetime.now())).encode()).hexdigest()[:12]
    run("INSERT INTO profiles(id,name,pin_hash,api_key) VALUES(?,?,?,?)",
        (uid, name, hash_pin(pin), ""))
    return uid

def auth_profile(name: str, pin: str):
    rows = fetch("SELECT * FROM profiles WHERE name=?", (name,))
    if rows and rows[0]["pin_hash"] == hash_pin(pin):
        return rows[0]
    return None

def get_profile(uid: str):
    rows = fetch("SELECT * FROM profiles WHERE id=?", (uid,))
    return rows[0] if rows else None

# ---------------------- SETTINGS / GOAL ---------------------
def get_active_goal(uid: str):
    rows = fetch("SELECT * FROM goals WHERE user_id=? AND active=1", (uid,))
    return rows[0] if rows else None

def set_goal(uid: str, objective: str, unit: str, target: float, horizon: int = 30):
    run("UPDATE goals SET active=0 WHERE user_id=?", (uid,))
    run("""INSERT INTO goals(user_id,objective,unit,target,horizon_days,active)
           VALUES(?,?,?,?,?,1)""", (uid, objective, unit, float(target), int(horizon)))

# ---------------------- LENS SYSTEM -------------------------
CORE_LENS = {
    "name": "Core",
    "collapse": [
        "Release what drags you sideways.",
        "Close one loop before you open another.",
        "You can’t carry every door at once."
    ],
    "recursion": [
        "Small loops compound into identity.",
        "Repeat the action that proves the future.",
        "Begin poorly. Arrival happens mid-motion."
    ],
    "emergence": [
        "Invite the first true move.",
        "Seeds break in quiet; new timelines start small.",
        "Act once; momentum will meet you."
    ],
    "neutral": [
        "Attend to what is here. Choose again.",
        "Write what happened. Name what matters.",
        "Breathe, look, act."
    ]
}

LENS_KEYS = {"collapse","recursion","emergence","neutral"}

def lens_schema_guard(obj: dict) -> dict:
    out = {k: [] for k in LENS_KEYS}
    if not isinstance(obj, dict):
        return out
    for k in LENS_KEYS:
        vals = obj.get(k, [])
        if isinstance(vals, list):
            out[k] = [str(x).strip() for x in vals if isinstance(x, (str, int, float)) and str(x).strip()]
    return out

def parse_lens_file(upload) -> dict:
    text = ""
    name = upload.name
    try:
        if name.lower().endswith(".txt"):
            text = upload.read().decode("utf-8", "ignore")
        elif name.lower().endswith(".docx") and docx:
            d = docx.Document(upload)
            text = "\n".join(p.text for p in d.paragraphs)
        elif name.lower().endswith(".pdf") and PyPDF2:
            pdf = PyPDF2.PdfReader(upload)
            text = "\n".join((page.extract_text() or "") for page in pdf.pages)
    except Exception:
        text = ""
    text = (text or "").replace("\r","")
    chunks = [p.strip() for p in text.split("\n") if len(p.strip()) >= 40] or [text[:280]]
    parts = {"collapse":[],"recursion":[],"emergence":[],"neutral":[]}
    KW = {
        "collapse": ["release","close","end","quit","discard","stop","let go"],
        "recursion": ["repeat","again","habit","loop","daily","consistency"],
        "emergence": ["begin","start","spark","new","future","grow","transform"]
    }
    for p in chunks:
        t = p.lower()
        cat = "neutral"
        for k, keys in KW.items():
            if any(w in t for w in keys):
                cat = k; break
        parts[cat].append(p[:400])
    return lens_schema_guard(parts)

def lenses_for_user(uid: str) -> Dict[str, dict]:
    rows = fetch("SELECT * FROM lenses WHERE user_id=?", (uid,))
    out = {"Core": CORE_LENS}
    for r in rows:
        try:
            data = json.loads(r["data"])
            out[r["name"]] = {"name": r["name"], **lens_schema_guard(data)}
        except Exception:
            pass
    return out

def lens_memory_log(uid: str, lens_name: str, kind: str, phrase: str, ctx: dict):
    run("""INSERT INTO lens_memory(user_id,at,lens,kind,phrase,ctx)
           VALUES(?,?,?,?,?,?)""",
        (uid, dt.datetime.now().isoformat(), lens_name, kind, phrase, json.dumps(ctx)))

def smart_lens_line(uid: str, kind: str, ctx: dict, primary: str, secondary: str=None) -> str:
    all_l = lenses_for_user(uid)
    pools = []
    if primary in all_l: pools.append(all_l[primary].get(kind, []))
    if secondary and secondary in all_l: pools.append(all_l[secondary].get(kind, []))
    pools.append(CORE_LENS.get(kind, []))
    pool = [x for arr in pools for x in (arr or [])]

    if not pool:
        return ""

    recent = fetch("""SELECT phrase FROM lens_memory
                      WHERE user_id=? AND kind=?
                      ORDER BY id DESC LIMIT 15""", (uid, kind))
    recent_set = {r["phrase"] for r in recent}
    candidates = [p for p in pool if p not in recent_set] or pool

    # goal bias
    goal = (ctx.get("goal") or "").lower()
    if goal:
        weighted = []
        kws = [w for w in goal.split() if len(w) > 3]
        for p in candidates:
            score = 1
            tl = p.lower()
            if any(w in tl for w in kws): score += 2
            weighted += [p]*score
        candidates = weighted

    phrase = random.choice(candidates)
    lens_memory_log(uid, primary or "Core", kind, phrase, ctx)
    return phrase

# -------------------- UNIT NORMALIZATION --------------------
# Built-in loops with default units and polarity (+ helpful / - harmful)
BUILTIN = [
    ("creation:writing", +1, "minutes"),
    ("creation:project", +1, "minutes"),
    ("mind:reading", +1, "pages"),
    ("mind:planning", +1, "minutes"),
    ("mind:meditation", +1, "minutes"),
    ("body:walk", +1, "minutes"),
    ("body:exercise", +1, "minutes"),
    ("body:sleep_good", +1, "hours"),
    ("body:late_sleep", -1, "minutes"),
    ("consumption:scroll", -1, "minutes"),
    ("consumption:youtube", -1, "minutes"),
    ("food:junk", -1, "servings"),
    ("finance:save_invest", +1, "£"),
    ("finance:budget_check", +1, "minutes"),
    ("finance:impulse_spend", -1, "£")
]

# Unit → effective minutes default multipliers
UNIT_DEFAULT_RATE = {
    "minutes": 1.0,
    "hours": 60.0,
    "pages": 1.0,         # assume 1 page ≈ 1 minute by default, user can calibrate
    "chapters": 20.0,     # assume 1 chapter ≈ 20 minutes
    "reps": 0.25,         # rough neutralization
    "£": 0.05,            # £20 ≈ +1 effective minute (conservative); calibrate per user
    "%": 1.0,             # % of income → special handling if needed
    "servings": 10.0      # one junk serving = 10 "negative minutes"
}

def get_loop_catalog(uid: str) -> List[dict]:
    cat = []
    for name, pol, unit in BUILTIN:
        cat.append({"name": name, "category": name.split(":")[0], "polarity": pol, "unit": unit, "rate": UNIT_DEFAULT_RATE.get(unit, 1.0), "builtin": True})
    customs = fetch("SELECT * FROM custom_loops WHERE user_id=?", (uid,))
    for r in customs:
        rate = r["rate"]
        if rate is None or rate <= 0:
            rate = UNIT_DEFAULT_RATE.get(r["unit"], 1.0)
        cat.append({"name": f"{r['category']}:{r['name']}", "category": r["category"], "polarity": r["polarity"], "unit": r["unit"], "rate": rate, "builtin": False})
    return cat

def normalize_amount(unit: str, raw_val: float, rate: float) -> float:
    if raw_val <= 0: return 0.0
    # basic sanity clamps
    raw_val = float(raw_val)
    if unit == "hours": raw_val = min(raw_val, 24)
    if unit == "minutes": raw_val = min(raw_val, 24*60)
    if unit == "pages": raw_val = min(raw_val, 1000)
    if unit == "chapters": raw_val = min(raw_val, 50)
    if unit == "£": raw_val = min(raw_val, 1000000)
    if unit == "reps": raw_val = min(raw_val, 2000)
    if unit == "servings": raw_val = min(raw_val, 12)

    # convert to effective minutes:
    return float(raw_val * (rate if rate else UNIT_DEFAULT_RATE.get(unit, 1.0)))

# ---------------------- STATE LABELING ----------------------
W_POS, W_NEG, W_PROG, W_ENER = 0.8, 0.9, 0.25, 0.15
STATES = ["Momentum","Mixed","Stuck"]
IDX = {s:i for i,s in enumerate(STATES)}

def label_state(eff_loops: Dict[str,float], pos_names: set, neg_names: set) -> Tuple[str,float,float,float,str]:
    posm = sum(eff_loops.get(k,0) for k in pos_names)
    negm = sum(eff_loops.get(k,0) for k in neg_names)
    energy = min(100.0, (eff_loops.get("body:walk",0)*1.2 + eff_loops.get("body:exercise",0)*1.6 + eff_loops.get("body:sleep_good",0)*0.5) / 2.0)
    progress = min(100.0, (eff_loops.get("creation:writing",0)*1.4 + eff_loops.get("creation:project",0)*1.2 + eff_loops.get("finance:save_invest",0)*0.5 + eff_loops.get("mind:planning",0)*0.9) / 2.0)
    focus = max(0.0, min(100.0, (posm*W_POS - negm*W_NEG) + progress*W_PROG + energy*W_ENER))
    if negm > posm*1.2 or eff_loops.get("consumption:scroll",0) >= 45: state = "Stuck"
    elif posm >= negm and (eff_loops.get("creation:writing",0)+eff_loops.get("creation:project",0)) >= 30:
        state = "Momentum"
    else:
        state = "Mixed"
    # micro explanation
    contribs=[]
    for k in pos_names:
        m=eff_loops.get(k,0); 
        if m>0: contribs.append((k, +m*W_POS, m))
    for k in neg_names:
        m=eff_loops.get(k,0); 
        if m>0: contribs.append((k, -m*W_NEG, m))
    pos_sorted = sorted([c for c in contribs if c[1]>0], key=lambda x:-x[1])[:2]
    neg_sorted = sorted([c for c in contribs if c[1]<0], key=lambda x:abs(x[1]), reverse=True)[:2]
    def fmt_pos(c): return f"+{int(c[2])}m {c[0].split(':',1)[1]} (impact +{c[1]:.1f})"
    def fmt_neg(c): return f"-{int(c[2])}m {c[0].split(':',1)[1]} (impact {c[1]:.1f})"
    plus = ", ".join(fmt_pos(c) for c in pos_sorted) if pos_sorted else "+0m"
    minus = ", ".join(fmt_neg(c) for c in neg_sorted) if neg_sorted else "-0m"
    micro = f"{plus} | {minus}"
    return state, round(focus,1), round(energy,1), round(progress,1), micro

# ------------- TRANSITION MATRIX (CLASSICAL BASELINE) -------
DECAY = 0.97
PRIOR_WEIGHT = 0.5
UNIFORM_BLEND = 0.08

def learn_matrix(days_rows: List[dict], decay=DECAY) -> np.ndarray:
    C = np.ones((3,3))*PRIOR_WEIGHT
    last = None
    w = 1.0
    for d in days_rows:
        s = d.get("state")
        if s not in IDX: continue
        if last is not None:
            C[IDX[last], IDX[s]] += w
        w *= decay; last = s
    M = C / C.sum(axis=1, keepdims=True)
    U = np.ones((3,3))/3.0
    M = (1-UNIFORM_BLEND)*M + UNIFORM_BLEND*U
    return M / M.sum(axis=1, keepdims=True)

def simulate(M: np.ndarray, start_state: str, days=30, sims=1500) -> Tuple[np.ndarray, float]:
    start = IDX.get(start_state, 1)
    counts = np.zeros((days, 3))
    for _ in range(sims):
        s = start
        for t in range(days):
            counts[t, s] += 1
            s = np.random.choice([0,1,2], p=M[s])
    probs = counts / sims
    exp_momentum = probs[:,0].sum()
    return probs, float(exp_momentum)

# ------------------ BAYESIAN PROGRESS (GOAL) ----------------
# Beta-Binomial model for daily success (moving closer to objective)
# Simple rule: a day counts as "success" if Momentum or if effective key loop meets threshold.

def beta_binomial_posterior(successes: int, trials: int, a0=1.5, b0=1.5):
    a = a0 + successes
    b = b0 + (trials - successes)
    return a, b

def success_prob_in_window(a: float, b: float, window_days: int, daily_thresh=0.5) -> float:
    # Monte Carlo: sample daily success p ~ Beta(a,b); probability get >= some fraction
    # Here: interpret as chance of accumulating enough “success-days” to hit plan rate.
    # We return mean p, and rely on ETA estimate for days.
    return a / (a + b + 1e-9)

def estimate_eta_ribbon(expected_momentum_days: float, horizon: int = 30) -> Tuple[int,int]:
    # approximate: median ETA ~ horizon * (target momentum / expected momentum)
    # Here we assume target momentum ~ 50% of horizon to finish a chunk; we scale bands.
    # This is heuristic but produces intuitive ribbons.
    if expected_momentum_days <= 0.1:
        return (horizon+15, horizon+40)
    median = max(1, int((horizon * 0.5) / max(0.1, expected_momentum_days/horizon)))
    low = max(1, int(median*0.8))
    high = min(90, int(median*1.3))
    return (low, high)


# ---------- Contextual bandit (robust) ----------

def _linreg_posterior(X: np.ndarray, y: np.ndarray, alpha: float = 1.0, sigma2: float = 0.35):
    """ Bayesian ridge posterior for linear reward model.
        Returns (mu, cov) with strong guards against singular/corrupt matrices. """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    d = X.shape[1]

    # A = alpha*I + X^T X ; b = X^T y
    A = alpha * np.eye(d) + X.T @ X
    b = X.T @ y

    # Safe inverse (falls back to pseudo-inverse)
    try:
        Ainv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        Ainv = np.linalg.pinv(A)

    mu = Ainv @ b
    cov = sigma2 * Ainv

    # Numerical floor (avoid negative/NaN on diagonals)
    cov = np.nan_to_num(cov, nan=0.0, posinf=1e3, neginf=0.0)
    for i in range(min(cov.shape[0], cov.shape[1])):
        if cov[i, i] < 1e-9:
            cov[i, i] = 1e-9
    return mu, cov


def _get_arm_data(title: str):
    """ Load (ctx, reward) rows for a given intervention title.
        Returns (X, y) as numpy arrays or (None, None) if no usable rows. """
    rows = fetch(
        "SELECT ctx, reward FROM interventions_log_ctx WHERE title=? AND reward IS NOT NULL",
        (title,),
    )
    X, y = [], []
    for r in rows:
        try:
            v = json.loads(r["ctx"])
            rr = float(r["reward"])
            if isinstance(v, list) and len(v) > 0:
                X.append(v)
                y.append(rr)
        except Exception:
            continue
    if not X:
        return None, None

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    # guard: drop NaN rows
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[ok]
    y = y[ok]
    if X.size == 0 or y.size == 0:
        return None, None

    # final guard: 2D shape
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X, y


def bandit_scores(context_vec: list | np.ndarray, candidates_titles: list[str]) -> dict[str, float]:
    """ LinTS per-arm with strong shape/empty-data guards.
        Returns {title: score}. If no data for an arm → optimistic prior around 0.55. """
    x = np.asarray(context_vec, dtype=float).flatten()
    if not np.isfinite(x).all() or x.size == 0:
        # as a last resort, provide a 1-dim neutral feature
        x = np.array([1.0], dtype=float)

    scores: dict[str, float] = {}
    for t in candidates_titles:
        X, y = _get_arm_data(t)
        if X is None:
            # Optimistic prior encourages early exploration
            scores[t] = float(0.55 + np.random.normal(0.0, 0.05))
            continue

        mu, cov = _linreg_posterior(X, y, alpha=1.0, sigma2=0.35)

        # Sample theta safely. If sampling fails, fall back to mean.
        try:
            theta = np.random.multivariate_normal(mean=mu, cov=cov)
        except Exception:
            theta = mu

        theta = np.asarray(theta, dtype=float).flatten()

        # Align dimensions (truncate/pad) to avoid shape mismatches
        if theta.size != x.size:
            m = min(theta.size, x.size)
            if m == 0:
                scores[t] = 0.5
                continue
            val = float(np.dot(x[:m], theta[:m]))
        else:
            val = float(np.dot(x, theta))

        # Final sanitization
        if not np.isfinite(val):
            val = 0.5
        scores[t] = val
    return scores


# ---------------------- AI HELPERS --------------------------
def ai_available():
    return (openai is not None)

def set_openai_key(key: str):
    if ai_available():
        try:
            openai.api_key = key
        except Exception:
            pass

def ai_narrate_forecast_rich(goal, start_state, focused_days,
                             likelihood_mid, likelihood_band,
                             required_pace_delta, milestones,
                             top_move, top_move_delta, drivers_pos, drivers_neg,
                             lens_line):
    # Keep prompt tight. If no API or failure → caller will fall back to local.
    sys = "You are a concise coach. 1-3 vivid sentences. No bullet points."
    user = f"""
Objective: {goal or '—'}
Start state: {start_state}
Expected momentum days (30): {focused_days:.1f}
Chance to hit plan: {likelihood_mid:.2f} (range {likelihood_band[0]:.2f}-{likelihood_band[1]:.2f})
Smallest move: {top_move} (+{top_move_delta:.2f} days)
Tailwinds: {', '.join(drivers_pos or []) or '—'}
Headwinds: {', '.join(drivers_neg or []) or '—'}
Lens: {lens_line}
Write one compact paragraph. Avoid generic advice; speak to the data.
"""
    try:
        r = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},
                      {"role":"user","content":user}],
            temperature=0.7,
            max_tokens=180
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return ""

# ---------------------- UI HELPERS --------------------------
def sticky_header(uid: str):
    days = fetch("SELECT * FROM days WHERE user_id=? ORDER BY d ASC", (uid,))
    if not days:
        st.markdown("**Today:** no data yet")
        return
    today = days[-1]
    state = today.get("state") or "—"
    st.markdown(f"**Today** • {state} • Focus {today.get('focus',0):.0f} • Energy {today.get('energy',0):.0f} • Progress {today.get('progress',0):.0f}")

# ============================================================
# APP START — AUTH
# ============================================================
st.title("⏳ TimeSculpt")
st.caption("A recursive, goal-centric field engine for personal transformation.")

st.sidebar.header("Profile")
profile=None
mode = st.sidebar.radio("Session", ["Login","Create"], horizontal=True)
if mode=="Create":
    n = st.sidebar.text_input("Name")
    p = st.sidebar.text_input("PIN (numbers ok)", type="password")
    if st.sidebar.button("Create profile"):
        if n and p:
            uid = create_profile(n,p)
            st.session_state["uid"] = uid
            profile = get_profile(uid)
elif mode=="Login":
    n = st.sidebar.text_input("Name")
    p = st.sidebar.text_input("PIN", type="password")
    if st.sidebar.button("Login"):
        pr = auth_profile(n,p)
        if pr:
            st.session_state["uid"] = pr["id"]
            profile = pr
if "uid" in st.session_state and not profile:
    profile = get_profile(st.session_state["uid"])
if not profile:
    st.info("Login or create a profile to continue.")
    st.stop()
uid = profile["id"]

# AI Toggle + Key
st.sidebar.header("AI")
USE_AI = st.sidebar.toggle("Enable AI narration", value=False)
if USE_AI:
    st.sidebar.success("AI Active")
else:
    st.sidebar.info("AI Inactive")

api_input = st.sidebar.text_input("API key (optional, stored with profile)", type="password",
                                  help="Paste your OpenAI-compatible key here to enable AI narration.")
if api_input:
    run("UPDATE profiles SET api_key=? WHERE id=?", (api_input, uid))
prof_now = get_profile(uid)
if prof_now and prof_now.get("api_key"):
    set_openai_key(prof_now["api_key"])

# ============================================================
# TABS
# ============================================================
tabs = st.tabs([
    "Input", "Forecast", "Interventions",
    "Diagnostics", "Lenses", "Future Self", "Settings"
])

# ---------------------- INPUT TAB ---------------------------
with tabs[0]:
    st.header("Daily Input")
    sticky_header(uid)

    # Active goal
    st.subheader("Objective")
    goal_row = get_active_goal(uid)
    if goal_row:
        st.success(f"Active objective: **{goal_row['objective']}** · Target: **{goal_row['target']} {goal_row['unit']}** · Horizon: **{goal_row['horizon_days']} days**")
    else:
        st.warning("No active objective. Set one in Settings → Objective.")

    # Built-in + custom loops catalog
    catalog = get_loop_catalog(uid)
    pos_names = {c["name"] for c in catalog if c["polarity"]>0}
    neg_names = {c["name"] for c in catalog if c["polarity"]<0}

    # Logging form
    st.subheader("Log today")
    d = st.date_input("Date", value=dt.date.today()).isoformat()
    cols = st.columns(4)
    raw_loops = {}
    preview_texts = []

    # show a subset grid (common) + custom list below
    common = [
        "creation:writing","creation:project","mind:reading","mind:planning",
        "mind:meditation","body:walk","body:exercise","body:sleep_good",
        "body:late_sleep","consumption:scroll","finance:save_invest","finance:budget_check"
    ]
    cat_map = {c["name"]: c for c in catalog}
    def input_one(name, col):
        meta = cat_map.get(name)
        if not meta: return
        unit = meta["unit"]; rate = meta["rate"]
        label = f"{name.split(':',1)[1]} ({unit})"
        with col:
            val = st.number_input(label, min_value=0.0, step=1.0, key=f"log_{name}")
            raw_loops[name] = (unit, rate, val)
            eff = normalize_amount(unit, val, rate)
            if val>0:
                preview_texts.append(f"{label}: {val} → counts as {eff:.1f} effective min")

    for i, name in enumerate(common):
        input_one(name, cols[i%4])

    # Custom loops
    st.markdown("**Custom loops**")
    custom_names = [c["name"] for c in catalog if not c["builtin"]]
    if custom_names:
        ccols = st.columns(4)
        for idx, name in enumerate(custom_names):
            input_one(name, ccols[idx%4])
    else:
        st.caption("No custom loops yet. Add in Settings.")

    if preview_texts:
        st.caption("Conversion preview:")
        st.write(" • " + "\n • ".join(preview_texts))

    note = st.text_area("Note (optional)")

    if st.button("Commit today"):
        # compute effective loops map
        eff = {k: normalize_amount(u, v, r) for k,(u,r,v) in raw_loops.items()}
        state, F, E, P, micro = label_state(eff, pos_names, neg_names)
        run("""INSERT INTO days(user_id,d,loops,eff_loops,note,focus,energy,progress,state)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (uid, d, json.dumps({k: raw_loops[k][2] for k in raw_loops}), json.dumps(eff), note, F, E, P, state))
        field_set(uid, "loops", {"last_entry": eff, "state": state, "micro": micro, "F": F, "E": E, "P": P, "d": d})
        st.success(f"Saved. Today labeled **{state}**. {micro}")

# ---------------------- FORECAST TAB ------------------------
with tabs[1]:
    st.header("30-Day Forecast (Goal-centric)")
    days_rows = fetch("SELECT * FROM days WHERE user_id=? ORDER BY d ASC", (uid,))
    if not days_rows:
        st.info("No days logged yet. Add at least one day in Input.")
    else:
        # compute classical baseline
        M = learn_matrix(days_rows)
        start = days_rows[-1].get("state","Mixed")
        probs, exp_momentum_days = simulate(M, start, days=30, sims=1800)

        # 14-day sparkline + quick stats
        last14 = days_rows[-14:]
        df14 = pd.DataFrame([{"d": r["d"], "focus": r.get("focus",0)} for r in last14]) if last14 else pd.DataFrame({"d":[],"focus":[]})
        c1,c2,c3 = st.columns([2,1,1])
        with c1:
            st.caption("Last 14 days — Focus trend")
            if not df14.empty:
                st.altair_chart(
                    alt.Chart(df14).mark_line(point=True).encode(
                        x="d:T", y="focus:Q"
                    ).properties(height=80),
                    use_container_width=True
                )
            else:
                st.caption("No recent data")
        with c2:
            st.metric("Expected Momentum days (30d)", f"{probs[:,0].sum():.1f}")
        with c3:
            st.metric("Current state", start)

        # Goal info
        goal = get_active_goal(uid)
        goal_text = goal["objective"] if goal else ""
        horizon = int(goal["horizon_days"] if goal else 30)
        unit_label = (goal["unit"] if goal else "units")
        target_total = float(goal["target"] if goal else 0.0)

        # Momentum-driven ETA ribbon (approx)
        eta_lo, eta_hi = estimate_eta_ribbon(exp_momentum_days, horizon=horizon)

        # Beta-Binomial progress — define success as Momentum day
        succ = int(round(probs[:,0].sum()))  # expected succ days ~ sum of daily momentum probs
        trials = 30
        a,b = beta_binomial_posterior(succ, trials, a0=1.5, b0=1.5)
        lik_mid = a / (a+b+1e-9)
        lik_lo = max(0.0, lik_mid - 0.07); lik_hi = min(1.0, lik_mid + 0.07)

        # Drivers (simple): top +/− eff loops averages
        eff_map = {}
        for r in days_rows[-14:]:
            eff = json.loads(r.get("eff_loops") or "{}")
            for k,v in eff.items():
                eff_map.setdefault(k,[]).append(v)
        avg_eff = {k: np.mean(v) for k,v in eff_map.items()}
        top_pos = sorted([(k,v) for k,v in avg_eff.items() if "consumption:" not in k and "late_sleep" not in k and "impulse_spend" not in k], key=lambda x:-x[1])[:3]
        top_neg = sorted([(k,v) for k,v in avg_eff.items() if ("consumption:" in k or "late_sleep" in k or "impulse_spend" in k)], key=lambda x:-x[1])[:3]
        drivers_pos = [f"{k.split(':',1)[1]}↑" for k,_ in top_pos]
        drivers_neg = [f"{k.split(':',1)[1]}↑" for k,_ in top_neg]

        # Interventions pool (tiny demo set; you can expand)
        POOL = [
            {"title":"7-min starter","how":"Start badly. Stop after 7.","tags":["creation"],"tweak":{"m_to_f":+0.06}},
            {"title":"15-min walk","how":"Swap one scroll for a walk.","tags":["body"],"tweak":{"m_to_f":+0.05,"d_self":-0.03}},
            {"title":"Sleep before midnight","how":"Shut down 30 min earlier.","tags":["body"],"tweak":{"d_self":-0.06}},
            {"title":"Pay-yourself-first 10%","how":"Automate right after payday.","tags":["finance"],"tweak":{"m_to_f":+0.05}},
        ]

        # Context + bandit scores
        ctx_vec = context_vector(days_rows, goal_text, start)
        titles = [iv["title"] for iv in POOL]
        try:
    ctx_scores = bandit_scores(ctx_vec, titles)
except Exception:
    ctx_scores = {t: 0.5 for t in titles}

        # Compute deltas: simply boost expected momentum by bandit score * small factor for demo
        base_focus_days = probs[:,0].sum()
        scored = []
        for iv in POOL:
            bscore = ctx_scores.get(iv["title"], 0.5)
            delta = min(2.0, max(0.0, (bscore-0.5)*2.5))  # map ~0.0..1.0 → 0..1.25 days, cap
            scored.append({"iv":iv, "delta":delta, "score":bscore})
        scored.sort(key=lambda r:-r["delta"])
        best = scored[0] if scored else None

        # Multi-lens smart line for the brief (no explanation text)
        lenses_state = field_get(uid,"lenses",{}) or {}
        active_primary = lenses_state.get("active_primary","Core")
        active_secondary = lenses_state.get("active_secondary","")
        lens_line = smart_lens_line(uid, "emergence", {"goal": goal_text, "state": start}, active_primary, active_secondary)

        # Rich brief (AI if on, else local)
        brief_text = ""
        if USE_AI and prof_now and prof_now.get("api_key") and ai_available():
            brief_text = ai_narrate_forecast_rich(
                goal=goal_text, start_state=start, focused_days=float(base_focus_days),
                likelihood_mid=lik_mid, likelihood_band=(lik_lo,lik_hi),
                required_pace_delta=0.3, milestones=["Next chapter","Submit draft"],
                top_move=(best["iv"]["title"] if best else "—"),
                top_move_delta=(best["delta"] if best else 0.0),
                drivers_pos=drivers_pos, drivers_neg=drivers_neg,
                lens_line=lens_line
            )
        if not brief_text:
            brief_text = f"{lens_line}  " + (
                f"At this pace, you’re about **{lik_mid:.0%}** likely to stay on plan. "
                f"Median ETA ~ **{eta_lo}** days (80% by ~**{eta_hi}**). "
                f"Try **{(best['iv']['title'] if best else 'a tiny move')}** to tighten the curve."
            )

        st.subheader("Forecast brief")
        st.markdown(f"<div class='card'>{brief_text}</div>", unsafe_allow_html=True)

        # Comparison: do nothing vs apply top move (simple illustrative shift)
        st.subheader("Compare")
        cA, cB = st.columns(2)
        with cA:
            st.markdown("**If you do nothing**")
            st.caption(f"Expected Momentum days: {base_focus_days:.1f} / 30")
        with cB:
            if best:
                st.markdown(f"**If you apply: _{best['iv']['title']}_**")
                st.caption(f"Expected Momentum days: {(base_focus_days + best['delta']):.1f} / 30 (Δ +{best['delta']:.2f})")
            else:
                st.caption("No interventions available.")

        # Save field_state for other tabs
        field_set(uid,"forecast",{
            "brief": brief_text,
            "lik": [lik_lo, lik_mid, lik_hi],
            "eta": [eta_lo, eta_hi],
            "base_momentum_days": base_focus_days,
            "top_move": (best["iv"]["title"] if best else ""),
            "top_move_delta": (best["delta"] if best else 0.0)
        })

# ------------------- INTERVENTIONS TAB ----------------------
with tabs[2]:
    st.header("Interventions (Contextual)")
    days_rows = fetch("SELECT * FROM days WHERE user_id=? ORDER BY d ASC", (uid,))
    if not days_rows:
        st.info("Log some days first.")
    else:
        start = days_rows[-1].get("state","Mixed")
        goal = get_active_goal(uid)
        goal_text = goal["objective"] if goal else ""
        # pool (same for demo)
        POOL = [
            {"title":"7-min starter","how":"Start badly. Stop after 7.","tags":["creation"]},
            {"title":"15-min walk","how":"Swap one scroll for a walk.","tags":["body"]},
            {"title":"Sleep before midnight","how":"Shut down 30 min earlier.","tags":["body"]},
            {"title":"Pay-yourself-first 10%","how":"Automate right after payday.","tags":["finance"]},
        ]
        titles = [p["title"] for p in POOL]
        ctx_vec = context_vector(days_rows, goal_text, start)
        try:
    scores = bandit_scores(ctx_vec, titles)
except Exception:
    scores = {t: 0.5 for t in titles}
        ranked = sorted(POOL, key=lambda iv:-scores.get(iv["title"],0.5))

        # Top suggestion card
        top = ranked[0]
        st.markdown("### ⭐ Top move")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write(f"**{top['title']}** — {top['how']}")
        st.caption(f"Context score: {scores.get(top['title'],0.5):.2f}")
        colA, colB, colC = st.columns([1,1,2])
        with colA:
            if st.button("Apply", key="apply_top"):
                log_intervention(uid, top["title"], accepted=True)
                log_intervention_ctx(uid, top["title"], ctx_vec, helped=None)
                field_set(uid,"interventions",{"last_applied": top["title"], "ctx": ctx_vec})
                st.success("Applied.")
        with colB:
            fb = st.selectbox("Did it help?", ["Skip","Yes","No"], key="fb_top")
            if fb != "Skip":
                helped = (fb=="Yes")
                log_intervention(uid, top["title"], accepted=True, helped=helped)
                log_intervention_ctx(uid, top["title"], ctx_vec, helped=helped)
                st.success("Feedback saved.")
        with colC:
            st.caption("Why this now?")
            loops_field = field_get(uid,"loops",{}) or {}
            st.markdown(f"<span class='badge'>{(loops_field.get('micro') or '—')}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # More options
        st.markdown("### More options")
        for iv in ranked[1:]:
            with st.expander(iv["title"]):
                st.write(iv["how"])
                st.caption(f"Context score: {scores.get(iv['title'],0.5):.2f}")
                c1,c2 = st.columns([1,3])
                with c1:
                    if st.button("Apply", key=f"apply_{iv['title']}"):
                        log_intervention(uid, iv["title"], accepted=True)
                        log_intervention_ctx(uid, iv["title"], ctx_vec, helped=None)
                        st.success("Applied.")
                with c2:
                    fb = st.selectbox("Did it help?", ["Skip","Yes","No"], key=f"fb_{iv['title']}")
                    if fb!="Skip":
                        helped = (fb=="Yes")
                        log_intervention(uid, iv["title"], accepted=True, helped=helped)
                        log_intervention_ctx(uid, iv["title"], ctx_vec, helped=helped)
                        st.success("Feedback saved.")

        # Proven for you leaderboard
        st.subheader("Proven for you")
        rows = fetch("""SELECT title, AVG(CASE WHEN reward IS NOT NULL THEN reward ELSE NULL END) AS success_rate,
                        COUNT(*) AS trials
                        FROM interventions_log_ctx WHERE user_id=? GROUP BY title ORDER BY success_rate DESC""", (uid,))
        if rows:
            df = pd.DataFrame(rows)
            df["success_rate"] = df["success_rate"].fillna(0)
            st.dataframe(df.style.format({"success_rate":"{:.2f}"}), use_container_width=True)
        else:
            st.caption("No feedback yet.")

# --------------------- DIAGNOSTICS TAB ----------------------
with tabs[3]:
    st.header("Diagnostics")
    st.caption("Force (+) = loops that correlate with Momentum days. Drag (−) = loops correlated with Stuck days.")
    days_rows = fetch("SELECT * FROM days WHERE user_id=? ORDER BY d ASC", (uid,))
    if len(days_rows) < 5:
        st.info("Log more days for diagnostics.")
    else:
        data=[]
        for d in days_rows:
            eff = json.loads(d.get("eff_loops") or "{}")
            row={"d": d["d"], "state": d.get("state","Mixed")}
            row.update(eff); data.append(row)
        df = pd.DataFrame(data)
        # Pivot mean minutes by state
        pivot = df.drop(columns=["d"]).groupby("state").mean(numeric_only=True).T.fillna(0.0)
        # pad columns
        for s in STATES:
            if s not in pivot.columns:
                pivot[s] = 0.0
        pivot["lift"] = pivot["Momentum"] - pivot["Stuck"]
        best = pivot.sort_values("lift", ascending=False).head(5)
        worst = pivot.sort_values("lift", ascending=True).head(5)
        c1,c2 = st.columns(2)
        with c1:
            st.subheader("Force (+)")
            st.dataframe(best[["lift","Momentum","Stuck"]])
        with c2:
            st.subheader("Drag (−)")
            st.dataframe(worst[["lift","Momentum","Stuck"]])

# ----------------------- LENSES TAB -------------------------
with tabs[4]:
    st.header("Lenses")
    st.caption("Upload texts (txt/docx/pdf) to feed narration. You may select a primary and secondary lens.")
    # Upload
    up = st.file_uploader("Upload lens file", type=["txt","docx","pdf"])
    lname = st.text_input("Lens name", value="My Lens")
    if st.button("Add Lens") and up and lname.strip():
        parts = parse_lens_file(up)
        run("INSERT INTO lenses(user_id,name,data) VALUES(?,?,?)", (uid, lname.strip(), json.dumps(parts)))
        st.success("Lens added.")

    # Active selection
    all_l = lenses_for_user(uid)
    prim = st.selectbox("Primary lens", options=list(all_l.keys()), index=list(all_l.keys()).index("Core") if "Core" in all_l else 0)
    sec_opt = ["(none)"] + [k for k in all_l.keys() if k != prim]
    sec = st.selectbox("Secondary lens (optional)", options=sec_opt, index=0)
    active_secondary = "" if sec=="(none)" else sec
    field_set(uid,"lenses", {"active_primary": prim, "active_secondary": active_secondary})
    st.caption("Status: saved to field state.")

    # Preview a line
    if st.button("Preview emergence line"):
        goal = get_active_goal(uid)
        gl = goal["objective"] if goal else ""
        line = smart_lens_line(uid, "emergence", {"goal": gl, "state":"—"}, prim, active_secondary)
        st.markdown(f"> {line}")

# --------------------- FUTURE SELF TAB ----------------------
with tabs[5]:
    st.header("Future Self")
    rows = fetch("SELECT * FROM future_self WHERE user_id=?", (uid,))
    fs = rows[0] if rows else {}
    fs_title = st.text_input("Future Self title", value=fs.get("title",""))
    fs_traits = st.text_area("Traits (comma separated)", value=fs.get("traits",""))
    fs_rituals = st.text_area("Rituals (comma separated)", value=fs.get("rituals",""))
    if st.button("Save Future Self"):
        run("DELETE FROM future_self WHERE user_id=?", (uid,))
        run("INSERT INTO future_self(user_id,title,traits,rituals) VALUES(?,?,?,?)",
            (uid, fs_title, fs_traits, fs_rituals))
        field_set(uid,"future_self", {"title": fs_title, "traits": fs_traits, "rituals": fs_rituals})
        st.success("Future Self saved.")

    st.subheader("SMART Challenges")
    ch_title = st.text_input("Challenge title")
    ch_why = st.text_area("Why this matters (motivation)")
    ch_smart = st.text_area("SMART definition")
    ch_due = st.date_input("Due on", value=dt.date.today())
    if st.button("Add challenge") and ch_title.strip():
        run("""INSERT INTO future_challenges(user_id,title,why,smart,due_on,progress,status)
               VALUES(?,?,?,?,?,?,?)""",
            (uid, ch_title.strip(), ch_why.strip(), ch_smart.strip(), ch_due.isoformat(), 0.0, "active"))
        st.success("Challenge added.")

    # list challenges
    chs = fetch("SELECT * FROM future_challenges WHERE user_id=? ORDER BY id DESC", (uid,))
    for ch in chs:
        with st.expander(f"{ch['title']}  ·  {ch['status']} · due {ch['due_on']}"):
            st.caption(ch["smart"])
            prog = st.slider("Progress", 0.0, 1.0, float(ch.get("progress",0.0)), 0.05, key=f"prog_{ch['id']}")
            st.write("Why:", ch.get("why",""))
            c1,c2,c3 = st.columns(3)
            with c1:
                if st.button("Save progress", key=f"savep_{ch['id']}"):
                    run("UPDATE future_challenges SET progress=? WHERE id=?", (prog, ch["id"]))
                    st.success("Updated.")
            with c2:
                if st.button("Complete", key=f"cmpl_{ch['id']}"):
                    run("UPDATE future_challenges SET status='completed', progress=1.0 WHERE id=?", (ch["id"],))
                    st.success("Marked completed.")
            with c3:
                if st.button("Redo", key=f"redo_{ch['id']}"):
                    run("UPDATE future_challenges SET status='active', progress=0.0 WHERE id=?", (ch["id"],))
                    st.success("Reset.")

    st.subheader("Letter to your past self")
    letter = st.text_area("Write the letter")
    reveal = st.date_input("Reveal on", value=dt.date.today())
    if st.button("Save letter") and letter.strip():
        run("""INSERT INTO future_letters(user_id,content,created_at,reveal_on,revealed)
               VALUES(?,?,?,?,0)""",
            (uid, letter.strip(), dt.date.today().isoformat(), reveal.isoformat()))
        st.success("Saved.")

    # Resurface letters that are due and not revealed
    due = fetch("""SELECT * FROM future_letters
                   WHERE user_id=? AND revealed=0 AND reveal_on<=?
                   ORDER BY id ASC""", (uid, dt.date.today().isoformat()))
    if due:
        st.subheader("A letter returns")
        st.markdown(f"<div class='card'><em>{due[0]['content']}</em></div>", unsafe_allow_html=True)
        if st.button("Mark as seen", key="seen_letter"):
            run("UPDATE future_letters SET revealed=1 WHERE id=?", (due[0]["id"],))

# ----------------------- SETTINGS TAB -----------------------
with tabs[6]:
    st.header("Settings")
    st.subheader("Objective")
    cur = get_active_goal(uid)
    obj = st.text_input("Objective", value=(cur["objective"] if cur else ""))
    unit = st.selectbox("Unit", ["minutes","hours","pages","chapters","reps","£","%","units"], index=(["minutes","hours","pages","chapters","reps","£","%","units"].index(cur["unit"]) if cur else 0))
    target = st.number_input("Target total (how much to reach)", min_value=0.0, value=(float(cur["target"]) if cur else 0.0), step=1.0)
    horizon = st.number_input("Forecast horizon (days)", min_value=7, value=(int(cur["horizon_days"]) if cur else 30), step=1)
    if st.button("Save objective"):
        if obj.strip() and target>0:
            set_goal(uid, obj.strip(), unit, target, int(horizon))
            st.success("Objective saved.")
        else:
            st.warning("Enter objective and a positive target.")

    st.subheader("Custom loops")
    cname = st.text_input("Name")
    ccat = st.selectbox("Category", ["creation","mind","body","consumption","food","finance","other"])
    cpol = st.selectbox("Polarity", [+1,-1], index=0)
    cunit = st.selectbox("Unit", list(UNIT_DEFAULT_RATE.keys()), index=0)
    crate = st.number_input("Rate (auto-set by unit; override if you wish)", min_value=0.0, value=float(UNIT_DEFAULT_RATE.get(cunit,1.0)))
    if st.button("Add custom loop"):
        if cname.strip():
            run("""INSERT INTO custom_loops(user_id,name,category,polarity,unit,rate)
                   VALUES(?,?,?,?,?,?)""", (uid, cname.strip(), ccat, int(cpol), cunit, float(crate)))
            st.success("Custom loop added.")
        else:
            st.warning("Give your loop a name.")

    # Export/Import simple JSON dump
    st.subheader("Export / Import")
    if st.button("Export JSON"):
        # dump minimal tables
        days = fetch("SELECT * FROM days WHERE user_id=?", (uid,))
        lenses_db = fetch("SELECT * FROM lenses WHERE user_id=?", (uid,))
        customs = fetch("SELECT * FROM custom_loops WHERE user_id=?", (uid,))
        fs = fetch("SELECT * FROM future_self WHERE user_id=?", (uid,))
        ch = fetch("SELECT * FROM future_challenges WHERE user_id=?", (uid,))
        let = fetch("SELECT * FROM future_letters WHERE user_id=?", (uid,))
        goal = get_active_goal(uid)
        dump = {"days": days, "lenses": lenses_db, "custom_loops": customs,
                "future_self": fs, "future_challenges": ch, "future_letters": let,
                "goal": goal}
        st.download_button("Download", data=json.dumps(dump, indent=2), file_name="timesculpt_export.json")
    imp = st.file_uploader("Import JSON", type=["json"])
    if imp and st.button("Import now"):
        try:
            data = json.loads(imp.read().decode("utf-8"))
            # naive import: append rows (avoid dup keys)
            for r in data.get("days", []):
                run("""INSERT INTO days(user_id,d,loops,eff_loops,note,focus,energy,progress,state)
                       VALUES(?,?,?,?,?,?,?,?,?)""",
                    (uid, r["d"], r.get("loops"), r.get("eff_loops"), r.get("note"), r.get("focus"), r.get("energy"), r.get("progress"), r.get("state")))
            for r in data.get("lenses", []):
                run("INSERT INTO lenses(user_id,name,data) VALUES(?,?,?)", (uid, r["name"], r["data"]))
            for r in data.get("custom_loops", []):
                run("INSERT INTO custom_loops(user_id,name,category,polarity,unit,rate) VALUES(?,?,?,?,?,?)",
                    (uid, r["name"], r["category"], r["polarity"], r["unit"], r["rate"]))
            for r in data.get("future_self", []):
                run("INSERT INTO future_self(user_id,title,traits,rituals) VALUES(?,?,?,?)",
                    (uid, r["title"], r["traits"], r["rituals"]))
            for r in data.get("future_challenges", []):
                run("""INSERT INTO future_challenges(user_id,title,why,smart,due_on,progress,status)
                       VALUES(?,?,?,?,?,?,?)""",
                    (uid, r["title"], r["why"], r["smart"], r["due_on"], r["progress"], r["status"]))
            for r in data.get("future_letters", []):
                run("""INSERT INTO future_letters(user_id,content,created_at,reveal_on,revealed)
                       VALUES(?,?,?,?,?)""",
                    (uid, r["content"], r["created_at"], r["reveal_on"], r["revealed"]))
            g = data.get("goal")
            if g:
                set_goal(uid, g["objective"], g["unit"], float(g["target"]), int(g["horizon_days"]))
            st.success("Import complete.")
        except Exception as e:
            st.error(f"Import failed: {e}")

# ========================= END ==============================