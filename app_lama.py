import os
import json
from typing import Optional

import pandas as pd
import streamlit as st

try:
    from langchain.agents import create_agent
    from langchain_core.tools import tool
    from langchain_ollama import ChatOllama
except ModuleNotFoundError as e:
    st.error(
        "Required libraries are missing.\n\n"
        "Run:\n\n"
        "python -m pip install streamlit pandas numpy openpyxl langchain langchain-core langchain-ollama ollama tabulate"
    )
    st.code(str(e))
    st.stop()


# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="TripPulse AI Arena",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# Clean, Dark, High-Contrast Gaming UI
# =========================================================
st.markdown(
    """
    <style>
    :root {
        --bg: #070B16;
        --panel: #0F172A;
        --panel2: #111827;
        --card: #101828;
        --border: rgba(148, 163, 184, 0.22);
        --text: #F8FAFC;
        --muted: #CBD5E1;
        --cyan: #22D3EE;
        --purple: #8B5CF6;
        --pink: #EC4899;
        --green: #22C55E;
        --yellow: #FACC15;
        --red: #FB7185;
    }

    html, body, .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(34, 211, 238, 0.18), transparent 28%),
            radial-gradient(circle at 85% 10%, rgba(236, 72, 153, 0.15), transparent 28%),
            linear-gradient(135deg, #070B16 0%, #0B1020 55%, #130A1F 100%) !important;
        color: var(--text) !important;
    }

    /* Universal text safety */
    .stApp, .stApp p, .stApp div, .stApp span, .stApp label, .stApp small,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: var(--text) !important;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 6rem;
        max-width: 1500px;
    }

    section[data-testid="stSidebar"] {
        background: #0B1220 !important;
        border-right: 1px solid rgba(34, 211, 238, 0.22);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    .hero {
        background: linear-gradient(135deg, rgba(34, 211, 238, 0.14), rgba(139, 92, 246, 0.14), rgba(236, 72, 153, 0.10));
        border: 1px solid rgba(34, 211, 238, 0.28);
        border-radius: 28px;
        padding: 30px;
        margin-bottom: 18px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.35), 0 0 35px rgba(34,211,238,0.12);
        animation: floatHero 4s ease-in-out infinite;
    }

    @keyframes floatHero {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
        100% { transform: translateY(0px); }
    }

    .hero-title {
        font-size: 44px;
        line-height: 1.05;
        font-weight: 900;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #22D3EE, #A78BFA, #F472B6, #FACC15);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
    }

    .hero-sub {
        color: var(--muted) !important;
        font-size: 16px;
        line-height: 1.6;
        max-width: 950px;
    }

    .chips {
        margin-top: 18px;
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }

    .chip {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 9px 14px;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(34, 211, 238, 0.28);
        color: var(--text) !important;
        font-size: 13px;
        font-weight: 700;
    }

    .panel {
        background: rgba(15, 23, 42, 0.88);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.25);
    }

    .panel-title {
        font-size: 19px;
        font-weight: 850;
        margin-bottom: 8px;
        color: var(--text) !important;
    }

    .panel-sub {
        font-size: 13px;
        color: var(--muted) !important;
    }

    .status-dot {
        width: 11px;
        height: 11px;
        background: var(--green);
        border-radius: 50%;
        display: inline-block;
        margin-right: 9px;
        animation: pulse 1.6s infinite;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(34, 197, 94, .65); }
        70% { box-shadow: 0 0 0 12px rgba(34, 197, 94, 0); }
        100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
    }

    .kpi {
        background: linear-gradient(135deg, rgba(34, 211, 238, 0.12), rgba(139, 92, 246, 0.12));
        border: 1px solid rgba(34, 211, 238, 0.22);
        border-radius: 22px;
        padding: 18px;
        min-height: 112px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.22);
    }

    .kpi-value {
        font-size: 33px;
        font-weight: 900;
        color: #67E8F9 !important;
        margin-bottom: 4px;
    }

    .kpi-label {
        font-size: 13px;
        color: var(--muted) !important;
        font-weight: 700;
    }

    .kpi-icon {
        font-size: 24px;
        margin-bottom: 6px;
    }

    /* File uploader dark and readable */
    div[data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.92) !important;
        border: 2px dashed rgba(34, 211, 238, 0.36) !important;
        border-radius: 22px !important;
        padding: 15px !important;
    }

    div[data-testid="stFileUploader"] * {
        color: var(--text) !important;
    }

    div[data-testid="stFileUploader"] section {
        background: rgba(2, 6, 23, 0.55) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(148, 163, 184, 0.22) !important;
    }

    div[data-testid="stFileUploader"] button {
        background: linear-gradient(90deg, #06B6D4, #8B5CF6, #EC4899) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 800 !important;
    }

    [data-testid="stFileUploaderFile"] {
        background: rgba(2, 6, 23, 0.75) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(148, 163, 184, 0.20) !important;
    }

    [data-testid="stFileUploaderFile"] * {
        color: var(--text) !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        background: rgba(7, 11, 22, 0.94) !important;
        border-top: 1px solid rgba(34, 211, 238, 0.16) !important;
    }

    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {
        background: rgba(15, 23, 42, 0.98) !important;
        color: var(--text) !important;
        border: 1px solid rgba(34, 211, 238, 0.30) !important;
        border-radius: 18px !important;
    }

    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInput"] input::placeholder {
        color: #E2E8F0 !important;
        opacity: .85 !important;
    }

    /* Chat bubbles */
    div[data-testid="stChatMessage"] {
        background: rgba(15, 23, 42, 0.90) !important;
        border: 1px solid rgba(148, 163, 184, 0.20) !important;
        border-radius: 22px !important;
        padding: 12px !important;
        box-shadow: 0 10px 28px rgba(0,0,0,0.20);
    }

    div[data-testid="stChatMessage"] * {
        color: var(--text) !important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        background: rgba(15, 23, 42, 0.92) !important;
        border: 1px solid rgba(148, 163, 184, 0.20) !important;
        border-radius: 18px !important;
        overflow: hidden;
    }

    /* Expander */
    details, [data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.92) !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        border-radius: 16px !important;
    }

    details *, [data-testid="stExpander"] * {
        color: var(--text) !important;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        background: rgba(15, 23, 42, 0.92) !important;
        border: 1px solid rgba(34, 197, 94, 0.25) !important;
        border-radius: 16px !important;
    }

    [data-testid="stAlert"] * {
        color: var(--text) !important;
    }

    /* Code blocks */
    pre, code, .stCode {
        background: rgba(2, 6, 23, 0.95) !important;
        color: #E0F2FE !important;
        border-radius: 14px !important;
        border: 1px solid rgba(34, 211, 238, 0.18) !important;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, #06B6D4, #8B5CF6, #EC4899) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 900 !important;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.28);
    }

    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 0 28px rgba(236, 72, 153, 0.38);
    }

    .footer {
        color: #94A3B8 !important;
        font-size: 12px;
        padding-top: 16px;
        border-top: 1px solid rgba(148, 163, 184, 0.15);
        margin-top: 30px;
    }

    /* remove random white containers */
    div, section, article {
        color: var(--text) !important;
    }

    input, textarea {
        color: var(--text) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Header / UI Helpers
# =========================================================
def render_hero() -> None:
    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">🚚 TripPulse AI Arena</div>
            <div class="hero-sub">
                A fully offline Agentic AI dashboard for trip monitoring analytics.
                Upload your CSV/Excel file and ask natural language questions.
                The local Ollama agent will select the correct analytics tool and respond.
            </div>
            <div class="chips">
                <span class="chip">🧠 Local Model: {model_name}</span>
                <span class="chip">🔐 No API Key</span>
                <span class="chip">📊 Pandas Tools</span>
                <span class="chip">⚡ Offline Mode</span>
                <span class="chip">🎮 High Contrast UI</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_panel() -> None:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title"><span class="status-dot"></span>Offline Engine Active</div>
            <div class="panel-sub">
                Provider: Ollama Local | Tools: NDD, Delay, Destination Delay, Geofence Miss, Data Loss, Transporter Summary, Vehicle Summary, Email Summary
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi(value, label, icon) -> None:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">{title}</div>
            <div class="panel-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


render_hero()
render_status_panel()


# =========================================================
# Data + Utility Functions
# =========================================================
CURRENT_DF: Optional[pd.DataFrame] = None


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(col).strip() for col in df.columns]
    return df


def find_column(df: pd.DataFrame, possible_names: list[str]) -> Optional[str]:
    normalized_cols = {str(col).strip().lower(): col for col in df.columns}

    for name in possible_names:
        if name.strip().lower() in normalized_cols:
            return normalized_cols[name.strip().lower()]

    for col in df.columns:
        col_low = str(col).strip().lower()
        for name in possible_names:
            if name.strip().lower() in col_low:
                return col

    return None


def duration_to_hours(value) -> float:
    if pd.isna(value):
        return 0.0

    value = str(value).strip()

    if value == "" or value.lower() in ["nan", "none", "null"]:
        return 0.0

    if ":" in value:
        parts = value.split(":")
        try:
            return float(parts[0]) + ((float(parts[1]) if len(parts) > 1 else 0) / 60)
        except Exception:
            return 0.0

    try:
        return float(value)
    except Exception:
        return 0.0


def dataframe_to_text(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None:
        return "No data available."

    if len(df) == 0:
        return "No matching records found."

    return df.head(max_rows).to_string(index=False)


def clean_answer(answer) -> str:
    if answer is None:
        return ""

    if isinstance(answer, list):
        text_parts = []

        for item in answer:
            if isinstance(item, dict):
                text_parts.append(str(item.get("text", "")))
            else:
                text_parts.append(str(item))

        return "\n".join([t for t in text_parts if t.strip()]).strip()

    if isinstance(answer, dict):
        if "text" in answer:
            return str(answer.get("text", "")).strip()
        if "content" in answer:
            return clean_answer(answer.get("content"))
        return str(answer)

    return str(answer).strip()


def get_df() -> pd.DataFrame:
    global CURRENT_DF

    if CURRENT_DF is None:
        raise ValueError("No trip data uploaded yet.")

    return CURRENT_DF


# =========================================================
# Agent Tools
# =========================================================
@tool
def data_summary_tool() -> str:
    """Get total rows, total columns, and available column names from uploaded trip data."""
    df = get_df()

    return json.dumps(
        {
            "total_records": int(len(df)),
            "total_columns": int(len(df.columns)),
            "columns": list(df.columns),
        },
        indent=2,
    )


@tool
def trip_status_summary_tool() -> str:
    """Use this when user asks for trip status count or overall trip status summary."""
    df = get_df()
    col = find_column(df, ["Trip Status", "TRIP STATUS", "trip_status"])

    if col is None:
        return "Trip Status column not found."

    summary = df[col].fillna("Blank").value_counts().reset_index()
    summary.columns = ["Trip Status", "Count"]

    return dataframe_to_text(summary)


@tool
def delayed_trips_tool(threshold_hours: int = 10) -> str:
    """Find delayed trips above a threshold in hours. Default threshold is 10 hours."""
    df = get_df()

    status_col = find_column(df, ["Trip Status", "TRIP STATUS"])
    duration_col = find_column(
        df,
        [
            "Onwards Travel Duration(Hrs.Mins)",
            "Onward Duration",
            "Onwards Travel Duration",
            "onward duration",
        ],
    )

    if status_col is None:
        return "Trip Status column not found."

    if duration_col is None:
        return "Onwards Travel Duration column not found."

    data = df.copy()
    data["Onward_Duration_Hours"] = data[duration_col].apply(duration_to_hours)

    result = data[
        (data[status_col].astype(str).str.upper() == "TRIP MONITORED")
        & (data["Onward_Duration_Hours"] > threshold_hours)
    ]

    return dataframe_to_text(result)


@tool
def destination_delay_tool() -> str:
    """Find destination with highest average delay based on onward travel duration."""
    df = get_df()

    destination_col = find_column(df, ["Destination", "DESTINATION", "destination"])
    duration_col = find_column(
        df,
        [
            "Onwards Travel Duration(Hrs.Mins)",
            "Onward Duration",
            "Onwards Travel Duration",
            "onward duration",
        ],
    )
    status_col = find_column(df, ["Trip Status", "TRIP STATUS", "trip_status"])

    if destination_col is None:
        return "Destination column not found."

    if duration_col is None:
        return "Onwards Travel Duration column not found."

    data = df.copy()
    data["Onward_Duration_Hours"] = data[duration_col].apply(duration_to_hours)

    if status_col is not None:
        data = data[data[status_col].astype(str).str.upper() == "TRIP MONITORED"]

    if data.empty:
        return "No monitored trip data found for destination delay analysis."

    summary = (
        data.groupby(destination_col)
        .agg(
            Trip_Count=("Onward_Duration_Hours", "count"),
            Average_Duration_Hours=("Onward_Duration_Hours", "mean"),
            Maximum_Duration_Hours=("Onward_Duration_Hours", "max"),
        )
        .reset_index()
        .sort_values("Average_Duration_Hours", ascending=False)
    )

    summary["Average_Duration_Hours"] = summary["Average_Duration_Hours"].round(2)
    summary["Maximum_Duration_Hours"] = summary["Maximum_Duration_Hours"].round(2)

    top = summary.iloc[0]

    result_text = (
        f"Destination with highest average delay/duration: {top[destination_col]} "
        f"with average duration {top['Average_Duration_Hours']} hours "
        f"across {int(top['Trip_Count'])} trip(s).\n\n"
    )

    result_text += dataframe_to_text(summary.head(10))

    return result_text


@tool
def ndd_trips_tool() -> str:
    """Find NDD or NDD-EPF trips from trip status or trip remark."""
    df = get_df()

    status_col = find_column(df, ["Trip Status", "TRIP STATUS"])
    remark_col = find_column(df, ["Trip Remark", "TRIP REMARK"])

    if status_col is None and remark_col is None:
        return "Trip Status and Trip Remark columns not found."

    condition = pd.Series(False, index=df.index)

    if status_col:
        condition = condition | df[status_col].astype(str).str.upper().str.contains("NDD", na=False)

    if remark_col:
        condition = condition | df[remark_col].astype(str).str.upper().str.contains("NDD", na=False)

    return dataframe_to_text(df[condition])


@tool
def geofence_miss_tool() -> str:
    """Find geofence miss trips."""
    df = get_df()
    col = find_column(df, ["Geofence Hit Miss Status", "Geofence Hit/Miss Status", "GEOFENCE HIT MISS STATUS"])

    if col is None:
        return "Geofence Hit Miss Status column not found."

    return dataframe_to_text(df[df[col].astype(str).str.upper().str.contains("MISS", na=False)])


@tool
def data_loss_tool() -> str:
    """Find data loss or EPF related trips."""
    df = get_df()

    remark_col = find_column(df, ["Trip Remark", "TRIP REMARK"])
    accountability_col = find_column(df, ["Accountability", "ACCOUNTABILITY"])

    if remark_col is None and accountability_col is None:
        return "Trip Remark and Accountability columns not found."

    condition = pd.Series(False, index=df.index)

    if remark_col:
        condition = condition | df[remark_col].astype(str).str.upper().str.contains(
            "DATA LOSS|EPF", na=False, regex=True
        )

    if accountability_col:
        condition = condition | df[accountability_col].astype(str).str.upper().str.contains(
            "DATA LOSS|EPF", na=False, regex=True
        )

    return dataframe_to_text(df[condition])


@tool
def transporter_summary_tool() -> str:
    """Create transporter-wise trip summary."""
    df = get_df()

    transporter_col = find_column(df, ["Transporter Name", "TRANSPORTER_NAME", "Integration Transporter Name", "Transporter"])
    status_col = find_column(df, ["Trip Status", "TRIP STATUS"])

    if transporter_col is None:
        return "Transporter column not found."

    if status_col:
        summary = df.groupby([transporter_col, status_col]).size().reset_index(name="Count")
    else:
        summary = df[transporter_col].fillna("Blank").value_counts().reset_index()
        summary.columns = ["Transporter", "Count"]

    return dataframe_to_text(summary)


@tool
def vehicle_summary_tool() -> str:
    """Create vehicle-wise trip count summary."""
    df = get_df()
    col = find_column(df, ["Vehicle", "VEHICLE_NO", "Vehicle No", "vehicle"])

    if col is None:
        return "Vehicle column not found."

    summary = df[col].fillna("Blank").value_counts().reset_index()
    summary.columns = ["Vehicle", "Trip Count"]

    return dataframe_to_text(summary.head(20))


@tool
def client_email_summary_tool() -> str:
    """Draft a client-ready email summary from trip monitoring observations."""
    df = get_df()

    return f"""Dear Team,

Please find below the trip monitoring summary.

Total Records Reviewed: {len(df)}

Key Observation Areas:

1. Delayed Trips above 10 hours:
{delayed_trips_tool.invoke({"threshold_hours": 10})}

2. NDD / NDD-EPF Trips:
{ndd_trips_tool.invoke({})}

3. Geofence Miss Trips:
{geofence_miss_tool.invoke({})}

4. Data Loss / EPF Trips:
{data_loss_tool.invoke({})}

Request you to kindly review the highlighted cases and take necessary action wherever required.

Thanks & Regards,
Trip Monitoring Agent
"""


# =========================================================
# Agent Creation
# =========================================================
def get_agent_tools():
    return [
        data_summary_tool,
        trip_status_summary_tool,
        delayed_trips_tool,
        destination_delay_tool,
        ndd_trips_tool,
        geofence_miss_tool,
        data_loss_tool,
        transporter_summary_tool,
        vehicle_summary_tool,
        client_email_summary_tool,
    ]


def get_system_prompt():
    return """
You are a logistics Trip Monitoring Agentic AI assistant.
Understand the user's question, select the correct tool, analyze uploaded trip data, and provide concise business-friendly output.
Always use tools for data questions. Never invent trip counts or vehicle numbers.
If a required column is missing, clearly mention the missing column.
For destination delay questions, use destination_delay_tool.
"""


def create_ollama_llm():
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

    return ChatOllama(
        model=ollama_model,
        temperature=0,
    )


def create_agent_with_llm(llm):
    return create_agent(
        model=llm,
        tools=get_agent_tools(),
        system_prompt=get_system_prompt(),
    )


def extract_agent_answer(response):
    answer = ""

    try:
        answer = response["messages"][-1].content
    except Exception:
        answer = ""

    answer = clean_answer(answer)

    if not answer or str(answer).strip() == "":
        try:
            for msg in reversed(response["messages"]):
                if msg.__class__.__name__ == "ToolMessage":
                    answer = clean_answer(msg.content)
                    break
        except Exception:
            pass

    if not answer or str(answer).strip() == "":
        answer = "Tool executed successfully, but the local Ollama model returned blank response."

    return clean_answer(answer)


def run_agent_with_ollama(user_question: str):
    try:
        llm = create_ollama_llm()
        agent = create_agent_with_llm(llm)

        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_question}]}
        )

        answer = extract_agent_answer(response)
        model_used = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

        return answer, f"Ollama Local ({model_used})", []

    except Exception as e:
        error_text = str(e)
        help_text = (
            "Ollama local model failed. Please check:\n\n"
            "1. Ollama is installed.\n"
            "2. Ollama is running.\n"
            "3. Model is downloaded. Example command:\n"
            "   ollama pull qwen2.5:7b\n\n"
            "4. Test in terminal:\n"
            "   ollama run qwen2.5:7b\n\n"
            f"Actual error:\n{error_text}"
        )

        return help_text, "Ollama Failed", [error_text]


# =========================================================
# Sidebar
# =========================================================
st.sidebar.markdown("## 🎮 AI Control Panel")
st.sidebar.success("🟢 Offline mode: No API key required")
st.sidebar.markdown("### 🧠 Ollama Setup")
st.sidebar.code(
    """Install Ollama:
https://ollama.com/download

Download model:
ollama pull qwen2.5:7b

Optional:
OLLAMA_MODEL=qwen2.5:7b"""
)
st.sidebar.info("This app uses only local Ollama. Gemini, OpenAI, and Groq are removed.")

st.sidebar.markdown("### 🕹️ Example Missions")
st.sidebar.code(
    """Show trip status summary
Find delayed trips above 10 hours
Which destination has the highest delays?
Show NDD trips
Show geofence miss trips
Show data loss trips
Transporter wise summary
Vehicle wise count
Draft email summary"""
)


# =========================================================
# Main App
# =========================================================
uploaded_file = st.file_uploader("🎮 Upload Trip Data CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        uploaded_df = pd.read_csv(uploaded_file)
    else:
        uploaded_df = pd.read_excel(uploaded_file)

    uploaded_df = clean_column_names(uploaded_df)
    CURRENT_DF = uploaded_df

    st.success("✅ File uploaded successfully. AI dashboard unlocked!")

    status_col = find_column(uploaded_df, ["Trip Status", "TRIP STATUS"])
    remark_col = find_column(uploaded_df, ["Trip Remark", "TRIP REMARK"])
    geofence_col = find_column(uploaded_df, ["Geofence Hit Miss Status", "Geofence Hit/Miss Status"])
    duration_col = find_column(uploaded_df, ["Onwards Travel Duration(Hrs.Mins)", "Onward Duration", "Onwards Travel Duration"])

    total_rows = len(uploaded_df)
    ndd_count = 0
    geofence_miss_count = 0
    delay_count = 0

    if status_col:
        ndd_count += int(uploaded_df[status_col].astype(str).str.upper().str.contains("NDD", na=False).sum())

    if remark_col:
        ndd_count += int(uploaded_df[remark_col].astype(str).str.upper().str.contains("NDD", na=False).sum())

    if geofence_col:
        geofence_miss_count = int(uploaded_df[geofence_col].astype(str).str.upper().str.contains("MISS", na=False).sum())

    if duration_col:
        temp_df = uploaded_df.copy()
        temp_df["__duration_hours"] = temp_df[duration_col].apply(duration_to_hours)
        delay_count = int((temp_df["__duration_hours"] > 10).sum())

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_kpi(total_rows, "Total Records", "📦")

    with col2:
        render_kpi(ndd_count, "NDD Signals", "📍")

    with col3:
        render_kpi(geofence_miss_count, "Geofence Miss", "🛰️")

    with col4:
        render_kpi(delay_count, "Delay > 10 Hrs", "⏱️")

    st.markdown("### 🧾 Data Preview")
    render_panel("Uploaded Data Snapshot", "First 10 rows from your trip monitoring file.")
    st.dataframe(uploaded_df.head(10), use_container_width=True)

    st.markdown("### 💬 AI Command Chat")
    render_panel(
        "Ask Anything",
        "Examples: Find NDD cases | Which destination has the highest delays? | Draft client email summary",
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    user_question = st.chat_input("Ask anything about uploaded trip data...")

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)

        st.session_state.chat_history.append(
            {"role": "user", "content": user_question}
        )

        with st.chat_message("assistant"):
            with st.spinner("🎮 Offline AI Agent is selecting tools and preparing answer..."):
                answer, used_model, errors = run_agent_with_ollama(user_question)

                st.caption(f"Model used: {used_model}")
                st.markdown(clean_answer(answer))

                if errors:
                    with st.expander("Error details"):
                        for err in errors:
                            st.write(err)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": clean_answer(answer)}
        )

    bottom_col1, bottom_col2 = st.columns([1, 3])

    with bottom_col1:
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    with bottom_col2:
        with st.expander("🎯 Sample Questions"):
            st.markdown(
                """
                - Show trip status summary
                - Find NDD cases
                - Find delayed trips above 10 hours
                - Which destination has the highest delays?
                - Show geofence miss cases
                - Show data loss and EPF cases
                - Transporter wise summary
                - Vehicle wise count
                - Draft client email summary
                """
            )

else:
    render_panel(
        "📁 Upload Required",
        "Please upload a CSV or Excel trip monitoring file to unlock the AI dashboard.",
    )

    st.markdown("### Recommended Columns")
    st.markdown(
        """
        - Trip Status
        - Trip Remark
        - Vehicle
        - Transporter Name
        - DI No
        - Customer Name
        - Destination
        - Onwards Travel Duration(Hrs.Mins)
        - Geofence Hit Miss Status
        - Accountability
        """
    )


st.markdown('<div class="footer">🎮 TripPulse AI Arena | Offline Ollama + LangChain + Streamlit + Pandas</div>', unsafe_allow_html=True)
