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

st.set_page_config(page_title="Offline Ollama Agentic AI - Trip Monitoring", page_icon="🤖", layout="wide")
st.markdown("### 🎯 Mission Control Dashboard")
st.markdown("<div class=\"small-muted\">Upload CSV/Excel and interact with your offline AI trip analyst.</div>", unsafe_allow_html=True)
st.warning("Learning/project demo only. Verify output before client communication.")


# ---------------- Gaming / Mobile App Style UI ----------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, rgba(0,255,255,.18), transparent 30%),
                radial-gradient(circle at top right, rgba(255,0,180,.18), transparent 30%),
                linear-gradient(135deg,#06111f,#0b1020,#14091f);
    color: white;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d1b2a,#111827);
    border-right: 1px solid rgba(0,245,255,.25);
}
.hero-card {
    padding: 28px;
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(0,245,255,.16), rgba(236,72,153,.14));
    border: 1px solid rgba(255,255,255,.15);
    box-shadow: 0 0 35px rgba(0,245,255,.18);
    margin-bottom: 18px;
    animation: floatCard 4s ease-in-out infinite;
}
@keyframes floatCard {
    0% {transform: translateY(0)}
    50% {transform: translateY(-6px)}
    100% {transform: translateY(0)}
}
.hero-title {
    font-size: 42px;
    font-weight: 900;
    background: linear-gradient(90deg,#00f5ff,#ff36d6,#facc15);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    color: #d1d5db;
    font-size: 16px;
    margin-top: 8px;
}
.glass-card {
    padding: 18px;
    border-radius: 22px;
    background: rgba(255,255,255,.08);
    border: 1px solid rgba(255,255,255,.14);
    box-shadow: 0 8px 28px rgba(0,0,0,.28);
    margin-bottom: 14px;
}
.metric-card {
    padding: 18px;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(0,245,255,.16), rgba(124,58,237,.18));
    border: 1px solid rgba(0,245,255,.25);
    text-align: center;
    box-shadow: 0 0 22px rgba(0,245,255,.12);
}
.metric-number {
    font-size: 32px;
    font-weight: 900;
    color: #67e8f9;
}
.metric-label {
    color: #d1d5db;
    font-size: 13px;
}
.chip {
    display: inline-block;
    padding: 8px 13px;
    margin: 4px;
    border-radius: 999px;
    background: rgba(0,245,255,.12);
    border: 1px solid rgba(0,245,255,.28);
    color: #e0f7ff;
    font-size: 13px;
}
.pulse-dot {
    height: 10px;
    width: 10px;
    background: #22c55e;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
    animation: pulse 1.6s infinite;
}
@keyframes pulse {
    0% {box-shadow: 0 0 0 0 rgba(34,197,94,.8)}
    70% {box-shadow: 0 0 0 12px rgba(34,197,94,0)}
    100% {box-shadow: 0 0 0 0 rgba(34,197,94,0)}
}
div[data-testid="stFileUploader"] {
    padding: 18px;
    border-radius: 22px;
    border: 2px dashed rgba(0,245,255,.35);
    background: rgba(255,255,255,.06);
}
.stButton button {
    border-radius: 16px;
    background: linear-gradient(90deg,#06b6d4,#7c3aed,#ec4899);
    color: white;
    border: 0;
    font-weight: 800;
    box-shadow: 0 0 18px rgba(124,58,237,.35);
}
.stButton button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 28px rgba(236,72,153,.55);
}
div[data-testid="stChatMessage"] {
    border-radius: 22px;
    padding: 12px;
    background: rgba(255,255,255,.07);
    border: 1px solid rgba(255,255,255,.10);
}
.small-muted {color:#9ca3af;font-size:13px;}
</style>
""", unsafe_allow_html=True)

def render_hero():
    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    st.markdown(f"""
    <div class="hero-card">
        <div class="hero-title">🚚 TripPulse AI Arena</div>
        <div class="hero-subtitle">
            Offline Ollama Agentic AI assistant for trip monitoring analytics.
            Upload trip data, ask questions, and let the local AI agent choose the right tool.
        </div>
        <br>
        <span class="chip">🧠 Local Model: {model_name}</span>
        <span class="chip">⚡ No API Key</span>
        <span class="chip">📊 Trip Analytics</span>
        <span class="chip">🎮 Gaming UI</span>
    </div>
    <div class="glass-card">
        <span class="pulse-dot"></span><b>Offline Engine Active</b><br>
        <span class="small-muted">Provider: Ollama Local | Mode: API-Free | Tools: NDD, Delay, Geofence, Data Loss, Email Summary</span>
    </div>
    """, unsafe_allow_html=True)

def render_metric_card(value, label):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-number">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

render_hero()


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
    """
    Extract only readable text from model response and remove Gemini metadata/signature.
    """

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

@tool
def data_summary_tool() -> str:
    """Get total rows, total columns, and available column names from uploaded trip data."""
    df = get_df()
    return json.dumps({"total_records": int(len(df)), "total_columns": int(len(df.columns)), "columns": list(df.columns)}, indent=2)

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
    duration_col = find_column(df, ["Onwards Travel Duration(Hrs.Mins)", "Onward Duration", "Onwards Travel Duration", "onward duration"])
    if status_col is None:
        return "Trip Status column not found."
    if duration_col is None:
        return "Onwards Travel Duration column not found."
    data = df.copy()
    data["Onward_Duration_Hours"] = data[duration_col].apply(duration_to_hours)
    result = data[(data[status_col].astype(str).str.upper() == "TRIP MONITORED") & (data["Onward_Duration_Hours"] > threshold_hours)]
    return dataframe_to_text(result)


@tool
def destination_delay_tool() -> str:
    """Find destination with highest average delay based on onward travel duration."""
    df = get_df()

    destination_col = find_column(
        df,
        ["Destination", "DESTINATION", "destination"]
    )

    duration_col = find_column(
        df,
        [
            "Onwards Travel Duration(Hrs.Mins)",
            "Onward Duration",
            "Onwards Travel Duration",
            "onward duration"
        ]
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
            Maximum_Duration_Hours=("Onward_Duration_Hours", "max")
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
        condition = condition | df[remark_col].astype(str).str.upper().str.contains("DATA LOSS|EPF", na=False, regex=True)
    if accountability_col:
        condition = condition | df[accountability_col].astype(str).str.upper().str.contains("DATA LOSS|EPF", na=False, regex=True)
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
        client_email_summary_tool
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
    """
    Creates offline/local Ollama model.
    No Gemini/OpenAI/Groq API key is required.

    Before running:
    1. Install Ollama from https://ollama.com/download
    2. Run: ollama pull qwen2.5:7b
    3. Test: ollama run qwen2.5:7b
    """

    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

    return ChatOllama(
        model=ollama_model,
        temperature=0
    )


def create_agent_with_llm(llm):
    return create_agent(
        model=llm,
        tools=get_agent_tools(),
        system_prompt=get_system_prompt()
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


def run_agent_with_ollama(user_question):
    """
    Runs the agent using local/offline Ollama model.
    """

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
            "Ollama local model failed. Please check:\\n\\n"
            "1. Ollama is installed.\\n"
            "2. Ollama is running.\\n"
            "3. Model is downloaded. Example command:\\n"
            "   ollama pull qwen2.5:7b\\n\\n"
            "4. Test in terminal:\\n"
            "   ollama run qwen2.5:7b\\n\\n"
            f"Actual error:\\n{error_text}"
        )
        return help_text, "Ollama Failed", [error_text]


def create_trip_monitoring_agent():
    llm = create_ollama_llm()
    return create_agent_with_llm(llm)


st.sidebar.header("🎮 AI Control Panel")
st.sidebar.success("🟢 Offline mode: No API key required")
st.sidebar.code("""Install Ollama:
https://ollama.com/download

Download model:
ollama pull qwen2.5:7b

Optional environment variable:
OLLAMA_MODEL=qwen2.5:7b""")
st.sidebar.info("This app uses only local Ollama. Gemini, OpenAI, and Groq are removed.")

st.sidebar.write("🕹️ Example Missions:")
st.sidebar.code("""Show trip status summary
Find delayed trips above 10 hours
Which destination has the highest delays?
Show NDD trips
Show geofence miss trips
Show data loss trips
Transporter wise summary
Vehicle wise count
Draft email summary""")

uploaded_file = st.file_uploader("🎮 Upload Trip Data CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        uploaded_df = pd.read_csv(uploaded_file)
    else:
        uploaded_df = pd.read_excel(uploaded_file)

    uploaded_df = clean_column_names(uploaded_df)
    CURRENT_DF = uploaded_df

    st.success("✅ File uploaded successfully. AI dashboard unlocked!")

    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card(len(uploaded_df), "Total Rows")
    with col2:
        render_metric_card(len(uploaded_df.columns), "Total Columns")
    with col3:
        render_metric_card("LOCAL", "AI Mode")

    st.markdown("### 🧾 Data Preview")
    st.markdown('<div class="glass-card">Preview of uploaded trip monitoring data.</div>', unsafe_allow_html=True)
    st.dataframe(uploaded_df.head(10), use_container_width=True)

    st.markdown("### 💬 AI Command Chat")

    st.markdown(
        '<div class="glass-card">Ask any trip-monitoring question. Example: <b>Find NDD cases</b> or <b>Which destination has the highest delays?</b></div>',
        unsafe_allow_html=True
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Show previous chat messages
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            if chat["role"] == "assistant":
                st.text(chat["content"])
            else:
                st.write(chat["content"])

    user_question = st.chat_input("Ask anything about uploaded trip data...")

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)

        st.session_state.chat_history.append(
            {"role": "user", "content": user_question}
        )

        with st.chat_message("assistant"):
            with st.spinner("🎮 AI Agent is loading tools, thinking, and preparing answer..."):
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

    col_clear, col_examples = st.columns(2)

    with col_clear:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    with col_examples:
        with st.expander("Sample Questions"):
            st.write("""
You can ask:
- Show trip status summary
- Find NDD cases
- Find delayed trips above 10 hours
- Which destination has the highest delays?
- Which transporter has highest risk?
- Show geofence miss cases
- Show data loss and EPF cases
- Draft client email summary
- Create Excel report
- Prepare complete business analysis
""")

else:
    st.markdown('<div class="glass-card">📁 Upload your CSV or Excel file to start the AI mission.</div>', unsafe_allow_html=True)
    st.markdown("""
### Recommended Columns
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
""")

st.markdown("---")
st.caption("🎮 TripPulse AI Arena | Offline Ollama + LangChain + Streamlit + Pandas")
