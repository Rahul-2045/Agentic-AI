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
        "Missing libraries. Run:\n\n"
        "pip install streamlit pandas numpy openpyxl langchain langchain-core langchain-ollama ollama tabulate"
    )
    st.code(str(e))
    st.stop()

st.set_page_config(
    page_title="TripPulse AI Arena",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(34,211,238,.18), transparent 28%),
        radial-gradient(circle at 85% 10%, rgba(236,72,153,.15), transparent 28%),
        linear-gradient(135deg,#070B16,#0B1020,#130A1F) !important;
    color:#F8FAFC !important;
}
.stApp * { color:#F8FAFC !important; }
.block-container { padding-top:1.2rem; max-width:1450px; }
section[data-testid="stSidebar"] {
    background:#0B1220 !important;
    border-right:1px solid rgba(34,211,238,.22);
}
.hero {
    background:linear-gradient(135deg,rgba(34,211,238,.14),rgba(139,92,246,.14),rgba(236,72,153,.10));
    border:1px solid rgba(34,211,238,.28);
    border-radius:28px;
    padding:30px;
    margin-bottom:18px;
    box-shadow:0 20px 60px rgba(0,0,0,.35),0 0 35px rgba(34,211,238,.12);
}
.hero-title {
    font-size:44px;
    font-weight:900;
    background:linear-gradient(90deg,#22D3EE,#A78BFA,#F472B6,#FACC15);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}
.hero-sub { color:#CBD5E1 !important; font-size:16px; line-height:1.6; }
.chip {
    display:inline-block;
    padding:9px 14px;
    margin:10px 6px 0 0;
    border-radius:999px;
    background:rgba(15,23,42,.85);
    border:1px solid rgba(34,211,238,.28);
    font-size:13px;
    font-weight:700;
}
.panel {
    background:rgba(15,23,42,.88);
    border:1px solid rgba(148,163,184,.22);
    border-radius:22px;
    padding:20px;
    margin-bottom:18px;
    box-shadow:0 12px 35px rgba(0,0,0,.25);
}
.panel-title { font-size:19px; font-weight:850; margin-bottom:8px; }
.panel-sub { font-size:13px; color:#CBD5E1 !important; }
.kpi {
    background:linear-gradient(135deg,rgba(34,211,238,.12),rgba(139,92,246,.12));
    border:1px solid rgba(34,211,238,.22);
    border-radius:22px;
    padding:18px;
    min-height:112px;
}
.kpi-value { font-size:33px;font-weight:900;color:#67E8F9 !important; }
.kpi-label { font-size:13px;color:#CBD5E1 !important;font-weight:700; }
.kpi-icon { font-size:24px; }
div[data-testid="stFileUploader"] {
    background:rgba(15,23,42,.92) !important;
    border:2px dashed rgba(34,211,238,.36) !important;
    border-radius:22px !important;
    padding:15px !important;
}
div[data-testid="stFileUploader"] section {
    background:rgba(2,6,23,.55) !important;
    border-radius:16px !important;
}
div[data-testid="stFileUploader"] button,
.stButton button {
    background:linear-gradient(90deg,#06B6D4,#8B5CF6,#EC4899) !important;
    color:white !important;
    border:none !important;
    border-radius:14px !important;
    font-weight:800 !important;
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input {
    background:rgba(15,23,42,.98) !important;
    color:#F8FAFC !important;
    border:1px solid rgba(34,211,238,.30) !important;
    border-radius:18px !important;
}
div[data-testid="stChatMessage"] {
    background:rgba(15,23,42,.90) !important;
    border:1px solid rgba(148,163,184,.20) !important;
    border-radius:22px !important;
}
[data-testid="stDataFrame"], details, [data-testid="stExpander"], [data-testid="stAlert"] {
    background:rgba(15,23,42,.92) !important;
    border-radius:16px !important;
}
pre, code, .stCode {
    background:rgba(2,6,23,.95) !important;
    color:#E0F2FE !important;
}
.footer {
    color:#94A3B8 !important;
    font-size:12px;
    padding-top:16px;
    border-top:1px solid rgba(148,163,184,.15);
    margin-top:30px;
}
</style>
""", unsafe_allow_html=True)


def hero():
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    st.markdown(f"""
    <div class="hero">
        <div class="hero-title">🚚 TripPulse AI Arena</div>
        <div class="hero-sub">
            Offline AI assistant for trip monitoring. Ask general questions anytime.
            Upload CSV/Excel to unlock data analysis tools.
        </div>
        <span class="chip">🧠 Ollama: {model}</span>
        <span class="chip">🔐 No API Key</span>
        <span class="chip">📊 Pandas Tools</span>
        <span class="chip">🎮 Dark UI</span>
    </div>
    """, unsafe_allow_html=True)


def panel(title, subtitle=""):
    st.markdown(f"""
    <div class="panel">
        <div class="panel-title">{title}</div>
        <div class="panel-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def kpi(value, label, icon):
    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


hero()
panel("🟢 Offline Engine Active", "General chat works without upload. Data analytics works after upload.")

CURRENT_DF: Optional[pd.DataFrame] = None


def clean_column_names(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_column(df, names):
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.strip().lower() in normalized:
            return normalized[name.strip().lower()]
    for col in df.columns:
        low = str(col).strip().lower()
        for name in names:
            if name.strip().lower() in low:
                return col
    return None


def duration_to_hours(value):
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


def dataframe_to_text(df, max_rows=20):
    if df is None:
        return "No data available."
    if len(df) == 0:
        return "No matching records found."
    return df.head(max_rows).to_string(index=False)


def clean_answer(answer):
    if answer is None:
        return ""
    if isinstance(answer, list):
        parts = []
        for item in answer:
            parts.append(str(item.get("text", "")) if isinstance(item, dict) else str(item))
        return "\n".join([p for p in parts if p.strip()]).strip()
    if isinstance(answer, dict):
        if "text" in answer:
            return str(answer["text"]).strip()
        if "content" in answer:
            return clean_answer(answer["content"])
    return str(answer).strip()


def get_df():
    if CURRENT_DF is None:
        raise ValueError("No trip data uploaded yet.")
    return CURRENT_DF


@tool
def data_summary_tool() -> str:
    """Get total rows, total columns, and column names from uploaded trip data."""
    df = get_df()
    return json.dumps({"total_records": int(len(df)), "total_columns": int(len(df.columns)), "columns": list(df.columns)}, indent=2)


@tool
def trip_status_summary_tool() -> str:
    """Show trip status count or overall trip status summary."""
    df = get_df()
    col = find_column(df, ["Trip Status", "TRIP STATUS", "trip_status"])
    if col is None:
        return "Trip Status column not found."
    summary = df[col].fillna("Blank").value_counts().reset_index()
    summary.columns = ["Trip Status", "Count"]
    return dataframe_to_text(summary)


@tool
def delayed_trips_tool(threshold_hours: int = 10) -> str:
    """Find delayed trips above threshold hours. Default threshold is 10."""
    df = get_df()
    status_col = find_column(df, ["Trip Status", "TRIP STATUS"])
    duration_col = find_column(df, ["Onwards Travel Duration(Hrs.Mins)", "Onward Duration", "Onwards Travel Duration"])
    if duration_col is None:
        return "Onwards Travel Duration column not found."
    data = df.copy()
    data["Onward_Duration_Hours"] = data[duration_col].apply(duration_to_hours)
    if status_col:
        data = data[data[status_col].astype(str).str.upper() == "TRIP MONITORED"]
    return dataframe_to_text(data[data["Onward_Duration_Hours"] > threshold_hours])


@tool
def destination_delay_tool() -> str:
    """Find destination with highest average delay/duration."""
    df = get_df()
    dest_col = find_column(df, ["Destination", "DESTINATION", "destination"])
    duration_col = find_column(df, ["Onwards Travel Duration(Hrs.Mins)", "Onward Duration", "Onwards Travel Duration"])
    if dest_col is None:
        return "Destination column not found."
    if duration_col is None:
        return "Onwards Travel Duration column not found."
    data = df.copy()
    data["Onward_Duration_Hours"] = data[duration_col].apply(duration_to_hours)
    summary = (
        data.groupby(dest_col)
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
    return f"Destination with highest average delay/duration: {top[dest_col]} with average duration {top['Average_Duration_Hours']} hours.\n\n" + dataframe_to_text(summary.head(10))


@tool
def ndd_trips_tool() -> str:
    """Find NDD or NDD-EPF trips."""
    df = get_df()
    status_col = find_column(df, ["Trip Status", "TRIP STATUS"])
    remark_col = find_column(df, ["Trip Remark", "TRIP REMARK"])
    condition = pd.Series(False, index=df.index)
    if status_col:
        condition |= df[status_col].astype(str).str.upper().str.contains("NDD", na=False)
    if remark_col:
        condition |= df[remark_col].astype(str).str.upper().str.contains("NDD", na=False)
    return dataframe_to_text(df[condition])


@tool
def geofence_miss_tool() -> str:
    """Find geofence miss trips."""
    df = get_df()
    col = find_column(df, ["Geofence Hit Miss Status", "Geofence Hit/Miss Status"])
    if col is None:
        return "Geofence Hit Miss Status column not found."
    return dataframe_to_text(df[df[col].astype(str).str.upper().str.contains("MISS", na=False)])


@tool
def data_loss_tool() -> str:
    """Find data loss or EPF trips."""
    df = get_df()
    remark_col = find_column(df, ["Trip Remark", "TRIP REMARK"])
    account_col = find_column(df, ["Accountability", "ACCOUNTABILITY"])
    condition = pd.Series(False, index=df.index)
    if remark_col:
        condition |= df[remark_col].astype(str).str.upper().str.contains("DATA LOSS|EPF", na=False, regex=True)
    if account_col:
        condition |= df[account_col].astype(str).str.upper().str.contains("DATA LOSS|EPF", na=False, regex=True)
    return dataframe_to_text(df[condition])


@tool
def transporter_summary_tool() -> str:
    """Create transporter-wise trip summary."""
    df = get_df()
    transporter_col = find_column(df, ["Transporter Name", "Transporter"])
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

1. Delayed Trips:
{delayed_trips_tool.invoke({"threshold_hours": 10})}

2. NDD / NDD-EPF Trips:
{ndd_trips_tool.invoke({})}

3. Geofence Miss Trips:
{geofence_miss_tool.invoke({})}

4. Data Loss / EPF Trips:
{data_loss_tool.invoke({})}

Request you to kindly review the highlighted cases and take necessary action.

Thanks & Regards,
Trip Monitoring Agent
"""


def create_llm():
    return ChatOllama(model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"), temperature=0)


def ask_general_llm(question):
    prompt = f"""You are a helpful AI tutor for Data Analytics, AI Agents, Agentic AI, Python, SQL, and logistics trip monitoring.
Answer clearly and simply.

User question:
{question}
"""
    response = create_llm().invoke(prompt)
    return clean_answer(getattr(response, "content", response))


def create_data_agent():
    return create_agent(
        model=create_llm(),
        tools=[
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
        ],
        system_prompt=(
            "You are a logistics Trip Monitoring AI Agent. "
            "For uploaded-data questions, select the correct tool. "
            "Never invent trip counts, vehicles, destinations, or transporter names."
        ),
    )


def ask_data_agent(question):
    try:
        response = create_data_agent().invoke({"messages": [{"role": "user", "content": question}]})
        answer = clean_answer(response["messages"][-1].content)
        if not answer:
            for msg in reversed(response["messages"]):
                if msg.__class__.__name__ == "ToolMessage":
                    answer = clean_answer(msg.content)
                    break
        return answer or "Tool executed successfully, but model returned blank response.", []
    except Exception as e:
        return (
            "Ollama failed. Check that Ollama is installed, running, and model is downloaded.\n\n"
            "Commands:\nollama pull qwen2.5:7b\nollama run qwen2.5:7b\n\n"
            f"Actual error:\n{e}"
        ), [str(e)]


st.sidebar.markdown("## 🎮 AI Control Panel")
st.sidebar.success("🟢 Offline mode: No API key required")
st.sidebar.markdown("### 🧠 Ollama Setup")
st.sidebar.code("""Install Ollama:
https://ollama.com/download

Download model:
ollama pull qwen2.5:7b

Optional:
OLLAMA_MODEL=qwen2.5:7b""")
st.sidebar.markdown("### 🕹️ General Questions")
st.sidebar.code("""What is NDD?
What is geofence miss?
What is an AI Agent?
What is Agentic AI?
Explain LangChain tools""")
st.sidebar.markdown("### 📊 Data Questions")
st.sidebar.code("""Show trip status summary
Find NDD cases
Find delayed trips above 10 hours
Which destination has highest delays?
Draft email summary""")


uploaded_file = st.file_uploader("🎮 Upload Trip Data CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    uploaded_df = clean_column_names(uploaded_df)
    CURRENT_DF = uploaded_df
    st.success("✅ File uploaded successfully. Data tools unlocked!")

    status_col = find_column(uploaded_df, ["Trip Status", "TRIP STATUS"])
    remark_col = find_column(uploaded_df, ["Trip Remark", "TRIP REMARK"])
    geofence_col = find_column(uploaded_df, ["Geofence Hit Miss Status", "Geofence Hit/Miss Status"])
    duration_col = find_column(uploaded_df, ["Onwards Travel Duration(Hrs.Mins)", "Onward Duration", "Onwards Travel Duration"])

    ndd_count = 0
    if status_col:
        ndd_count += int(uploaded_df[status_col].astype(str).str.upper().str.contains("NDD", na=False).sum())
    if remark_col:
        ndd_count += int(uploaded_df[remark_col].astype(str).str.upper().str.contains("NDD", na=False).sum())

    miss_count = int(uploaded_df[geofence_col].astype(str).str.upper().str.contains("MISS", na=False).sum()) if geofence_col else 0

    delay_count = 0
    if duration_col:
        temp = uploaded_df.copy()
        temp["__duration_hours"] = temp[duration_col].apply(duration_to_hours)
        delay_count = int((temp["__duration_hours"] > 10).sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi(len(uploaded_df), "Total Records", "📦")
    with c2:
        kpi(ndd_count, "NDD Signals", "📍")
    with c3:
        kpi(miss_count, "Geofence Miss", "🛰️")
    with c4:
        kpi(delay_count, "Delay > 10 Hrs", "⏱️")

    st.markdown("### 🧾 Data Preview")
    panel("Uploaded Data Snapshot", "First 10 rows from your trip monitoring file.")
    st.dataframe(uploaded_df.head(10), use_container_width=True)
else:
    panel("💬 General Chat Mode Active", "Ask general AI/logistics questions without uploading data.")

st.markdown("### 💬 AI Command Chat")
panel("Ask Anything", "Without upload: general questions. With upload: trip data analysis.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

user_question = st.chat_input("Ask general question or upload data for trip analysis...")

if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)

    st.session_state.chat_history.append({"role": "user", "content": user_question})

    with st.chat_message("assistant"):
        if CURRENT_DF is None:
            with st.spinner("🧠 Answering general question using local Ollama..."):
                answer = ask_general_llm(user_question)
                st.caption("Mode: General Chat | Model: Ollama Local")
                st.markdown(answer)
        else:
            with st.spinner("🎮 Data Agent is selecting tools and preparing answer..."):
                answer, errors = ask_data_agent(user_question)
                st.caption("Mode: Data Agent | Model: Ollama Local")
                st.markdown(answer)
                if errors:
                    with st.expander("Error details"):
                        for err in errors:
                            st.write(err)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})

c1, c2 = st.columns([1, 3])
with c1:
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
with c2:
    with st.expander("🎯 Sample Questions"):
        st.markdown("""
**Without upload**
- What is NDD?
- What is geofence miss?
- What is an AI Agent?
- What is Agentic AI?
- Explain LangChain tools.

**After upload**
- Show trip status summary.
- Find NDD cases.
- Find delayed trips above 10 hours.
- Which destination has highest delays?
- Draft client email summary.
""")

st.markdown('<div class="footer">🎮 TripPulse AI Arena | General Chat + Data Agent | Ollama + LangChain + Streamlit + Pandas</div>', unsafe_allow_html=True)
