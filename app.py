
import os
import json
from typing import Optional
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

try:
    from langchain.agents import create_agent
    from langchain_core.tools import tool
    from langchain_google_genai import ChatGoogleGenerativeAI
except ModuleNotFoundError as e:
    st.error(
        "Required libraries are missing.\n\n"
        "Run:\n\n"
        "python -m pip install streamlit pandas numpy openpyxl python-dotenv "
        "langchain langchain-core langchain-google-genai google-generativeai tabulate"
    )
    st.code(str(e))
    st.stop()

load_dotenv()

st.set_page_config(page_title="Gemini Agentic AI - Trip Monitoring", page_icon="🤖", layout="wide")
st.title("🤖 Gemini Agentic AI Chat: Trip Monitoring Assistant")
st.write("Upload trip data and ask questions. Gemini + LangChain agent will select the correct tool and analyze the data.")
st.warning("Learning/project demo only. Verify output before client communication.")

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

def create_trip_monitoring_agent():
    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    if not api_key:
        st.error("GOOGLE_API_KEY is missing. Create `.env` file and add:\n\nGOOGLE_API_KEY=your_gemini_api_key_here\nGEMINI_MODEL=gemini-2.5-flash-lite")
        st.stop()

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=api_key)

    tools = [
        data_summary_tool, trip_status_summary_tool, delayed_trips_tool,
        ndd_trips_tool, geofence_miss_tool, data_loss_tool,
        transporter_summary_tool, vehicle_summary_tool, client_email_summary_tool
    ]

    system_prompt = """
You are a logistics Trip Monitoring Agentic AI assistant.
Understand the user's question, select the correct tool, analyze uploaded trip data, and provide concise business-friendly output.
Always use tools for data questions. Never invent trip counts or vehicle numbers.
If a required column is missing, clearly mention the missing column.
"""

    return create_agent(model=llm, tools=tools, system_prompt=system_prompt)

st.sidebar.header("Setup")
st.sidebar.write("Required `.env` file:")
st.sidebar.code("GOOGLE_API_KEY=your_gemini_api_key_here\nGEMINI_MODEL=gemini-2.5-flash-lite")

st.sidebar.write("Example Questions:")
st.sidebar.code("""Show trip status summary
Find delayed trips above 10 hours
Show NDD trips
Show geofence miss trips
Show data loss trips
Transporter wise summary
Vehicle wise count
Draft email summary""")

uploaded_file = st.file_uploader("Upload Trip Data CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        uploaded_df = pd.read_csv(uploaded_file)
    else:
        uploaded_df = pd.read_excel(uploaded_file)

    uploaded_df = clean_column_names(uploaded_df)
    CURRENT_DF = uploaded_df

    st.success("File uploaded successfully.")
    col1, col2 = st.columns(2)
    col1.metric("Total Rows", len(uploaded_df))
    col2.metric("Total Columns", len(uploaded_df.columns))

    st.subheader("Data Preview")
    st.dataframe(uploaded_df.head(10), use_container_width=True)

    st.subheader("Ask Questions")

    st.info(
        "This version is not hardcoded. User can ask any free-form question using the chat box below."
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

        agent = create_trip_monitoring_agent()

        with st.chat_message("assistant"):
            with st.spinner("Gemini Agent is thinking and selecting the correct tool..."):
                try:
                    response = agent.invoke(
                        {"messages": [{"role": "user", "content": user_question}]}
                    )

                    answer = ""

                    try:
                        answer = response["messages"][-1].content
                    except Exception:
                        answer = ""

                    # If Gemini final answer is blank, show the last tool output
                    if not answer or str(answer).strip() == "":
                        for msg in reversed(response["messages"]):
                            if msg.__class__.__name__ == "ToolMessage":
                                answer = msg.content
                                break

                    if not answer or str(answer).strip() == "":
                        answer = "Tool executed successfully, but Gemini returned blank response."

                except Exception as e:
                    error_text = str(e)

                    if "API key expired" in error_text or "API_KEY_INVALID" in error_text:
                        answer = (
                            "Gemini API key is expired or invalid. "
                            "Please create a new API key and update your .env file."
                        )
                    elif "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
                        answer = (
                            "Gemini quota limit reached. Please wait and try again, "
                            "or use another Gemini key/project, or enable billing."
                        )
                    elif "NOT_FOUND" in error_text or "404" in error_text:
                        answer = (
                            "Gemini model not found. Please check GEMINI_MODEL in your .env file."
                        )
                    else:
                        answer = f"Agent failed: {error_text}"

                st.text(str(answer))

        st.session_state.chat_history.append(
            {"role": "assistant", "content": str(answer)}
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
- Which transporter has highest risk?
- Show geofence miss cases
- Show data loss and EPF cases
- Draft client email summary
- Create Excel report
- Prepare complete business analysis
""")

else:
    st.info("Please upload CSV or Excel trip data to start.")
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
st.caption("Gemini Agentic AI Project | Gemini API + LangChain + Streamlit + Pandas")
