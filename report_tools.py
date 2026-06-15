from pathlib import Path
import pandas as pd
from langchain_core.tools import tool
from tools.utils import prepare_trip_df, get_cols, risk_by, quality_score

CURRENT_DF = None
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def set_dataframe(df):
    global CURRENT_DF
    CURRENT_DF = df

def get_df():
    if CURRENT_DF is None:
        raise ValueError("No trip data uploaded.")
    return prepare_trip_df(CURRENT_DF)

@tool
def create_excel_report_tool() -> str:
    """Create Excel report with summary, risk, delayed, geofence miss, data loss, NDD, and multi-issue sheets."""
    df = get_df()
    cols = get_cols(df)
    path = OUTPUT_DIR / "trip_monitoring_agent_report.xlsx"
    q = pd.DataFrame([quality_score(df)])
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        q.to_excel(writer, sheet_name="KPI Summary", index=False)
        risk_by(df, cols["transporter"]).to_excel(writer, sheet_name="Transporter Risk", index=False)
        df[df["Delay_Issue"] == True].to_excel(writer, sheet_name="Delayed", index=False)
        df[df["Geofence_Miss_Issue"] == True].to_excel(writer, sheet_name="Geofence Miss", index=False)
        df[df["Data_Loss_Issue"] == True].to_excel(writer, sheet_name="Data Loss", index=False)
        df[df["NDD_Issue"] == True].to_excel(writer, sheet_name="NDD", index=False)
        df[df["Issue_Count"] >= 2].to_excel(writer, sheet_name="Multi Issue", index=False)
    return f"Excel report created: {path}"

@tool
def create_text_rca_report_tool() -> str:
    """Create RCA text report with recommended actions."""
    df = get_df()
    q = quality_score(df)
    path = OUTPUT_DIR / "trip_monitoring_rca_report.txt"
    content = f"""Trip Monitoring RCA Report

Quality Score: {q['score']}%
Quality Bucket: {q['bucket']}

Observations:
- Delayed Trips: {q['delayed']}
- Geofence Miss: {q['geofence_miss']}
- Data Loss / EPF: {q['data_loss']}
- NDD: {q['ndd']}

Recommended Actions:
1. Delay: Validate route, unloading TAT, customer hold, transporter issue.
2. Geofence Miss: Validate geofence coordinates, radius and mapping.
3. Data Loss / EPF: Check device wiring, power supply and GPS health.
4. NDD: Validate actual unloading and trip event continuity.
"""
    path.write_text(content, encoding="utf-8")
    return f"RCA report created: {path}"

def get_report_tools():
    return [create_excel_report_tool, create_text_rca_report_tool]
