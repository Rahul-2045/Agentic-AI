import json
import pandas as pd
from langchain_core.tools import tool
from tools.utils import prepare_trip_df, get_cols, dataframe_to_text, group_count, risk_by, quality_score

CURRENT_DF = None

def set_dataframe(df):
    global CURRENT_DF
    CURRENT_DF = df

def get_df():
    if CURRENT_DF is None:
        raise ValueError("No trip data uploaded.")
    return prepare_trip_df(CURRENT_DF)

@tool
def trip_kpi_summary_tool() -> str:
    """Return complete KPI summary of trips, delay, geofence miss, data loss, NDD, and quality score."""
    df = get_df()
    q = quality_score(df)
    return json.dumps(q, indent=2)

@tool
def trip_status_summary_tool() -> str:
    """Return trip status wise count."""
    df = get_df()
    cols = get_cols(df)
    if not cols["status"]:
        return "Trip Status column missing."
    result = df[cols["status"]].fillna("Blank").value_counts().reset_index()
    result.columns = ["Trip Status", "Count"]
    return dataframe_to_text(result)

@tool
def transporter_risk_tool() -> str:
    """Return transporter wise risk score."""
    df = get_df()
    cols = get_cols(df)
    return dataframe_to_text(risk_by(df, cols["transporter"]))

@tool
def vehicle_risk_tool() -> str:
    """Return vehicle wise risk score."""
    df = get_df()
    cols = get_cols(df)
    return dataframe_to_text(risk_by(df, cols["vehicle"]))

@tool
def customer_risk_tool() -> str:
    """Return customer wise risk score."""
    df = get_df()
    cols = get_cols(df)
    return dataframe_to_text(risk_by(df, cols["customer"]))

@tool
def destination_risk_tool() -> str:
    """Return destination wise risk score."""
    df = get_df()
    cols = get_cols(df)
    return dataframe_to_text(risk_by(df, cols["destination"]))

@tool
def delay_cases_tool() -> str:
    """Return all delayed trip cases above 10 hours."""
    df = get_df()
    return dataframe_to_text(df[df["Delay_Issue"] == True].sort_values("Onward_Duration_Hours", ascending=False))

@tool
def geofence_miss_cases_tool() -> str:
    """Return all geofence miss trip cases."""
    df = get_df()
    return dataframe_to_text(df[df["Geofence_Miss_Issue"] == True])

@tool
def data_loss_cases_tool() -> str:
    """Return all data loss or EPF cases."""
    df = get_df()
    return dataframe_to_text(df[df["Data_Loss_Issue"] == True])

@tool
def ndd_cases_tool() -> str:
    """Return all NDD or NDD-EPF cases."""
    df = get_df()
    return dataframe_to_text(df[df["NDD_Issue"] == True])

@tool
def multi_issue_cases_tool() -> str:
    """Return trips having two or more monitoring issues."""
    df = get_df()
    return dataframe_to_text(df[df["Issue_Count"] >= 2].sort_values("Issue_Count", ascending=False))

@tool
def delay_by_transporter_tool() -> str:
    """Return delayed trip count by transporter."""
    df = get_df()
    cols = get_cols(df)
    return dataframe_to_text(group_count(df, cols["transporter"], "Delay_Issue"))

@tool
def geofence_miss_by_destination_tool() -> str:
    """Return geofence miss count by destination."""
    df = get_df()
    cols = get_cols(df)
    return dataframe_to_text(group_count(df, cols["destination"], "Geofence_Miss_Issue"))

@tool
def data_loss_by_vehicle_tool() -> str:
    """Return data loss count by vehicle."""
    df = get_df()
    cols = get_cols(df)
    return dataframe_to_text(group_count(df, cols["vehicle"], "Data_Loss_Issue"))

@tool
def ndd_by_customer_tool() -> str:
    """Return NDD count by customer."""
    df = get_df()
    cols = get_cols(df)
    return dataframe_to_text(group_count(df, cols["customer"], "NDD_Issue"))

def get_kpi_tools():
    return [
        trip_kpi_summary_tool, trip_status_summary_tool, transporter_risk_tool,
        vehicle_risk_tool, customer_risk_tool, destination_risk_tool,
        delay_cases_tool, geofence_miss_cases_tool, data_loss_cases_tool,
        ndd_cases_tool, multi_issue_cases_tool, delay_by_transporter_tool,
        geofence_miss_by_destination_tool, data_loss_by_vehicle_tool, ndd_by_customer_tool
    ]
