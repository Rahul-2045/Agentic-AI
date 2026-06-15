import json
from pathlib import Path
from typing import Optional
import pandas as pd


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(col).strip() for col in df.columns]
    return df


def find_column(df: pd.DataFrame, possible_names: list[str]) -> Optional[str]:
    normalized_cols = {str(col).strip().lower(): col for col in df.columns}
    for name in possible_names:
        key = name.strip().lower()
        if key in normalized_cols:
            return normalized_cols[key]
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
            hours = float(parts[0])
            minutes = float(parts[1]) if len(parts) > 1 else 0
            return hours + (minutes / 60)
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


def get_cols(df: pd.DataFrame) -> dict:
    return {
        "status": find_column(df, ["Trip Status", "TRIP STATUS", "trip_status"]),
        "remark": find_column(df, ["Trip Remark", "TRIP REMARK"]),
        "vehicle": find_column(df, ["Vehicle", "VEHICLE_NO", "Vehicle No", "vehicle"]),
        "transporter": find_column(df, ["Transporter Name", "Transporter", "TRANSPORTER_NAME", "Integration Transporter Name"]),
        "customer": find_column(df, ["Customer Name", "Customer", "CUSTOMER_NAME"]),
        "destination": find_column(df, ["Destination", "Ship To", "ShipTo", "Location"]),
        "duration": find_column(df, ["Onwards Travel Duration(Hrs.Mins)", "Onward Duration", "Onwards Travel Duration", "onward duration"]),
        "geofence": find_column(df, ["Geofence Hit Miss Status", "Geofence Hit/Miss Status", "GEOFENCE HIT MISS STATUS"]),
        "accountability": find_column(df, ["Accountability", "ACCOUNTABILITY"]),
        "di": find_column(df, ["DI No", "DI_NO", "DI Number"])
    }


def prepare_trip_df(df: pd.DataFrame) -> pd.DataFrame:
    data = clean_column_names(df.copy())
    cols = get_cols(data)

    if cols["duration"]:
        data["Onward_Duration_Hours"] = data[cols["duration"]].apply(duration_to_hours)
    else:
        data["Onward_Duration_Hours"] = 0.0

    if cols["status"]:
        data["Delay_Issue"] = (
            (data[cols["status"]].astype(str).str.upper() == "TRIP MONITORED") &
            (data["Onward_Duration_Hours"] > 10)
        )
        data["NDD_Issue"] = data[cols["status"]].astype(str).str.upper().str.contains("NDD", na=False)
    else:
        data["Delay_Issue"] = False
        data["NDD_Issue"] = False

    if cols["remark"]:
        data["NDD_Issue"] = data["NDD_Issue"] | data[cols["remark"]].astype(str).str.upper().str.contains("NDD", na=False)
        remark_issue = data[cols["remark"]].astype(str).str.upper().str.contains("DATA LOSS|EPF", na=False, regex=True)
    else:
        remark_issue = False

    if cols["geofence"]:
        data["Geofence_Miss_Issue"] = data[cols["geofence"]].astype(str).str.upper().str.contains("MISS", na=False)
    else:
        data["Geofence_Miss_Issue"] = False

    if cols["accountability"]:
        acc_issue = data[cols["accountability"]].astype(str).str.upper().str.contains("DATA LOSS|EPF", na=False, regex=True)
    else:
        acc_issue = False

    data["Data_Loss_Issue"] = remark_issue | acc_issue
    data["Issue_Count"] = data[["Delay_Issue", "Geofence_Miss_Issue", "Data_Loss_Issue", "NDD_Issue"]].sum(axis=1)
    return data


def group_count(df: pd.DataFrame, group_col: str, filter_col: Optional[str] = None) -> pd.DataFrame:
    data = df.copy()
    if filter_col:
        data = data[data[filter_col] == True]
    if group_col is None or data.empty:
        return pd.DataFrame()
    return data.groupby(group_col).size().reset_index(name="Count").sort_values("Count", ascending=False)


def risk_by(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col is None:
        return pd.DataFrame()
    rows = []
    for value in sorted(df[group_col].dropna().unique()):
        sub = df[df[group_col] == value]
        delayed = int(sub["Delay_Issue"].sum())
        geofence = int(sub["Geofence_Miss_Issue"].sum())
        data_loss = int(sub["Data_Loss_Issue"].sum())
        ndd = int(sub["NDD_Issue"].sum())
        total = len(sub)
        risk_score = delayed * 3 + geofence * 2 + data_loss * 3 + ndd * 2
        bucket = "High Risk" if risk_score >= 10 else ("Medium Risk" if risk_score >= 5 else "Low Risk")
        rows.append({
            "Name": value,
            "Total Trips": total,
            "Delayed": delayed,
            "Geofence Miss": geofence,
            "Data Loss/EPF": data_loss,
            "NDD": ndd,
            "Risk Score": risk_score,
            "Risk Bucket": bucket
        })
    return pd.DataFrame(rows).sort_values("Risk Score", ascending=False)


def quality_score(df: pd.DataFrame) -> dict:
    total = len(df)
    if total == 0:
        return {"score": 0, "bucket": "No Data"}
    delayed = int(df["Delay_Issue"].sum())
    geofence = int(df["Geofence_Miss_Issue"].sum())
    data_loss = int(df["Data_Loss_Issue"].sum())
    ndd = int(df["NDD_Issue"].sum())
    issue_count = delayed + geofence + data_loss + ndd
    score = max(0, round(100 - ((issue_count / total) * 100), 2))
    bucket = "Good" if score >= 85 else ("Average" if score >= 60 else "Poor")
    return {"score": score, "bucket": bucket, "total": total, "delayed": delayed, "geofence_miss": geofence, "data_loss": data_loss, "ndd": ndd}
