from langchain_core.tools import tool
from tools.utils import prepare_trip_df, quality_score

CURRENT_DF = None

def set_dataframe(df):
    global CURRENT_DF
    CURRENT_DF = df

def get_df():
    if CURRENT_DF is None:
        raise ValueError("No trip data uploaded.")
    return prepare_trip_df(CURRENT_DF)

@tool
def draft_client_email_tool() -> str:
    """Draft client-ready email for trip monitoring summary."""
    df = get_df()
    q = quality_score(df)
    return f"""Dear Team,

Please find below the trip monitoring summary.

Total Trips Reviewed: {q['total']}
Trip Quality Score: {q['score']}%
Quality Bucket: {q['bucket']}

Key Observations:
- Delayed Trips: {q['delayed']}
- Geofence Miss Trips: {q['geofence_miss']}
- Data Loss / EPF Trips: {q['data_loss']}
- NDD Trips: {q['ndd']}

Recommended Actions:
- Validate delayed trips and identify route/customer/transporter reasons.
- Review geofence miss cases and correct master/geofence mapping if required.
- Check GPS/device health for data loss or EPF cases.
- Validate NDD cases for actual destination reach and event capture.

Request you to kindly review and take necessary action.

Thanks & Regards,
Trip Monitoring Agent
"""

@tool
def draft_internal_escalation_tool() -> str:
    """Draft internal escalation note for operations/service team."""
    df = get_df()
    q = quality_score(df)
    return f"""Dear Team,

Please review the below monitoring exceptions on priority:

Delayed Trips: {q['delayed']}
Geofence Miss: {q['geofence_miss']}
Data Loss / EPF: {q['data_loss']}
NDD: {q['ndd']}

Requested action:
1. Operations team to validate trip delays.
2. Service team to check device/data loss cases.
3. Master team to validate geofence mapping.
4. Monitoring team to track closure.

Regards,
Trip Monitoring Agent
"""

def get_email_tools():
    return [draft_client_email_tool, draft_internal_escalation_tool]
