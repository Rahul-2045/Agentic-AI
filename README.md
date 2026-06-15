# Agentic Trip AI 90%

This is a stronger Agentic AI project using:

- Gemini LLM
- LangChain Agent
- Planner Agent
- Supervisor Agent
- Session Memory
- KPI Tools
- Report Generation Tool
- Email Drafting Tool
- RAG/SOP Search Tool
- Streamlit UI

## Run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Create `.env`:

```text
GOOGLE_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-2.5-flash
```

## Demo Questions

```text
Analyze trip monitoring quality and transporter risk
Create Excel report
Draft client email
Search SOP for geofence miss action
Find multi issue cases
Show delayed trips by transporter
Create RCA report
```
