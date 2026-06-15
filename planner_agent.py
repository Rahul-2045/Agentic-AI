def create_plan(user_goal: str) -> list[str]:
    goal = user_goal.lower()
    plan = ["Understand user goal"]
    if "report" in goal or "excel" in goal:
        plan.append("Create Excel or RCA report")
    if "email" in goal or "mail" in goal:
        plan.append("Draft email")
    if "sop" in goal or "why" in goal or "action" in goal:
        plan.append("Search SOP knowledge base")
    plan.append("Run relevant trip monitoring tools")
    plan.append("Summarize final answer")
    return plan
