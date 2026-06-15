def supervisor_review(answer: str) -> str:
    if not answer or len(answer.strip()) < 10:
        return "Supervisor Review: Output is too short. Please rerun analysis."
    return "Supervisor Review: Output generated successfully. Please verify before business use."
