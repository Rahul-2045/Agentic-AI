from datetime import datetime

class SessionMemory:
    def __init__(self):
        self.events = []

    def add(self, role: str, content: str):
        self.events.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "role": role,
            "content": content
        })

    def summary(self) -> str:
        if not self.events:
            return "No memory yet."
        return "\n".join([f"{e['time']} | {e['role']}: {e['content']}" for e in self.events[-10:]])
