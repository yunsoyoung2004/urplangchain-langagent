
def classify_stage(user_input: str) -> str:
    if "생각 안 해봤어요" in user_input:
        return "precontemplation"
    elif "고민 중이에요" in user_input:
        return "contemplation"
    elif "준비하고 있어요" in user_input:
        return "preparation"
    elif "실천하고 있어요" in user_input:
        return "action"
    return "empathy"
