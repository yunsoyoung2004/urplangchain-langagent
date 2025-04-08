
# LangGraph 기반 변화 단계 흐름 그래프 예시
def get_flow_graph():
    return {
        "precontemplation": ["contemplation"],
        "contemplation": ["preparation", "precontemplation"],
        "preparation": ["action", "contemplation"],
        "action": ["maintenance", "preparation"]
    }
