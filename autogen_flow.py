
# AutoGen TaskGroup 기반 단계 간 에이전트 협업 흐름
def autogen_transition_pipeline(stage, user_input):
    flow = {
        "precontemplation": "mi_agent",
        "contemplation": "cbt_agent",
        "preparation": "ppi_agent",
        "action": "action_agent"
    }
    # TaskGroup 방식의 다음 에이전트 라우팅 (단순 예시)
    return flow.get(stage, "empathy_agent")
